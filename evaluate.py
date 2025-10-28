from neuralop.models import FNO
from neuralop.training import Trainer

import torch
import numpy as np
import warp as wp

from fluid_sim import FluidSimulatorBase, FluidRenderer, DOMAIN_SIZE, dt, frame_count

# kernels
from fluid_sim import add_source

class FNOFluidSimulator(FluidSimulatorBase):
    """
    FNO-based fluid simulator that implements the same interface
    as StableFluidsSimulator for drop-in replacement
    """
    
    def __init__(self, model_path, grid_size, domain_size, device='cuda'):
        super().__init__(grid_size, domain_size)
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load trained model        
        self.operator = FNO(
            in_channels=7,      # u, v, p, rho, u_s, v_s, rho_s
            out_channels=4,     # predict next-step u, v, p, rho
            n_modes=(32, 32),   # number of modes in fourier space
            hidden_channels=64, # internal channel width
        )
        
        self.operator.load_checkpoint(model_path, 'best_model')
        self.operator.eval()

        self.preprocessor = torch.load(f'{model_path}/preprocessor.pt')
        
        print(f"FNO model loaded on {self.device}")
        #print(f"Model trained on grid size: {checkpoint.get('grid_size', 'unknown')}")
        
        # Current state (stored as numpy arrays)
        self.u = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.v = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.density = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.pressure = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.temp = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Source accumulators
        self.density_source = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.u_source = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.v_source = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.temp_source = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    def get_state(self):
        """Get current state as numpy arrays"""
        return {
            'u': self.u.copy(),
            'v': self.v.copy(),
            'density': self.density.copy(),
            'pressure': self.pressure.copy(),
            'temp': self.temp.copy()
        }
    
    def set_state(self, state):
        """Set state from numpy arrays"""
        self.u = state['u'].copy()
        self.v = state['v'].copy()
        self.density = state['density'].copy()
        self.pressure = state['pressure'].copy()
        self.temp = state['temp'].copy()
    
    def add_density_source(self, density_source):
        """Add density source (can be warp array or numpy)"""
        if isinstance(density_source, wp.array):
            density_source = density_source.numpy()
        self.density_source += density_source
    
    def add_velocity_source(self, vel_x_source, vel_y_source):
        """Add velocity source (can be warp arrays or numpy)"""
        if isinstance(vel_x_source, wp.array):
            vel_x_source = vel_x_source.numpy()
        if isinstance(vel_y_source, wp.array):
            vel_y_source = vel_y_source.numpy()
        self.u_source += vel_x_source
        self.v_source += vel_y_source
    
    def add_temp_source(self, temp_source):
        """Add temperature source (can be warp array or numpy)"""
        if isinstance(temp_source, wp.array):
            temp_source = temp_source.numpy()
        self.temp_source += temp_source
    
    def step(self, dt):
        """
        Advance simulation by one timestep using FNO
        
        The FNO predicts the next state given:
        - Current state: [u, v, density, pressure]
        """
        with torch.no_grad():
            # wp.launch(add_source, dim=(self.size, self.size),
            #         inputs=[self.u, self.u_source, dt])
            # wp.launch(add_source, dim=(self.size, self.size),
            #         inputs=[self.v, self.v_source, dt])
            # wp.launch(add_source, dim=(self.size, self.size),
            #         inputs=[self.density, self.density_source, dt])
            
            # Prepare input: stack current state and forces
            current_state = np.stack([self.u, self.v, self.density, self.pressure, self.u_source, self.v_source, self.density_source], axis=0)
            #forces = np.stack([self.u_source, self.v_source, self.density_source], axis=0)
                    
            cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
            print(self.density[cx, cy]);

            # Combine into input tensor (1, 4, H, W)
            current_state = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)

            # normalize
            mean = self.preprocessor['mean']
            std = self.preprocessor['std']
            current_state = (current_state - mean) / std
            
            # Predict next state (TODO: may need to unnormalize data?)
            next_state = self.operator.forward(current_state)
        
            # unnormalize output
            next_state = next_state*std[:4] + mean[:4]

            # Convert back to numpy
            next_state = next_state.cpu().numpy()[0]
            
            # Update state
            self.u = next_state[0]
            self.v = next_state[1]
            self.density = next_state[2]
            self.pressure = next_state[3]
        
        # Clear sources for next step
        self.density_source.fill(0)
        self.u_source.fill(0)
        self.v_source.fill(0)
        self.temp_source.fill(0)

ckpt = './ckpt/'


if __name__ == "__main__":
    # Initialize
    GRID_SIZE = 32
    sim = FNOFluidSimulator(ckpt, GRID_SIZE, DOMAIN_SIZE)
    renderer = FluidRenderer(sim, 'fluid_sim_fno.usd', fps=30)

    density_source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    density_source[cx-1:cx+2, cy-1:cy+2] = 500.0
    sim.add_density_source(wp.from_numpy(density_source, dtype=wp.float32))

    # Run simulation
    sim_time = 0.0
    for frame in range(int(300)):
        # Add sources

        temp_source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        temp_source[cx-1:cx+2, cy-1:cy+2] = 100.0
        
        vel_x_source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        vel_y_source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        vel_y_source[cx-1:cx+2, cy-1:cy+2] = 5.0
        vel_x_source[cx-1:cx+2, cy-1:cy+2] = np.sin(frame * 0.05) * 2.0
        
        sim.add_density_source(wp.from_numpy(density_source, dtype=wp.float32))
        #sim.add_temp_source(wp.from_numpy(temp_source, dtype=wp.float32))
        sim.add_velocity_source(wp.from_numpy(vel_x_source, dtype=wp.float32),
                            wp.from_numpy(vel_y_source, dtype=wp.float32))
        
        sim.step(dt)

        if frame % 1 == 0:
            renderer.render_frame(sim_time)
            print(f"Frame {frame}")
            sim_time += dt

    # Save to USD
    renderer.save()


