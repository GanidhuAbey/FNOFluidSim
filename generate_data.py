import warp as wp
import numpy as np
import h5py
from pathlib import Path
import argparse

from fluid_sim import StableFluidsSimulator

def generate_training_data(
    grid_size=128,
    domain_size=5.0,
    num_trajectories=100,
    trajectory_length=200,
    dt=0.01,
    output_file="fluid_training_data.h5"
):
    """
    Generate training data by running multiple fluid simulation trajectories
    with different initial conditions and forcing patterns.
    
    Args:
        grid_size: Size of the simulation grid
        num_trajectories: Number of different simulation runs
        trajectory_length: Number of timesteps per trajectory
        dt: Time step size
        output_file: Output HDF5 file path
    """
    
    print(f"Generating {num_trajectories} trajectories with {trajectory_length} steps each...")
    print(f"Grid size: {grid_size}x{grid_size}")
    
    # Data storage
    # For FNO, we typically store: (batch, time, channels, height, width)
    # Channels: [u, v, density, pressure]
    all_states = []
    all_forces = []
    
    for traj_idx in range(num_trajectories):
        print(f"\nTrajectory {traj_idx + 1}/{num_trajectories}")
        
        # Initialize simulator
        sim = StableFluidsSimulator(grid_size, domain_size)
        
        # Random initial conditions
        init_type = traj_idx % 4
        
        if init_type == 0:
            # Single source in center with random velocity
            cx, cy = grid_size // 2, grid_size // 2
            offset_x = np.random.randint(-10, 10)
            offset_y = np.random.randint(-10, 10)
            cx += offset_x
            cy += offset_y
        elif init_type == 1:
            # Source at random point in the center area.
            cx = np.random.randint(grid_size // 4, 3 * grid_size // 4)
            cy = np.random.randint(grid_size // 4, 3 * grid_size // 4)
        elif init_type == 2:
            # Edge source
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'bottom':
                cx = np.random.randint(0, grid_size - 1)
                cy = 0
            elif edge == 'top':
                cx = np.random.randint(0, grid_size - 1)
                cy = grid_size - 1
            elif edge == 'left':
                cx = 0
                cy = np.random.randint(0, grid_size - 1)
            else:
                cx = grid_size - 1
                cy = np.random.randint(0, grid_size - 1)
        else:
            # Random position
            cx = np.random.randint(0, grid_size - 1)
            cy = np.random.randint(0, grid_size - 1)
        
        # Random forcing parameters
        force_strength = np.random.uniform(3.0, 8.0)
        force_freq = np.random.uniform(0.02, 0.1)
        density_strength = np.random.uniform(300.0, 700.0)
        
        states_traj = []
        forces_traj = []
        
        for step in range(trajectory_length):
            # Apply forcing
            density_source = np.zeros((grid_size, grid_size), dtype=np.float32)
            vel_x_source = np.zeros((grid_size, grid_size), dtype=np.float32)
            vel_y_source = np.zeros((grid_size, grid_size), dtype=np.float32)
            
            # Add source with some variation
            radius = np.random.randint(1, 3)
            density_source[cx-radius:cx+radius+1, cy-radius:cy+radius+1] = density_strength
            vel_y_source[cx-radius:cx+radius+1, cy-radius:cy+radius+1] = force_strength
            vel_x_source[cx-radius:cx+radius+1, cy-radius:cy+radius+1] = np.sin(step * force_freq) * 2.0
            
            sim.add_density_source(wp.from_numpy(density_source, dtype=wp.float32))
            sim.add_velocity_source(
                wp.from_numpy(vel_x_source, dtype=wp.float32),
                wp.from_numpy(vel_y_source, dtype=wp.float32)
            )
            
            # Step simulation
            sim.step()
            
            # Collect state (convert to numpy)
            u = sim.u.numpy()
            v = sim.v.numpy()
            density = sim.density.numpy()
            pressure = sim.pressure.numpy()
            
            # Stack as channels: [u, v, density, pressure]
            state = np.stack([u, v, density, pressure], axis=0)  # (4, H, W)
            
            # Stack forces as well
            force = np.stack([vel_x_source, vel_y_source, density_source], axis=0)  # (3, H, W)
            
            states_traj.append(state)
            forces_traj.append(force)
            
            if (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{trajectory_length}")
        
        all_states.append(np.stack(states_traj))  # (T, 4, H, W)
        all_forces.append(np.stack(forces_traj))  # (T, 3, H, W)
    
    # Convert to numpy arrays
    all_states = np.array(all_states)  # (N_traj, T, 4, H, W)
    all_forces = np.array(all_forces)  # (N_traj, T, 3, H, W)
    
    print(f"\nData shape: {all_states.shape}")
    print(f"Forces shape: {all_forces.shape}")
    
    # Save to HDF5
    print(f"\nSaving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('states', data=all_states, compression='gzip')
        f.create_dataset('forces', data=all_forces, compression='gzip')
        f.attrs['grid_size'] = grid_size
        f.attrs['domain_size'] = domain_size
        f.attrs['dt'] = dt
        f.attrs['num_trajectories'] = num_trajectories
        f.attrs['trajectory_length'] = trajectory_length
    
    print("Done!")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate fluid simulation training data')
    parser.add_argument('--grid_size', type=int, default=32, help='Grid size')
    parser.add_argument('--num_traj', type=int, default=10, help='Number of trajectories')
    parser.add_argument('--traj_len', type=int, default=200, help='Trajectory length')
    parser.add_argument('--output', type=str, default='fluid_training_data.h5', help='Output file')
    
    args = parser.parse_args()
    
    generate_training_data(
        grid_size=args.grid_size,
        num_trajectories=args.num_traj,
        trajectory_length=args.traj_len,
        output_file=args.output
    )