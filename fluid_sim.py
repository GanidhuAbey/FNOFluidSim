import warp as wp
import numpy as np
import warp.sim
import warp.sim.render

# TODO implement borders?

wp.init()

# Simulation parameters
GRID_SIZE = 128
DOMAIN_SIZE = 5.0
TOTAL_SIM_TIME = 20.0
dt = 0.01
frame_count = TOTAL_SIM_TIME / dt
diffusion = 0.0001
viscosity = 0.00001
amb_temperature = 1.0;
gravity = 0.0  # Downward gravitational acceleration

@wp.kernel
def add_source(field: wp.array2d(dtype=wp.float32), source: wp.array2d(dtype=wp.float32), dt: float):
    """Add source to field"""
    i, j = wp.tid()
    if i < field.shape[0] and j < field.shape[1]:
        field[i, j] += dt * source[i, j]

@wp.kernel
def copy_fields(src: wp.array2d(dtype=wp.float32), dst: wp.array2d(dtype=wp.float32)):
    """Copies source field to destination field"""
    i, j = wp.tid()
    if i < src.shape[0] and j < src.shape[1]:
        dst[i, j] = src[i, j]

@wp.kernel
def boundary(x: wp.array2d(dtype=wp.float32), scale: wp.float32):
    i, j = wp.tid()

    if i == 0:
        x[i, j] = scale * x[i + 1, j]
    
    if i == x.shape[0] - 1:
        x[i, j] = scale * x[i - 1, j]
    
    if j == 0:
        x[i, j] = scale * x[i, j + 1]

    if j == x.shape[1] - 1:
        x[i, j] = scale * x[i, j - 1]

@wp.kernel
def linear_solver(x: wp.array2d(dtype=wp.float32), x_prev: wp.array2d(dtype=wp.float32), x0: wp.array2d(dtype=wp.float32), 
                  a: float, c: float):
    """Linear solver using Gauss-Seidel iteration"""
    i, j = wp.tid()
    if i > 0 and i < x.shape[0] - 1 and j > 0 and j < x.shape[1] - 1:
        # race condition??
        x[i, j] = (x0[i, j] + a * (x_prev[i-1, j] + x_prev[i+1, j] + x_prev[i, j-1] + x_prev[i, j+1])) / c

@wp.kernel
def divergence(vel_x: wp.array2d(dtype=wp.float32), vel_y: wp.array2d(dtype=wp.float32),
               div: wp.array2d(dtype=wp.float32)):
    """Calculate divergence of velocity field"""
    i, j = wp.tid()
    N = float(vel_x.shape[0] - 2)
    if i > 0 and i < vel_x.shape[0] - 1 and j > 0 and j < vel_x.shape[1] - 1:
        h = 1.0 / N
        div[i, j] = -0.5 * h * ((vel_x[i+1, j] - vel_x[i-1, j]) + 
                                (vel_y[i, j+1] - vel_y[i, j-1]))

@wp.kernel
def project_p(p: wp.array2d(dtype=wp.float32), p0: wp.array2d(dtype=wp.float32), div: wp.array2d(dtype=wp.float32)):
    """Solve Poisson equation for pressure"""
    i, j = wp.tid()
    if i > 0 and i < p.shape[0] - 1 and j > 0 and j < p.shape[1] - 1:
        p[i, j] = (div[i, j] + p0[i-1, j] + p0[i+1, j] + p0[i, j-1] + p0[i, j+1]) / 4.0

@wp.kernel
def project_subtract(vel_x: wp.array2d(dtype=wp.float32), vel_y: wp.array2d(dtype=wp.float32),
                     p: wp.array2d(dtype=wp.float32)):
    """Subtract pressure gradient from velocity"""
    i, j = wp.tid()
    N = float(vel_x.shape[0] - 2)
    if i > 0 and i < vel_x.shape[0] - 1 and j > 0 and j < vel_x.shape[1] - 1:
        h = 1.0 / N
        vel_x[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j]) / h
        vel_y[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1]) / h

@wp.kernel
def advect(d: wp.array2d(dtype=wp.float32), d0: wp.array2d(dtype=wp.float32),
           vel_x: wp.array2d(dtype=wp.float32), vel_y: wp.array2d(dtype=wp.float32), dt: float):
    """Semi-Lagrangian advection"""
    i, j = wp.tid()
    N = float(d.shape[0] - 2)
    dt0 = dt * N
    
    if i > 1 and i < d.shape[0] - 2 and j > 1 and j < d.shape[1] - 2:
        # trace the velocity backwards to position at previous time step.
        x = float(i) - dt0 * vel_x[i, j]
        y = float(j) - dt0 * vel_y[i, j]
        
        x = wp.clamp(x, 0.5, N + 0.5)
        y = wp.clamp(y, 0.5, N + 0.5)
        
        i0 = int(x)
        i1 = i0 + 1
        j0 = int(y)
        j1 = j0 + 1
        
        sx = x - float(i0)
        sy = y - float(j0)
        
        # use weighted average of neighboring cells at previous step
        # to compute quantity for current time step.
        d[i, j] = (1.0 - sx) * (1.0 - sy) * d0[i0, j0] + \
                  sx * (1.0 - sy) * d0[i1, j0] + \
                  (1.0 - sx) * sy * d0[i0, j1] + \
                  sx * sy * d0[i1, j1]

@wp.kernel
def apply_forces(vel_y: wp.array2d(dtype=wp.float32), temp: wp.array2d(dtype=wp.float32), base_temp: float, gravity: float, dt: float):
    """Apply forces"""
    i, j = wp.tid()
    k = 1.0
    t_s = 1.0
    if i < vel_y.shape[0] and j < vel_y.shape[1]:
        # gravity and bouyancy
        vel_y[i, j] += (t_s * temp[i, j]) * dt
        pass

@wp.kernel
def update_particle_colors(density: wp.array2d(dtype=wp.float32), 
                           colors: wp.array(dtype=wp.vec3)):
    """Update particle colors based on density"""
    i, j = wp.tid()
    if i < density.shape[0] and j < density.shape[1]:
        idx = i * density.shape[1] + j
        d = wp.min(1.0, density[i, j] / 50.0)
        colors[idx] = d * wp.vec3(1.0, 1.0, 1.0)

class FluidSimulatorBase:
    """Base class for fluid simulators - defines the interface"""
    
    def __init__(self, grid_size, domain_size):
        self.size = grid_size
        self.domain_size = domain_size
        self.cell_size = domain_size / grid_size
    
    def get_state(self):
        """
        Get current simulation state as numpy arrays
        Returns: dict with keys 'u', 'v', 'density', 'pressure', 'temp'
        """
        raise NotImplementedError
    
    def set_state(self, state):
        """
        Set simulation state from numpy arrays
        Args: dict with keys 'u', 'v', 'density', 'pressure', 'temp'
        """
        raise NotImplementedError
    
    def add_density_source(self, density_source):
        """Add density source"""
        raise NotImplementedError
    
    def add_velocity_source(self, vel_x_source, vel_y_source):
        """Add velocity source"""
        raise NotImplementedError
    
    def add_temp_source(self, temp_source):
        """Add temperature source"""
        raise NotImplementedError
    
    def step(self, dt):
        """Advance simulation by one timestep"""
        raise NotImplementedError

class StableFluidsSimulator(FluidSimulatorBase):
    def __init__(self, grid_size=GRID_SIZE, domain_size=DOMAIN_SIZE, diffusion=diffusion, viscosity=viscosity):
        super().__init__(grid_size, domain_size)
        
        self.diffusion = diffusion
        self.viscosity = viscosity

        # Velocity fields
        self.u = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.v = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.u_source = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.v_source = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.u_prev = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.v_prev = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        
        # Density
        self.density = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.density_prev = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.density_source = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        
        # Temperature
        self.temp = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.temp_prev = wp.zeros((grid_size, grid_size), dtype=wp.float32)

        # Pressure and divergence
        self.pressure_prev = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.pressure = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        self.divergence = wp.zeros((grid_size, grid_size), dtype=wp.float32)
        
        # Particle data for rendering
        self.num_particles = grid_size * grid_size
        self.positions = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.colors = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.init_particles()
    
    def get_state(self):
        """Get current state as numpy arrays"""
        return {
            'u': self.u.numpy().copy(),
            'v': self.v.numpy().copy(),
            'density': self.density.numpy().copy(),
            'pressure': self.pressure.numpy().copy(),
            'temp': self.temp.numpy().copy()
        }
    
    def set_state(self, state):
        """Set state from numpy arrays"""
        self.u = wp.from_numpy(state['u'], dtype=wp.float32)
        self.v = wp.from_numpy(state['v'], dtype=wp.float32)
        self.density = wp.from_numpy(state['density'], dtype=wp.float32)
        self.pressure = wp.from_numpy(state['pressure'], dtype=wp.float32)
        self.temp = wp.from_numpy(state['temp'], dtype=wp.float32)

    def init_particles(self):
        """Initialize particle grid"""
        pos_np = np.zeros((self.num_particles, 3), dtype=np.float32)
        for i in range(self.size):
            for j in range(self.size):
                idx = i * self.size + j
                pos_np[idx, 0] = (i + 0.5) * self.cell_size - self.domain_size / 2.0
                pos_np[idx, 1] = (j + 0.5) * self.cell_size - self.domain_size / 2.0
                pos_np[idx, 2] = 0.0
        self.positions = wp.from_numpy(pos_np, dtype=wp.vec3)
    
    def velocity_step(self):
        """Stable fluids velocity step"""
        wp.launch(add_source, dim=(self.size, self.size),
                 inputs=[self.u, self.u_source, dt])
        wp.launch(add_source, dim=(self.size, self.size),
                 inputs=[self.v, self.v_source, dt])

        # Apply external forces
        wp.launch(copy_fields, dim=(self.size, self.size),
                inputs=[self.temp, self.temp_prev]);
        wp.launch(apply_forces, dim=(self.size, self.size),
                 inputs=[self.v, self.temp, amb_temperature, gravity, dt])
        
        # Diffuse u
        wp.launch(copy_fields, dim=(self.size, self.size),
                inputs=[self.u, self.u_source])
        for _ in range(20):
            wp.launch(copy_fields, dim=(self.size, self.size),
                inputs=[self.u, self.u_prev])
            wp.launch(linear_solver, dim=(self.size, self.size),
                     inputs=[self.u, self.u_prev, self.u_source, dt * viscosity * self.cell_size * self.cell_size, 
                            1.0 + 4.0 * dt * viscosity * self.cell_size * self.cell_size])
        
        # Diffuse v
        wp.launch(copy_fields, dim=(self.size, self.size),
                    inputs=[self.v, self.v_source])
        for _ in range(20):
            wp.launch(copy_fields, dim=(self.size, self.size),
                    inputs=[self.v, self.v_prev])
            wp.launch(linear_solver, dim=(self.size, self.size),
                     inputs=[self.v, self.v_prev, self.v_source, dt * viscosity * self.cell_size * self.cell_size,
                            1.0 + 4.0 * dt * viscosity * self.cell_size * self.cell_size])
        
        # Project
        wp.launch(divergence, dim=(self.size, self.size),
                 inputs=[self.u, self.v, self.divergence])
        self.pressure.zero_()
        
        for _ in range(20):
            wp.launch(copy_fields, dim=(self.size, self.size),
                    inputs=[self.pressure, self.pressure_prev])
            wp.launch(linear_solver, dim=(self.size, self.size),
                     inputs=[self.pressure, self.pressure_prev, self.divergence, -self.cell_size * self.cell_size, 4.0])


        # Advect u
        wp.launch(copy_fields, dim=(self.size, self.size),
                 inputs=[self.u, self.u_source])
        wp.launch(advect, dim=(self.size, self.size),
                 inputs=[self.u, self.u_source, self.u_source, self.v_source, dt])
        
        # Advect v
        wp.launch(copy_fields, dim=(self.size, self.size),
                 inputs=[self.v, self.v_source])
        wp.launch(advect, dim=(self.size, self.size),
                 inputs=[self.v, self.v_source, self.u_source, self.v_source, dt])

        # Boundary

        wp.launch(boundary, dim=(self.size, self.size),
                 inputs=[self.pressure, 1.0])
        
        wp.launch(project_subtract, dim=(self.size, self.size),
                 inputs=[self.u, self.v, self.pressure])
        
        wp.launch(boundary, dim=(self.size, self.size),
                 inputs=[self.u, -1.0])
        wp.launch(boundary, dim=(self.size, self.size),
                 inputs=[self.v, -1.0])
        
        self.u_source.zero_()
        self.v_source.zero_()

    
    def density_step(self):
        """Stable fluids density step"""
        wp.launch(add_source, dim=(self.size, self.size),
                 inputs=[self.density, self.density_source, dt])

        wp.launch(copy_fields, dim=(self.size, self.size),
                 inputs=[self.density, self.density_source])
        for _ in range(20):
            wp.launch(copy_fields, dim=(self.size, self.size),
                 inputs=[self.density, self.density_prev])
            wp.launch(linear_solver, dim=(self.size, self.size),
                     inputs=[self.density, self.density_prev, self.density_source, dt * diffusion * self.cell_size * self.cell_size,
                            1.0 + 4.0 * dt * diffusion * self.cell_size * self.cell_size])
    
        wp.launch(copy_fields, dim=(self.size, self.size),
                 inputs=[self.density, self.density_source])
        wp.launch(advect, dim=(self.size, self.size),
                 inputs=[self.density, self.density_source, self.u, self.v, dt])
        
        wp.launch(boundary, dim=(self.size, self.size),
                 inputs=[self.density, 1.0])
        
        self.density_source.zero_()

    def add_temp_source(self, temp_source):
        """Add temperature at source"""
        wp.launch(add_source, dim=(self.size, self.size),
                  inputs=[self.temp_prev, temp_source, 1.0])

    def add_density_source(self, density_source):
        """Add density at source"""
        wp.launch(add_source, dim=(self.size, self.size),
                 inputs=[self.density_source, density_source, 1.0])
    
    def add_velocity_source(self, vel_x_source, vel_y_source):
        """Add velocity at source"""
        wp.launch(add_source, dim=(self.size, self.size),
                 inputs=[self.u_source, vel_x_source, 1.0])
        wp.launch(add_source, dim=(self.size, self.size),
                 inputs=[self.v_source, vel_y_source, 1.0])
    
    def step(self):
        """Advance simulation"""
        self.velocity_step()
        self.density_step()
    

@wp.kernel
def update_particle_colors(density: wp.array2d(dtype=wp.float32), 
                           colors: wp.array(dtype=wp.vec3)):
    """Update particle colors based on density"""
    i, j = wp.tid()
    if i < density.shape[0] and j < density.shape[1]:
        idx = i * density.shape[1] + j
        d = wp.min(1.0, density[i, j] / 50.0)
        colors[idx] = d * wp.vec3(1.0, 1.0, 1.0)

class FluidRenderer:
    """Renderer that works with any FluidSimulatorBase"""
    
    def __init__(self, simulator, output_path="fluid_sim.usd", fps=30):
        self.simulator = simulator
        self.renderer = wp.render.UsdRenderer(output_path, up_axis="z", fps=fps)
        
        # Particle data for rendering
        self.num_particles = simulator.size * simulator.size
        self.positions = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.colors = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.init_particles()
    
    def init_particles(self):
        """Initialize particle grid"""
        pos_np = np.zeros((self.num_particles, 3), dtype=np.float32)
        for i in range(self.simulator.size):
            for j in range(self.simulator.size):
                idx = i * self.simulator.size + j
                pos_np[idx, 0] = (i + 0.5) * self.simulator.cell_size - self.simulator.domain_size / 2.0
                pos_np[idx, 1] = (j + 0.5) * self.simulator.cell_size - self.simulator.domain_size / 2.0
                pos_np[idx, 2] = 0.0
        self.positions = wp.from_numpy(pos_np, dtype=wp.vec3)
    
    def update_colors(self):
        """Update particle colors from current simulator state"""
        state = self.simulator.get_state()
        density_wp = wp.from_numpy(state['density'], dtype=wp.float32)
        wp.launch(update_particle_colors, dim=(self.simulator.size, self.simulator.size),
                 inputs=[density_wp, self.colors])
    
    def render_frame(self, sim_time):
        """Render current frame"""
        self.update_colors()
        self.renderer.begin_frame(sim_time)
        cpu_pos = self.positions.numpy()
        cpu_colors = self.colors.numpy()
        self.renderer.render_points("particles", cpu_pos, 
                                   radius=self.simulator.cell_size, 
                                   colors=cpu_colors, as_spheres=False)
        self.renderer.end_frame()
    
    def save(self):
        """Save rendered output"""
        self.renderer.save()

if __name__ == "__main__":
    # Initialize
    sim = StableFluidsSimulator(GRID_SIZE, DOMAIN_SIZE)
    renderer = FluidRenderer(sim, 'fluid_sim.usd', fps=30)

    density_source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    density_source[cx-1:cx+2, cy-1:cy+2] = 500.0
    sim.add_density_source(wp.from_numpy(density_source, dtype=wp.float32))

    # Run simulation
    sim_time = 0.0
    for frame in range(int(frame_count)):
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
        
        sim.step()

        if frame % 1 == 0:
            renderer.render_frame(sim_time)
            print(f"Frame {frame}")
            sim_time += dt

    # Save to USD
    renderer.save()
    #print(f"Simulation saved to {output_path}")