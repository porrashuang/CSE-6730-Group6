import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_edge_land_mask(rows, cols):
    # Generate an ocean map with land around the edges
    land_mask = np.zeros((rows, cols), dtype=bool)

    # Set the edges as land
    land_mask[0, :] = True
    land_mask[-1, :] = True
    land_mask[:, 0] = True
    land_mask[:, -1] = True
    return land_mask

def generate_random_inputs(rows, cols):
    # Generate a random ocean map with initial concentration values
    # ocean_map = np.zeros((rows, cols))
    ocean_map = np.zeros((rows, cols))
    
    # Generate land mask with land on the edges only
    land_mask = generate_edge_land_mask(rows, cols)
    
    # Generate S array for pollutant sources/treatment (random values as a simulation)
    # Assign negative values to simulate waste processing (cleaning up) at some random locations
    S = np.zeros((rows, cols))
    # for _ in range(10):  # Adding 10 points for waste processing
    #     i, j = np.random.randint(1, rows-1), np.random.randint(1, cols-1)
    #     S[i, j] = -np.random.rand() * 0.5  # Negative value for cleanup
    
    # Assign positive values to simulate pollutant sources at some random locations
    for _ in range(4):  # Adding 10 points for pollutant sources
        i, j = np.random.randint(1, rows-1), np.random.randint(1, cols-1)
        S[i, j] = 100  # Positive value for pollution
    
    return ocean_map, land_mask, S

def generate_random_Dx_Dy(rows, cols):
    Dx = np.random.rand(rows, cols) * 0
    Dy = np.random.rand(rows, cols) * 0
    u = np.random.randn(rows, cols) * 10
    v = np.random.randn(rows, cols) * 10
    return Dx, Dy, u, v

def update_concentration(ocean_map, land_mask, S, Dx, Dy, u, v, dt, dx, dy, iterations):
    # Get the shape of the map
    rows, cols = ocean_map.shape
    
    # Initialize a copy of the ocean_map to store updates
    updated_map = np.copy(ocean_map)
    maps_over_time = [np.copy(ocean_map)]
    
    for _ in range(iterations):
        # Iterate through each cell in the ocean_map
        print(ocean_map.sum())
        for i in range(rows):
            for j in range(cols):
                # Skip updating if it's a land cell
                if land_mask[i, j]:
                    continue
                
                # Get the current concentration value
                C_current = ocean_map[i, j]
                
                # Calculate the contribution from the center cell
                C_new = (1 - 2 * dt * Dx[i, j] / dx**2 - 2 * dt * Dy[i,j] / dy**2 - ((abs(u[i,j]) if not land_mask[i + (1 if u[i, j] > 0 else -1),j] else 0)  * dt / dx) - ((abs(v[i,j]) if not land_mask[i,j + (1 if v[i, j] > 0 else -1)] else 0) * dt / dy)) * C_current + (S[i, j] * dt if _ < iterations / 2 else 0)
                # Calculate contributions from neighboring cells, ensuring boundary conditions
                # if (S[i, j] > 0) :
                #     print(S[i, j])
                # x + dx
                if i + 1 < rows and not land_mask[i + 1, j]:
                    C_new += (dt / dx) * (Dx[i + 1, j] / dx - (u[i + 1, j] if u[i + 1, j] < 0 else 0)) * ocean_map[i + 1, j]
                    # C_new += (dt / dx) * (Dx_ij / dx - (u_ij / 2)) * ocean_map[i + 1, j]
                # else :
                    # C_new += (dt / dx) * (Dx_ij / dx - (u[i - 1, j] if u[i - 1, j] < 0 else 0)) * C_current
                
                # x - dx
                if i - 1 >= 0 and not land_mask[i - 1, j]:
                    C_new += (dt / dx) * (Dx[i - 1, j] / dx + (u[i - 1, j] if u[i - 1, j] > 0 else 0)) * ocean_map[i - 1, j]
                    # C_new += (dt / dx) * (Dx_ij / dx + (u_ij / 2)) * ocean_map[i - 1, j]
                # else :
                #     C_new += (dt / dx) * (Dx_ij / dx + (u[i - 1, j] if u[i - 1, j] > 0 else 0)) * C_current
                
                # y + dy
                if j + 1 < cols and not land_mask[i, j + 1]:
                    C_new += (dt / dy) * (Dy[i, j + 1] / dy - (v[i, j + 1] if v[i, j + 1] < 0 else 0)) * ocean_map[i, j + 1]
                    # C_new += (dt / dy) * (Dy_ij / dy - (v_ij / 2)) * ocean_map[i, j + 1]
                # else :
                #     C_new += (dt / dy) * (Dy_ij / dy - (v[i, j + 1] if v[i, j + 1] < 0 else 0)) * C_current
                
                # y - dy
                if j - 1 >= 0 and not land_mask[i, j - 1]:
                    C_new += (dt / dy) * (Dy[i, j - 1] / dy + (v[i, j - 1] if v[i, j - 1] > 0 else 0)) * ocean_map[i, j - 1]
                    # C_new += (dt / dy) * (Dy_ij / dy + (v_ij / 2)) * ocean_map[i, j - 1]
                # else :
                #     C_new += (dt / dy) * (Dy_ij / dy + (v[i, j - 1] if v[i, j - 1] > 0 else 0)) * C_current

                # Update the value in the updated_map
                updated_map[i, j] = C_new
        
        # Update the ocean_map for the next iteration
        ocean_map = np.copy(updated_map)
        if _ % 10 == 0:
            maps_over_time.append(np.copy(ocean_map))
    
    return maps_over_time

def animate_concentration(maps_over_time):
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = 'viridis'
    cax = ax.imshow(np.log(maps_over_time[0] + 1), cmap=cmap, origin='lower', vmax=15)
    fig.colorbar(cax, ax=ax, label='Concentration')

    def update(frame):
        cax.set_data(maps_over_time[frame])
        ax.set_title(f'Concentration Map - Iteration {frame + 1}')

    ani = animation.FuncAnimation(fig, update, frames=len(maps_over_time), interval=1)
    plt.show()

# Parameters
rows, cols = 500, 500  # Map size (larger to better visualize)
dt = 0.01  # Time step
dx, dy = 1.0, 1.0  # Spatial step sizes
iterations = 5000  # Number of iterations

# Generate random Dx, Dy, u, v arrays
Dx, Dy, u, v = generate_random_Dx_Dy(rows, cols)

# Generate random inputs with waste processing points
ocean_map, land_mask, S = generate_random_inputs(rows, cols)

# Update concentration and collect maps over time
maps_over_time = update_concentration(ocean_map, land_mask, S, Dx, Dy, u, v, dt, dx, dy, iterations)

# Animate the concentration map
animate_concentration(maps_over_time)
