import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Grid parameters
Nz, Nr = 100, 100
z = np.linspace(0, 1, Nz)
r = np.linspace(0, 1, Nr)
dz, dr = z[1] - z[0], r[1] - r[0]

# Domain decomposition in z-direction
local_Nz = Nz // size
local_z_start = rank * local_Nz

# Create local grid
local_z = z[local_z_start: local_z_start + local_Nz]

# Initialize psi (poloidal flux function)
psi = np.zeros((local_Nz, Nr))

# Boundary condition
if rank == 0:
    psi[0, :] = 0  # Some boundary value
elif rank == size - 1:
    psi[-1, :] = 0  # Some boundary value

# Main iterative loop for relaxation method
for _ in range(1000):
    
    # Exchange data with neighboring processes
    if rank > 0:
        comm.send(psi[0, :], dest=rank-1, tag=0)
        psi[-1, :] = comm.recv(source=rank-1, tag=1)
    if rank < size - 1:
        psi[0, :] = comm.recv(source=rank+1, tag=0)
        comm.send(psi[-1, :], dest=rank+1, tag=1)

    # Compute new psi using finite differences (relaxation)
    for i in range(1, local_Nz-1):
        for j in range(1, Nr-1):
            laplacian = (psi[i+1, j] + psi[i-1, j] - 2*psi[i, j]) / dz**2 + \
                        (psi[i, j+1] + psi[i, j-1] - 2*psi[i, j]) / dr**2
            psi[i, j] = psi[i, j] + 0.25 * laplacian  # Relaxation factor 0.25

# Gather results at rank 0
gathered_psi = None
if rank == 0:
    gathered_psi = np.empty([Nz, Nr], dtype='d')

comm.Gather(sendbuf=psi, recvbuf=gathered_psi, root=0)

# Rank 0 can now visualize the solution using any plotting tool, for example, matplotlib
if rank == 0:
    import matplotlib.pyplot as plt
    plt.imshow(gathered_psi, extent=(0, 1, 0, 1))
    plt.colorbar()
    plt.title("Solution of Grad-Shafranov")
    plt.show()
