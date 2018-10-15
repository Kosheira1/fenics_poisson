from fenics import *

from Parameter_class import Device
from coordinate_class import coordinate_data
from setup_domains import *
import matplotlib.pyplot as plt

number_FE = 2  # [] Number of single-domain FE materials

# Initialize rectangle coordinates
FE_coords = []
DE_coords = []
SEM_coords = []
ME_coords = []

# Build device from rectangles, units in um
FE_coords.append([0.0, 0.007, 0.045, 0.002])
FE_coords.append([0.045, 0.007, 0.09, 0.002])
DE_coords.append([0.0, 0.002, 0.09, 0.0])
SEM_coords.append([0.0, 0.0, 0.09, -0.03])
SEM_coords.append([-0.05, -0.03, 0.14, -0.07])
SEM_coords.append([0.09, 0.0, 0.14, -0.03])
SEM_coords.append([-0.05, 0.0, 0.0, -0.03])
ME_coords.append([0.0, 0.012, 0.09, 0.007])
ME_coords.append([-0.05, -0.07, 0.14, -0.08])
ME_coords.append([-0.055, 0, -0.05, -0.03])
ME_coords.append([0.14, 0, 0.145, -0.03])

# Geometry and domains
geom_device = coordinate_data(FE_coords, DE_coords, SEM_coords, ME_coords)
domains = setup_domains(geom_device)

# Read mesh from file
mesh = Mesh('ncfet_complete.xml')

# Build Mesh functions of device that are later used in FEM expression.
NCFET = Device(domains, geom_device, mesh)
NCFET.assign_labels()
NCFET.compute_FE_midpointlist()

file = File('ncfet_domains.pvd')
file << NCFET.materials

plot(mesh)
plot(NCFET.materials)


# Prepare Mesh File for FE-Poisson Solver
FEPOISSON_subdomains = MeshFunction('size_t', mesh, 2)

# Compute number of non-metal layers that are not needed for FE-Poisson solver

num_nonmet = len(NCFET.domains['Ferroelectric']) + len(NCFET.domains['Dielectric']) + len(NCFET.domains['Semiconductor'])


FEPOISSON_subdomains.array()[NCFET.materials.array() < num_nonmet] = 100

submesh_FEPSOLVER = SubMesh(mesh, FEPOISSON_subdomains, 100)

file = File('ncfet_FEP.xml')
file << submesh_FEPSOLVER

# Prepare Mesh File for ViennaSHE input, requires a pvd file with all subdomains individually listed

# Place all FE and DE layers together to one.
ViennaSHE_file = File('NCFET_VIENNA.pvd')
combined_subdomains = MeshFunction('size_t', mesh, 2)

num_gatestack = len(NCFET.domains['Ferroelectric']) + len(NCFET.domains['Dielectric'])

combined_subdomains.array()[NCFET.materials.array() < num_gatestack] = 1

submesh_di = SubMesh(mesh, combined_subdomains, 1)

ViennaSHE_file << submesh_di

# Undoped Semiconductor layer
combined_subdomains.array()[NCFET.materials.array() == num_gatestack] = 2
combined_subdomains.array()[NCFET.materials.array() == num_gatestack + 1] = 2
submesh_sem = SubMesh(mesh, combined_subdomains, 2)

ViennaSHE_file << submesh_sem

# Semiconductor layers
for idx, SEM in enumerate(NCFET.domains['Semiconductor']):
    if(idx > 1):
        combined_subdomains.array()[NCFET.materials.array() == num_gatestack + idx] = 1 + idx
        submesh_semtemp = SubMesh(mesh, combined_subdomains, 1 + idx)
        ViennaSHE_file << submesh_semtemp


# Metal Layers
for idx, ME in enumerate(NCFET.domains['Metal']):
    combined_subdomains.array()[NCFET.materials.array() == num_nonmet + idx] = 1 + len(NCFET.domains['Semiconductor']) + idx
    submesh_metemp = SubMesh(mesh, combined_subdomains, 1 + len(NCFET.domains['Semiconductor']) + idx)
    ViennaSHE_file << submesh_metemp

plot(combined_subdomains)
plt.show()
