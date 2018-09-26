from fenics import *

from Parameter_class import Device
from coordinate_class import coordinate_data
from setup_domains import *

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
SEM_coords.append([0.09, 0.0, 0.14, -0.03])
SEM_coords.append([-0.05, 0.0, 0.0, -0.03])
SEM_coords.append([-0.05, -0.03, 0.14, -0.07])

# Geometry and domains
geom_device = coordinate_data(FE_coords, DE_coords, SEM_coords, ME_coords)
domains = setup_domains(geom_device)

# Read mesh from file
mesh = Mesh('ncfet.xml')

# Build Mesh functions of device that are later used in FEM expression.
NCFET = Device(domains, geom_device, mesh)
NCFET.assign_labels()
NCFET.compute_FE_midpointlist()

file = File('ncfet_domains.pvd')
file << NCFET.materials
