from fenics import *
from coordinate_class import coordinate_data

# Class that stores all of the domains with their respective Mesh Function representation. Allows to look up material properties, assigns mesh function labels. This class is needed to keep track of all single domain FE materials


class Device():
    def __init__(self, domains, coordinates, mesh):
        self.domains, self.coordinates, self.mesh = domains, coordinates, mesh
        self.materials = MeshFunction('size_t', self.mesh, 2)
        self.permi = MeshFunction('double', self.mesh, 2)
        self.Pol_r = MeshFunction('double', self.mesh, 2)
        # Also store global remnant polarization.
        self.P_G = 10.0 * 1E-14
        # Create Mesh Function for Doping levels
        self.Doping = MeshFunction('double', self.mesh, 2)
        self.FE_midpointlist = []
        self.epsilon_0 = 8.85E-18  # [F*um^-1]s

    def assign_labels(self):
        FE_layers = self.domains['Ferroelectric']
        for idx, FE in enumerate(FE_layers):
            FE.mark(self.materials, idx)

        DE_layers = self.domains['Dielectric']
        for idx, DE in enumerate(DE_layers):
            DE.mark(self.materials, idx + len(FE_layers))

        Sem_layers = self.domains['Semiconductor']
        for idx, SC in enumerate(Sem_layers):
            SC.mark(self.materials, idx + len(FE_layers) + len(DE_layers))

        Metal_layers = self.domains['Metal']
        for idx, ME in enumerate(Metal_layers):
            ME.mark(self.materials, idx + len(FE_layers) + len(DE_layers) + len(Sem_layers))

    def assign_permittivity(self, FE_epsilon_dict, dielectric_perm, sem_relperm):
        # Make sure you assign a dictionary for the Ferroelectric permittivities with the right number of entries.
        FE_layers = self.domains['Ferroelectric']
        for idx, FE in enumerate(FE_layers):
            FE.mark(self.permi, FE_epsilon_dict[idx])

        DE_layers = self.domains['Dielectric']
        for idx, DE in enumerate(DE_layers):
            DE.mark(self.permi, dielectric_perm)

        Sem_layers = self.domains['Semiconductor']
        for idx, SC in enumerate(Sem_layers):
            SC.mark(self.permi, sem_relperm)

        # Using dummy value for permittivity in metal, potential should always be known.
        Metal_layers = self.domains['Metal']
        for idx, ME in enumerate(Metal_layers):
            ME.mark(self.permi, 1.0)

    def assign_remnantpol(self, P_r):
        # Assuming each single domain FE materials follow the same S-curve we can assign a global remnant polarization to FE materials.
        for key, value in self.domains.items():
            if (key == 'Ferroelectric'):
                for FE in value:
                    FE.mark(self.Pol_r, P_r)

            else:
                for LY in value:
                    LY.mark(self.Pol_r, 0.0)

        self.P_G = P_r

    def update_permittivity(self, FE_epsilon_dict):
        # Only FE materials need to be updated.
        FE_layers = self.domains['Ferroelectric']
        for idx, FE in enumerate(FE_layers):
            FE.mark(self.permi, FE_epsilon_dict[idx])

    def material_cellindex(self, num):
        marked_cells = SubsetIterator(self.materials, num)
        for cells in marked_cells:
            big_index = cells.index()
            break

        return big_index

    def compute_FE_midpointlist(self):
        self.FE_midpointlist = []
        FE_coords = self.coordinates.data_dict['Ferroelectric']
        for idx, vals in enumerate(FE_coords):
            xval = 0.5 * (vals[0] + vals[2])
            yval = 0.5 * (vals[1] + vals[3])
            point = (xval, yval)
            self.FE_midpointlist.append(point)
