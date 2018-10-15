
class coordinate_data():
    def __init__(self, FE_coordinates, DE_coordinates, SEM_coordinates, MET_coordinates):
        self.FE_coords, self.DE_coords, self.SEM_coords, self.MET_coords = FE_coordinates, DE_coordinates, SEM_coordinates, MET_coordinates
        keys = ['Ferroelectric', 'Dielectric', 'Semiconductor', 'Metal']
        self.data_dict = dict([(key, []) for key in keys])

        for co in self.FE_coords:
            self.data_dict['Ferroelectric'].append(co)
        for co in self.DE_coords:
            self.data_dict['Dielectric'].append(co)
        for co in self.SEM_coords:
            self.data_dict['Semiconductor'].append(co)
        for co in self.MET_coords:
            self.data_dict['Metal'].append(co)
