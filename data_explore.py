import numpy as np
import h5py


if __name__ == '__main__':
    f = h5py.File('./Groningen_Data_Example.hdf5', 'r')
    # keys() shows the contents of the file

    # extract all the data.
    # note that these hdf4 datasets should behave exactly as np arrays

    # The reservoir data
    x_res = f['x_res']
    y_res = f['y_res']
    reservoir_outline = f['res_outline']
    x_outline = f['x_outline']
    y_outline = f['y_outline']
    # The original well extraction data
    extractions = f['extraction_data']
    wells_locations = f['wells_locations_list']
    wells_names = f['wells_names']
    # The computed pressures from the linearized pressure diffusion model
    pressures = f['pressures_list']
    x_mesh = f['x_mesh']  # The meshes used for the pressure diffusion model
    y_mesh = f['y_mesh']  # The meshes used for the pressure diffusion model
    # The computed deformations and coulomb stress change from the BorisBlocks model
    deformations = f['deformations_list']
    # The meshes used for the BorisBlocks model
    x_reservoir_mesh = f['x_reservoir_mesh']
    # The meshes used for the BorisBlocks model
    y_reservoir_mesh = f['y_reservoir_mesh']
    max_coulomb_stress = f['max_coulomb_stress']
    # the maximum coulomb stress smoothed with a 6km Gaussian2D kernel calculated 10m below the reservoir.
    smoothed_coulomb_stress = f['smoothed_coulomb_stress']

    # The simulation given in this example is from 1956 to 2019 for a total of 756 months
    y0, y1, dy = 1956, 2019, 12

    dates = np.linspace(y0, y1+1, (y1-y0)*dy+1)
