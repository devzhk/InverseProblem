import math
import scipy
from tqdm import tqdm

# Array and Dataformating
import numpy as np
import h5py
import pandas as pd


def run():
    pass


if __name__ == '__main__':
    f = h5py.File('data/Groningen_Data_Example.hdf5', 'r')
    # keys() shows the contents of the file

    # extract all the data.
    # note that these hdf4 datasets should behave exactly as np arrays

    # the maximum coulomb stress smoothed with a 6km Gaussian2D kernel calculated 10m below the reservoir.
    smoothed_coulomb_stress = f['smoothed_coulomb_stress']
    cat = pd.read_csv('data/catalog2.txt', sep='\t',
                      names=['Date', 'Time', 'Mag', 'Depth', 'RDX', 'RDY'])
    # creating the dates and times
    cat['DateTime'] = pd.to_datetime(
        cat.Date + ' ' + cat.Time, format='%d %m %Y %H %M %S')
    temp_index = pd.DatetimeIndex(cat.DateTime)
    cat['decDate'] = temp_index.year + temp_index.month/12 + (temp_index.day + (
        temp_index.hour + (temp_index.minute + (temp_index.second/60)/60)/24))/365.25
    # filtering the catalog to magnitudes > mc
    mc = 1.5
    cat = cat[cat.Mag > mc]
    cat = cat.reset_index()
    # The simulation given in this example is from 1956 to 2019 for a total of 756 months
    y0, y1, dy = 1956, 2019, 12

    dates = np.linspace(y0, y1+1, (y1-y0)*dy+1)

    DS = smoothed_coulomb_stress
    dates_R = np.array(cat.decDate)
    print(np.linspace(1992, 2019, (2019-1992)+1))
    R0, years = np.histogram(dates_R, np.linspace(1992, 2019, (2019-1992)+1))
    print(len(dates))
    print(len(DS))
