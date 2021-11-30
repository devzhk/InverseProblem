# Array and Dataformating
import numpy as np
import pandas as pd
import h5py
from .Functions_spatial import Get_mask


def meanpool(data):
    '''
    coarse
    '''
    pass


def load_data(datapath, catpath, fac=1):
    '''
    Load coulomb stress data 
    smoothed_coulomb_stree: 
    '''
    # -------------Accumulative events------------------------------------------------------------------------
    cat = pd.read_csv(catpath, sep='\t',
                      names=['Date', 'Time', 'Mag', 'Depth', 'RDX', 'RDY'])
    # creating the dates and times
    cat['DateTime'] = pd.to_datetime(
        cat.Date + ' ' + cat.Time, format='%d %m %Y %H %M %S')
    temp_index = pd.DatetimeIndex(cat.DateTime)
    cat['decDate'] = temp_index.year + temp_index.month/12 + (temp_index.day + (
        temp_index.hour + (temp_index.minute + (temp_index.second/60)/60)/24))/365.25
    mc = 1.5
    cat = cat[cat.Mag > mc]
    cat = cat.reset_index()
    # yearly ground truth
    Yearly_R0 = np.zeros((2018-1992+1,))
    for i in range(1, Yearly_R0.size):
        year = i+1992
        start = np.min(np.where(cat['decDate'] >= year))
        end = np.max(np.where(cat['decDate'] < year+1))

        Yearly_R0[i] = cat['index'][end+1]-cat['index'][start]

# --------------------------coulomb stress-------------------------------------------------------
    f = h5py.File(datapath, 'r')
    smoothed_coulomb_stress = f['smoothed_coulomb_stress']
    y0, y1, dy = 1956, 2019, 12
    dates = np.linspace(y0, y1+1, (y1-y0)*dy+1)
    DX, DY, a = np.shape(smoothed_coulomb_stress)
    Dx_new = DX//fac
    Dy_new = DY//fac

    data_counts, years = np.histogram(
        dates, np.linspace(1956, 2019, (2019-1956)+1))
    Yearly_Ds = np.zeros((Dx_new, Dy_new, len(years)-1))
    stop_i = 0

    for ti, t in enumerate(data_counts):
        stop_i += t
        Yearly_Ds[:, :, ti] = np.sum(
            smoothed_coulomb_stress[:, :, stop_i - t:stop_i], axis=2)/(t)

    return Yearly_Ds, Yearly_R0
