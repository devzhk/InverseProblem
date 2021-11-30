import numpy as np
from scipy import integrate
import torch


def int_R(Ds_t, theta, time, backend='scipy'):
    '''
    Function that computes the spatial integral over R in equation (4) for given parameter values theta at a specific time index t
    Args: 
        - Ds    
    '''
    r, t_a, Ds_c, Asigma = theta
    if backend == 'scipy':
        y = np.exp((Ds_t-Ds_c)/Asigma)
        denom = 1 + integrate.cumtrapz(y, time, initial=0, axis=2) / t_a
        R = r*y/denom
        R_avg = 0.25 * np.sum(R, axis=(0, 1))
    elif backend == 'pytorch':
        y = torch.exp((Ds_t-Ds_c)/Asigma)
        denom = 1 + cumtrapz(y, time, axis=2) / t_a
        R = r * y / denom
        R_avg = 0.25 * torch.sum(R, axis=(0, 1))
    return R_avg


def Get_mask(DSt, thresh):
    # Dst has size X*Y*t

    avg_ds = np.mean(DSt, axis=2)
    mask = np.ones_like(avg_ds)
    mask[avg_ds < thresh] = 0

    return mask


def tupleset(t, i, value):
    lst = list(t)
    lst[i] = value
    return tuple(lst)


def cumtrapz(
    y: torch.Tensor,
    x: np.ndarray,
    axis: int = -1,
) -> torch.Tensor:

    if x.ndim == 1:
        d = np.diff(x)
        # reshape to correct shape
        shape = [1] * y.ndim
        shape[axis] = -1
        d = d.reshape(shape)
    elif len(x.shape) != len(y.shape):
        raise ValueError("If given, shape of x must be 1-D or the "
                         "same as y.")
    else:
        d = np.diff(x, axis=axis)

    if d.shape[axis] != y.shape[axis] - 1:
        raise ValueError("If given, length of x along axis must be the "
                         "same as y.")

    d = torch.from_numpy(d)

    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = torch.cumsum(d * (y[slice1] + y[slice2]) / 2.0, dim=axis)

    shape = list(res.shape)
    shape[axis] = 1
    res = torch.cat([torch.zeros(shape, dtype=res.dtype), res], dim=axis)

    return res
