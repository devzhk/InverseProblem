from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from utils.helper import int_R
from utils.data import load_data


if __name__ == '__main__':
    datapath = 'data/Groningen_Data_Example.hdf5'
    catpath = 'data/catalog2.txt'
    y0, y1, dy = 1956, 2019, 12
    yearly_Ds, yearly_R = load_data(datapath, catpath)

    truth = torch.tensor(yearly_R, dtype=torch.float64)
    time_size = 50
    num_epoch = 1800

    times = np.arange(1956, 2019)
    pred_R = int_R(yearly_Ds[:, :, -time_size:],
                   [5e-6, 8700, 0.17, 0.006], times[-time_size:])

    theta = nn.Parameter(torch.tensor(
        [1.0, 5.0, 1.0, 5.0], dtype=torch.float64))
    scaler = torch.tensor([1e-6, 1000, 1e-2, 1e-3])
    optimizer = Adam([theta], lr=0.02)
    criterion = nn.MSELoss()
    pbar = tqdm(range(num_epoch), dynamic_ncols=True)
    for i in pbar:
        optimizer.zero_grad()
        torch_R = int_R(torch.tensor(yearly_Ds[:, :, -time_size:], dtype=torch.float64),
                        theta * scaler,
                        times[-time_size:],
                        backend='pytorch')
        loss = criterion(torch_R[-27:], truth)
        loss.backward()
        optimizer.step()
        pbar.set_description(
            (
                f'Epoch : {i}; loss: {loss.item()}'
            )
        )
    print('Approximated value of theta: r, t_a, Ds_c, Asigma')
    print((theta * scaler).detach().numpy())
    line1, = plt.plot(range(1992, 2019), yearly_R, label='Ground truth')
    # line2, = plt.plot(times[-time_size:], pred_R, label='Prediction')
    line3, = plt.plot(times[-time_size:], torch_R.detach().numpy(),
                      label='Prediction of torch version')
    plt.legend()
    plt.ylabel('Number of cummulative events')
    plt.xlabel('Year')
    plt.title('Optimization based method')
    plt.savefig('figs/optim_map.png', bbox_inches='tight', dpi=400)
    plt.show()
