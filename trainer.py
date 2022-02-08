import json
from typing import Tuple
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils import standart_scale

def fn_to_optimize(x, theta0, theta1) -> float:
    return theta1 * x + theta0

def optimize(x: np.ndarray, y: np.ndarray, lr=0.03, epochs=10000, early_stopping_patience=10, early_stopping_min_delta=1e-9) -> Tuple[float, float]:
    theta0 = 0
    theta1 = 0

    best_epoch_loss = None
    best_epoch_idx = 0

    for e in range(epochs):
        outs = np.array([fn_to_optimize(x[i], theta0, theta1) for i in range(x.shape[0])])
        this_epoch_loss = np.sum((outs - y)**2)

        rss_theta0 = outs - y
        rss_theta1 = (outs - y) * x
        d_theta0 = np.sum(rss_theta0) / x.shape[0]
        d_theta1 = np.sum(rss_theta1) / x.shape[0]

        theta0 -= lr * d_theta0
        theta1 -= lr * d_theta1

        if best_epoch_loss is None or best_epoch_loss - this_epoch_loss > early_stopping_min_delta:
            best_epoch_loss = this_epoch_loss
            best_epoch_idx = e

        if early_stopping_patience is not None and e - best_epoch_idx > early_stopping_patience:
            break

    return theta0, theta1

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p', '--save-plot', dest='plot', default=None, action='store_true', help='Save plot of the solution to file')
    parser.add_argument('-r', '--learning-rate', dest='lr', type=float, help='Specify learning rate', default=0.03)
    parser.add_argument('--num-epochs', dest='epochs', type=int, help='Specify number of training epochs', default=1000)
    parser.add_argument('--stop-patience', dest='patience', type=int, help='Specify early stopping patience', default=10)
    parser.add_argument('--stop-min-delta', dest='delta', type=float, help='Specify early stopping minimum delta', default=1e-9)
    parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                        help='Print how answer is figured out')

    args = parser.parse_args()

    data = pd.read_csv('data.csv')
    km_scaled = standart_scale(data['km'])
    price_scaled = standart_scale(data['price'])

    print('Optimization started. Initial mae={}'.format(np.mean(np.abs(fn_to_optimize(data['km'], 0, 0) - data['price']))))

    theta0, theta1 = optimize(km_scaled[0], price_scaled[0], 
                                epochs=args.epochs,
                                lr=args.lr, 
                                early_stopping_patience=args.patience,
                                early_stopping_min_delta=args.delta)

    theta1 = (theta1 * price_scaled[1]) / km_scaled[1]
    theta0 = (theta0 - (km_scaled[2] * theta1) / km_scaled[1]) * price_scaled[1] + price_scaled[2]

    data['predition'] = fn_to_optimize(data['km'], theta0, theta1)
    print('Optimization completed. Achieved mae={}'.format(np.mean(np.abs(data['predition'] - data['price']))))

    if args.plot:
        sns.lineplot(x='km', y='value', hue='variable', data=pd.melt(data, ['km']))
        plt.savefig('plot.png')

    with open('config.json', 'w') as f:
        json.dump({'theta1': theta1, 'theta0': theta0}, f)
    print(f'Theta1={theta1} and Theta0={theta0} are saved to ./config.json')

if __name__ == '__main__':
    main()