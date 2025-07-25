import torch
from block_tanh import MDN, truncated_gaussian_distribution, truncated_gaussian_cdf
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from scipy.stats import truncnorm

plt.rcParams['font.size'] = '35'
plt.rcParams["font.family"] = "Times New Roman"
v_max, s_max, step = 35.0, 100.0, 5  # m/s, m


# define a function to compute the mean and std of the truncated Gaussian distribution
def tg_mean_std(X, mdn_model, sum_flag=True):
    # input X array
    X = torch.tensor(X, dtype=torch.float32)
    pi, sigma, mu = mdn_model(X)
    # compute the predicted mean and std
    pi, mu, sigma = pi.detach().numpy(), mu.detach().numpy(), sigma.detach().numpy()
    # if len(pi.shape) == 1:
    #     pi, mu, sigma = pi.reshape(1, -1), mu.reshape(1, -1), sigma.reshape(1, -1)
    if sum_flag:
        mean, std = np.sum(pi * mu, axis=1), np.sqrt(np.sum(pi ** 2 * sigma ** 2, axis=1))
        return mean, std
    else:
        return pi, sigma, mu


# compute the MAE and MAPE between the predicted and true values
def compute_error(true_mean, pred_mean):
    mae = np.mean(np.abs(true_mean - pred_mean))
    mape = np.mean(np.abs(true_mean - pred_mean) / true_mean)
    return mae, mape


# get the probability density of the truncated Gaussian distribution
def tg_pdf(sigma, mu, y):
    a, b = (0 - mu) / sigma, (s_max - mu) / sigma
    rv = truncnorm(a, b, loc=mu, scale=sigma)
    return rv.pdf(y)


def sum_tg_pdf(pi, sigma, mu, y=np.linspace(0, s_max, int(s_max) * step)):
    # for each element in the sigma, mu array, weighted sum the truncated Gaussian distribution
    pdf = np.zeros_like(y)
    for ind in range(len(pi)):
        pdf += pi[ind] * tg_pdf(sigma[ind], mu[ind], y)
    return pdf


if __name__ == '__main__':
    # all_model_type = ['av_hv', 'hv_av', 'hv_hv']
    for model_type in ['hv_av']:
        # different parameters for different model types
        if model_type == 'hv_av':
            epoch, hidden_dim, number_of_Gaussian = 20000, 64, 10
        elif model_type == 'av_hv':
            epoch, hidden_dim, number_of_Gaussian = 20000, 64, 10
        elif model_type == 'hv_hv':
            epoch, hidden_dim, number_of_Gaussian = 15000, 128, 100
        # load the model
        mdn_model = MDN(input_dim=2, n_hidden=hidden_dim, n_gaussians=number_of_Gaussian)
        mdn_model.load_state_dict(torch.load(f'./trained_model/{model_type}_model.pth'))
        mdn_model.eval()
        # load the dataset
        data_path = './data/waymo/' + model_type + '_data.parquet'
        data = pl.read_parquet(data_path)
        X = data.select(['leader_speed', 'follower_speed'])
        y = data.select(['spacing'])
        X, y = X.to_numpy(), y.to_numpy()
        all_input, all_output_mean, all_output_std = [], [], []
        leader_speed_test, follower_speed_test = X[:, 0], X[:, 1]
        for i in range(2, int(v_max)):
            for j in range(1, int(v_max)):
                subset = (leader_speed_test >= i) & (leader_speed_test < i + 1) & (follower_speed_test >= j) & (
                        follower_speed_test < j + 1)
                X_sub, y_sub = X[subset], y[subset]
                if len(y_sub) > 120:
                    all_input.append(np.mean(X_sub, axis=0))
                    all_output_mean.append(np.mean(y_sub))
                    # compute the learned Mixtures of Gaussians for each subset
                    pi, sigma, mu = tg_mean_std(np.mean(X_sub, axis=0), mdn_model, sum_flag=False)
                    # get the truncated Gaussian distribution for each subset
                    pred_density = sum_tg_pdf(pi, sigma, mu)
                    # # plot the distribution of the empirical spacing in X_sub and the predicted distribution
                    # _, ax = plt.subplots()
                    # ax.hist(y_sub, bins=50, density=True, alpha=0.4, color='b', label='Waymo data')
                    # ax.plot(np.linspace(0, s_max, int(s_max) * step), pred_density, c='r', label='MDN model')
                    # ax.set_xlabel('Spacing (m)')
                    # ax.set_ylabel('Probability density')
                    # ax.set_xlim([0, s_max])
                    # ax.legend()
                    # # set box aspect ratio
                    # ax.set_box_aspect(1)
                    # # plt.title(f'{model_type} model: leader_speed={i} m/s, follower_speed={j} m/s', fontsize=20)
                    # print(f'{model_type} model: leader_speed={i} m/s, follower_speed={j} m/s')
                    # plt.show()
        # compute the learned Mixtures of Gaussians for each subset
        all_input, all_output_mean = np.array(all_input), np.array(all_output_mean)
        pred_mean, pred_std = tg_mean_std(all_input, mdn_model, sum_flag=True)
        # compute the MAE and MAPE between the predicted and true values
        mae, mape = compute_error(np.array(all_output_mean), pred_mean)
        print(f'{model_type} model: MAE={mae}, MAPE={mape} with data size={data.shape[0]}')
        sm_min, sm_max = np.min(all_output_mean), np.max(all_output_mean)
        # plot the predicted mean vs the true mean and y=x line
        _, ax = plt.subplots()
        ax.scatter(all_output_mean, pred_mean, c='b')
        ax.plot([sm_min, sm_max], [sm_min, sm_max], c='r', label='y=x')
        ax.set_xlabel('Empirical mean spacing (m)')
        ax.set_ylabel('Predicted mean spacing (m)')
        ax.set_box_aspect(1)
        ax.set_xlim([sm_min, sm_max])
        ax.legend()
        # plt.title(f'{model_type} model')
        plt.show()
# av_hv model: MAE=1.6900274216131954, MAPE=0.05251494066691744 with data size=29136
# hv_av model: MAE=1.6949510607558453, MAPE=0.05991766155294307 with data size=43454
# hv_hv model: MAE=1.4964835679249782, MAPE=0.048772181874552414 with data size=153721
