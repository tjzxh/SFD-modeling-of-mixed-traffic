import torch
from sympy.abc import alpha

from block_tanh import MDN
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from scipy.stats import truncnorm
from matplotlib import colors

plt.rcParams['font.size'] = '35'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.titlepad'] = 50
v_max, s_max, step = 35.0, 100.0, 5  # m/s, m


def tg_cdf(sigma, mu, y):
    a, b = (0 - mu) / sigma, (s_max - mu) / sigma
    rv = truncnorm(a, b, loc=mu, scale=sigma)
    return np.diff(rv.cdf(y), axis=0, prepend=0)


def sum_tg_cdf(pi, sigma, mu, y):
    # for each element in the sigma, mu array, weighted sum the truncated Gaussian distribution
    cdf = np.zeros_like(y)
    for ind in range(len(pi)):
        cdf += pi[ind] * tg_cdf(sigma[ind], mu[ind], y)
    return cdf


# iterate over all the speed and spacing
def equilibrium_state_calculate(model_type, se, ve):
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
    # read the __equilibrium_pvv.csv
    equilibrium_pvv = pl.read_csv(f'./csv_results/{model_type}__equilibrium_pvv.csv')
    # get all the pairs of speed and spacing
    all_pairs = np.zeros((len(ve) * len(se), 7))
    for i, speed in enumerate(ve):
        fix_speed_pairs = np.zeros((len(se), 7))
        fix_speed_pairs[:, 0] = speed
        fix_speed_pairs[:, 1] = speed
        fix_speed_pairs[:, 2] = se
        # calculate the probability of each pair of speed and spacing
        X = torch.tensor(np.array([speed, speed]), dtype=torch.float32)
        pi, sigma, mu = mdn_model(X)
        pi, sigma, mu = pi.detach().numpy(), sigma.detach().numpy(), mu.detach().numpy()
        # compute the predicted probability and normalize it
        y = se.reshape(-1, 1)
        mdn_prob = sum_tg_cdf(pi, sigma, mu, y)
        # filter the probability that is larger than 1/len(y)/2 and normalize it
        mdn_prob = np.where(mdn_prob > 1 / len(y) / 2, mdn_prob, 0)
        # find the index of non-zero probability
        non_zero_index = np.where(mdn_prob > 0)[0]
        # filter the longest consecutive list
        if len(non_zero_index) > 0:
            longest_consecutive = np.split(non_zero_index, np.where(np.diff(non_zero_index) != 1)[0] + 1)
            longest_consecutive = max(longest_consecutive, key=len)
            # zero the probability that is not in the longest consecutive list
            mdn_prob[~np.isin(np.arange(len(y)), longest_consecutive)] = 0
        mdn_prob = mdn_prob / np.sum(mdn_prob)
        fix_speed_pairs[:, 4] = mdn_prob.reshape(-1)
        # calculate the std of the equilibrium spacing for each speed
        fix_speed_pairs[:, 5] = mdn_prob.reshape(-1) * (se - np.dot(mdn_prob.reshape(-1), se.reshape(-1))) ** 2
        # compute the mean of the equilibrium spacing for each speed
        fix_speed_pairs[:, 6] = mdn_prob.reshape(-1) * se.reshape(-1)
        # find the pvv that is the same as the speed
        sub_pvv = equilibrium_pvv.filter(equilibrium_pvv['speed'] == speed)
        # get the equilibrium probability
        pvv = sub_pvv['pvv'].to_numpy() if speed > 0 else 1
        # P(Vi,Si|Vi-1) = P(Si|Vi-1,Vi) * P(Vi|Vi-1)
        mdn_prob = mdn_prob * pvv
        fix_speed_pairs[:, 3] = mdn_prob.reshape(-1)
        # save the pairs of speed and spacing to the all_pairs
        all_pairs[i * len(se):(i + 1) * len(se)] = fix_speed_pairs
    # save the equilibrium states and corresponding probabilities to dataframes
    equilibrium_data = pl.from_numpy(all_pairs,
                                     schema=['speed', 'equilibrium_speed', 'spacing', 'pvsv', 'equilibrium_prob',
                                             'variance', 'mean'])
    # filter out the rows with probability 0
    equilibrium_data = equilibrium_data.filter(equilibrium_data['equilibrium_prob'] > 0)
    # select the columns: equilibrium_speed, spacing, equilibrium_prob
    equilibrium_data = equilibrium_data.select(
        ['equilibrium_speed', 'spacing', 'pvsv', 'equilibrium_prob', 'variance', 'mean'])
    equilibrium_data.write_csv(f'{model_type}_mdn_equilibrium.csv')
    # sum the variance for each speed
    statis_data = equilibrium_data.group_by('equilibrium_speed').agg(spacing_std=pl.sum('variance').sqrt(),
                                                                     spacing_mean=pl.sum('mean'))
    # add column spacing_cv=pl.col('spacing_std') / pl.col('spacing_mean')
    statis_data = statis_data.with_columns(spacing_cv=pl.col('spacing_std') / pl.col('spacing_mean'))
    # compute the avg spacing_cv
    avg_scv = statis_data['spacing_cv'].mean()
    print(f"the average cv of {model_type} is {avg_scv}")
    # sort the variance data by the equilibrium speed
    statis_data = statis_data.sort('equilibrium_speed')
    # plot the meshgrid of equilibrium states with the probability
    fig, ax = plt.subplots()
    pcm = ax.scatter(equilibrium_data['spacing'].to_numpy(), equilibrium_data['equilibrium_speed'].to_numpy(),
                     c=equilibrium_data['equilibrium_prob'].to_numpy(), cmap='Greens',
                     norm=colors.TwoSlopeNorm(vmin=0, vcenter=0.15, vmax=1))
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', label='Estimated equilibrium states',
                              color='g', markersize=6, lw=0, alpha=0.5)]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=25)
    # # plot the real data
    # plt.scatter(data['spacing'].to_numpy(), data['follower_speed'].to_numpy(), s=0.1, label='Waymo Data', c='k')
    # # plot the mean of the equilibrium spacing for each speed
    # ax.scatter(statis_data['spacing_mean'].to_numpy(), statis_data['equilibrium_speed'].to_numpy(), c='r', marker='.',
    #            label='Estimated mean equilibrium spacing')
    plt.xlabel('Equilibrium spacing (m)')
    plt.ylabel('Equilibrium speed (m/s)')
    cb = fig.colorbar(pcm, ax=ax, label='Probability')
    cb.set_ticks([0, 1])
    plt.title(f'{model_type} MDN')
    ax.set_box_aspect(1)
    ax.set_xlim([0, 80])
    # plt.legend()
    plt.show()
    return statis_data


if __name__ == '__main__':
    ve = np.linspace(0, 20, 20 * 2 + 1)
    se = np.linspace(0, 100, 100 * 1 + 1)
    equilibrium_state_calculate('hv_av', se, ve)
    equilibrium_state_calculate('av_hv', se, ve)
    equilibrium_state_calculate('hv_hv', se, ve)
    # for model_type in ['hv_av', 'av_hv', 'hv_hv']:
    #     statis_data = equilibrium_state_calculate(model_type, se, ve)
    #     # plot the spacing_cv for each speed
    #     plt.plot(statis_data['equilibrium_speed'], statis_data['spacing_cv'], label=model_type, marker='o')
    #     plt.xlabel('Equilibrium speed (m/s)')
    #     plt.ylabel('CV of equilibrium spacing')
    #     plt.legend()
    # plt.show()
