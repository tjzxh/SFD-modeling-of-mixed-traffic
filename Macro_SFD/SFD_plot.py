import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import find_peaks
from scipy.stats import entropy
from scipy.signal import savgol_filter

plt.rcParams['font.size'] = '35'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.titlepad'] = 14
n = 20
dec = 0.5
# read the platoon arrangement data
workbook = 'platoon_arrangement_' + str(n) + '.xlsx'
all_metric = []
for ind, pct in enumerate([0.0, 0.05, 0.1, 0.15, 0.2, 0.25]):
    # read the sheet as a dataframe for each p
    df = pl.read_excel(source=workbook, sheet_name=f'p={np.round(pct, 2)}', read_options={"has_header": True},
                       schema_overrides={"speed": pl.Float32, "density": pl.Float32, "probability": pl.Float64})
    # compute the flow and round it to the nearest integer
    df = df.with_columns(q=(pl.col('speed') * pl.col('density') / dec).round() * dec,
                         k=(pl.col('density') / dec).round() * dec)
    dfo = df.group_by(['q', 'k']).agg(prob=pl.sum('probability'))
    # compute the entropy of the (q,k) pairs
    qk_entropy = entropy(dfo['prob'].to_numpy())
    # group by density and normalize the probability
    df = dfo.group_by('k').agg(norm_prob=pl.col('prob') / pl.sum('prob'), flow=pl.col('q'))
    df = df.explode(['flow', 'norm_prob'])
    # group the data by density and compute the mean flow and variance with the prob
    all_qk_st = []
    for kn in np.unique(df['k'].to_numpy()):
        sdf = df.filter(pl.col('k') == kn)
        sd = sdf.to_numpy()
        ks, ps, qs = sd[:, 0], sd[:, 1], sd[:, 2]
        norm_prob = ps / np.sum(ps)
        mean_q = np.sum(qs * norm_prob)
        sigma_q = np.sqrt(np.sum((qs - mean_q) ** 2 * norm_prob))
        all_qk_st.append([kn, mean_q, sigma_q, len(qs)])
    # plot fill_between the mean flow and its variance
    fig, ax = plt.subplots(layout='tight', figsize=(10, 10))
    all_qk_st = np.array(all_qk_st)
    dfl = df.to_numpy()
    k, p, q = dfl[:, 0], dfl[:, 1], dfl[:, 2]
    # # violin plot the joint probability
    # ax.violinplot(p, widths=0.7)

    # plot all the ve, density with joint probability
    pcm = ax.scatter(k, q, s=10)
    # pcm = ax.scatter(k, q, c=p, s=10, cmap='Greens', norm=colors.TwoSlopeNorm(vmin=0, vcenter=0.01, vmax=1))
    # apply Savitzky-Golay filter for the mean flow
    wl = [70, 75, 70, 70, 65, 65]
    y_filter = savgol_filter(all_qk_st[:, 1], wl[ind], 3)
    # fing peaks of filtered flow
    maxq_ind = np.argmax(y_filter)
    kc, qm, qstd = all_qk_st[maxq_ind, 0], all_qk_st[maxq_ind, 1], all_qk_st[maxq_ind, 2]
    # # qstd is the average of the flow std around the peak
    # qstd = np.mean(all_qk_st[maxq_ind - 1:maxq_ind + 2, 2])
    ax.plot(all_qk_st[:, 0], y_filter, 'r', lw=2, label='Estimated mean flow')
    # # plot the original mean flow
    # ax.plot(all_qk_st[:, 0], all_qk_st[:, 1], c='r')
    ax.set_xlabel('Density (veh/km)')
    ax.set_ylabel('Flow (veh/h)')
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 5500])
    cb = fig.colorbar(pcm, ax=ax, shrink=0.67, label='Probability')
    cb.set_ticks([0, 1])
    ax.set_box_aspect(1)
    ax.legend(fontsize=20, loc='upper left')
    plt.show()
    # # save the figure
    # plt.savefig(f'platoon_arrangement_{n}_p={np.round(pct, 2)}.png', dpi=500)
    # close the figure for this instance in loop to save memory
    # plt.close(fig)
    # all_metric.append([n, pct, kc, qm, qstd, qk_entropy])
    # print(f'p={pct}, kc={kc}, qm={qm}, qstd={qstd}')
# # save the data to Excel file with headers
# all_metric = np.array(all_metric)
# metric = pl.from_numpy(all_metric, schema=['n', 'pct', 'kc', 'qm', 'qstd', 'qk_entropy'])
# metric.write_csv('SFD_metric_smooth.csv')
