import numpy as np
import polars as pl
from scipy.stats import entropy
from scipy.signal import savgol_filter

n = 20
dec = 0.5
# read the platoon arrangement data
workbook = 'platoon_arrangement_' + str(n) + '.xlsx'
# workbook = 'AV_platoon_control.xlsx'
all_metric = []
for ind, pct in enumerate(np.linspace(0, 1, 2 * 10 + 1)):  # np.linspace(0, 1, 2 * 10 + 1) OR range(n + 1)
    # read the sheet as a dataframe for each p
    # change to the number of AVs when workbook = 'AV_platoon_control.xlsx'
    pct = int(pct * n) if workbook == 'AV_platoon_control.xlsx' else pct
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
    all_qk_st = np.array(all_qk_st)
    # # apply Savitzky-Golay filter for the mean flow
    # wl = [70, 75, 70, 70, 65, 65]
    # tt = wl[ind] if ind <= len(wl) - 1 else 65
    # y_filter = savgol_filter(all_qk_st[:, 1], tt, 3)
    # # fing peaks of filtered flow
    y_filter = all_qk_st[:, 1]  # no smoothing
    maxq_ind = np.argmax(y_filter)
    kc, qm, _ = all_qk_st[maxq_ind, 0], all_qk_st[maxq_ind, 1], all_qk_st[maxq_ind, 2]
    # qstd is the average of the flow std around the peak
    qstd = np.mean(all_qk_st[maxq_ind - 1:maxq_ind + 1, 2])
    all_metric.append([n, pct, kc, qm, qstd, qk_entropy])
# save the data to Excel file with headers
all_metric = np.array(all_metric)
metric = pl.from_numpy(all_metric, schema=['n', 'pct', 'kc', 'qm', 'qstd', 'qk_entropy'])
metric.write_csv('SFD_metric_all.csv')
# metric.write_csv('spatial_control_metric_smooth.csv')
