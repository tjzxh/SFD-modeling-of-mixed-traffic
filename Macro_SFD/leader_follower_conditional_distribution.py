import polars as pl
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '35'
plt.rcParams["font.family"] = "Times New Roman"
# for each model type, read the data
all_model_type = ['hv_av', 'hv_hv', 'av_hv']
dec = 0.5
for model_type in all_model_type:
    data_path = './data/waymo/' + model_type + '_data.parquet'
    df = pl.read_parquet(data_path)
    # round the follower speed, leader speed and spacing to nearest integer
    data = df.with_columns(vs=(pl.col('follower_speed') / dec).round() * dec,
                           vl=(pl.col('leader_speed') / dec).round() * dec,
                           s=(pl.col('spacing') / dec).round() * dec)
    # group by the leader speed and follower speed, and count the number of each group
    agg_data = data.group_by(['vl', 'vs']).agg(count=pl.count('s'))
    # sort the data by leader speed and follower speed
    agg_data = agg_data.sort('vl', 'vs')
    # get the conditional distribution of follower speed given leader speed
    agg_array = agg_data.to_numpy()
    # get the unique leader speed in ascending order
    vl = np.unique(agg_array[:, 0])
    vl = np.sort(vl)
    # for each leader speed, get the conditional distribution of follower speed
    for leader_speed in vl:
        # get the index of the leader speed
        idx = agg_array[:, 0] == leader_speed
        # get the conditional distribution of follower speed given leader speed
        conditional_distribution = agg_array[idx, 2]
        # normalize the conditional distribution
        agg_array[idx, 2] = conditional_distribution / np.sum(conditional_distribution)
        # # bar plot the probability for discrete follower speed
        # plt.bar(agg_array[idx, 1], agg_array[idx, 2], label=f'Leader speed: {leader_speed}')
        # plt.xlabel('Follower speed (m/s)')
        # plt.ylabel('Probability')
        # plt.legend()
        # plt.title(model_type)
        # plt.show()
    # save the agg_array to a csv file
    agg_df = pl.from_numpy(agg_array, schema=['leader_speed', 'follower_speed', 'probability'])
    agg_df.write_csv(f'{model_type}_speed_conditional_distribution.csv')
    # # find the row that leader speed equals to follower speed
    # idx = agg_array[:, 0] == agg_array[:, 1]
    # equal_df = pl.from_numpy(agg_array[idx, 1:], schema=['speed', 'pvv'])
    # equal_df.write_csv(f'{model_type}__equilibrium_pvv.csv')
    # ####### plot the follower speed vs leader speed
    # fig, ax = plt.subplots()
    # # plt.scatter(data['vl'].to_numpy(), data['vs'].to_numpy(), c='b', s=0.1)
    # # filter out the data with leader speed and follower speed less than 20 m/s
    # data = data.filter(
    #     (pl.col('leader_speed') <= 20) & (pl.col('follower_speed') <= 20) & (pl.col('follower_speed') > 0.1) & (
    #             pl.col('leader_speed') > 0.1))
    # plt.scatter(data['leader_speed'].to_numpy(), data['follower_speed'].to_numpy(), alpha=0.4, color='b', s=0.1)
    # # plot y=x line
    # plt.plot([0, 20], [0, 20], c='r', label='y=x')
    # plt.xlabel('Leader speed (m/s)')
    # plt.ylabel('Follower speed (m/s)')
    # ax.set_box_aspect(1)
    # # set x,ylim
    # plt.xlim([0, 20])
    # plt.ylim([0, 20])
    # # plot the title higher than the plot
    # plt.title(model_type, y=1.1)
    # # common settings for all plots
    # plt.tight_layout()
    # plt.show()
