# Given n vehicles in total, np AVs (1) and n-np HVs (2)
from more_itertools import distinct_permutations
import numpy as np
import polars as pl
from itertools import product
from xlsxwriter import Workbook


# get the equilibrium spacing and prob under the given speed and model type
def equilibrium(model_type):
    # load the equilibrium data
    equilibrium_data = pl.read_csv(f'{model_type}_mdn_equilibrium.csv')
    # convert the data to numpy array
    equilibrium_data = equilibrium_data.to_numpy()
    return equilibrium_data


def platoon_arrangement(n, p):
    AV_num, HV_num = int(p), n - int(p)
    platoon = [1] * AV_num + [2] * HV_num
    all_dpm = distinct_permutations(platoon)
    all_platoon = []
    # iteratively give all the unique permutations
    for platoon in list(all_dpm):
        # convert the tuple to array
        platoon = np.array(platoon)
        # diff the array along the row to find the change of vehicle type
        platoon_diff = np.diff(platoon)
        # count the number of AV-HV (-1 in diff), HV-AV (1 in diff) for each row
        HV_AV_platoon, AV_HV_platoon = np.sum(platoon_diff == 1), np.sum(platoon_diff == -1)
        # calculate the number of all AV following pairs (np if the zero-th vehicle is HV, otherwise np-1)
        all_AV = AV_num - (platoon[0] == 1)
        all_HV = HV_num - (platoon[0] == 2)
        # calculate the number of AV-AV (np-AV_HV), HV-HV (n-np-HV_AV)
        AV_AV_platoon, HV_HV_platoon = all_AV - AV_HV_platoon, all_HV - HV_AV_platoon
        # append the AV-HV, AV-AV, HV-AV, HV-HV to the list
        all_platoon.append([AV_HV_platoon, AV_AV_platoon, HV_AV_platoon, HV_HV_platoon])
    # calculate the unique set of (AV-HV, AV-AV, HV-AV, HV-HV) and their counts
    unique, counts = np.unique(np.array(all_platoon), axis=0, return_counts=True)
    # covert the unique set to array
    unique = np.array(unique)
    prob = counts / np.sum(counts)
    return unique, prob


# compute the Cartesian product of equilibrium spacings for given equilibrium speed and platoon arrangement
def cartesian(av_hv_equi, hv_av_equi, hv_hv_equi, arrangement):
    AV_HV, AV_AV, HV_AV, HV_HV = arrangement[0], arrangement[1], arrangement[2], arrangement[3]
    av_hv_s, av_hv_p = av_hv_equi[:, 1].tolist(), av_hv_equi[:, 2].tolist()
    hv_av_s, hv_av_p = hv_av_equi[:, 1].tolist(), hv_av_equi[:, 2].tolist()
    hv_hv_s, hv_hv_p = hv_hv_equi[:, 1].tolist(), hv_hv_equi[:, 2].tolist()
    # recursively compute the Cartesian product of two lists for each vehicle type
    s_av, p_av, s_hv_av, p_hv_av, s_hv_hv, p_hv_hv = [], [], [], [], [], []
    if AV_HV or AV_AV:
        s_av, p_av = repeat_cartesian_product(av_hv_s, av_hv_p, AV_HV + AV_AV)
    if HV_AV:
        s_hv_av, p_hv_av = repeat_cartesian_product(hv_av_s, hv_av_p, HV_AV)
    if HV_HV:
        s_hv_hv, p_hv_hv = repeat_cartesian_product(hv_hv_s, hv_hv_p, HV_HV)
    # combine above three types of spacings and probabilities
    lg, jpb = cartesian_product(s_av, p_av, s_hv_av, p_hv_av)
    all_length, all_joint_prob = cartesian_product(lg, jpb, s_hv_hv, p_hv_hv)
    return all_length, all_joint_prob


# recursively repeat the Cartesian product of two lists
def repeat_cartesian_product(s, p, num):
    if num == 1:
        return s, p
    if num == 2:
        return cartesian_product(s, p, s, p)
    if num % 2 == 0:
        return cartesian_product(repeat_cartesian_product(s, p, num / 2)[0],
                                 repeat_cartesian_product(s, p, num / 2)[1],
                                 repeat_cartesian_product(s, p, num / 2)[0],
                                 repeat_cartesian_product(s, p, num / 2)[1])
    else:
        return cartesian_product(repeat_cartesian_product(s, p, num - 1)[0],
                                 repeat_cartesian_product(s, p, num - 1)[1],
                                 s, p)


# compute the Cartesian product of two lists
def cartesian_product(sa, pa, sb, pb):
    # check whether contains empty list
    if not len(sa):
        return sb, pb
    if not len(sb):
        return sa, pa
    # # compute the Cartesian product of spacing for each vehicle type
    all_spacing_pairs = list(product(sa, sb))
    all_spacing_pairs = np.array(all_spacing_pairs)
    # for each tuple, compute the sum
    all_length = np.sum(all_spacing_pairs, axis=1)
    # compute the Cartesian product of probability for each vehicle type
    all_prob_pairs = list(product(pa, pb))
    all_prob_pairs = np.array(all_prob_pairs)
    # for each tuple, compute the product
    all_joint_prob = np.prod(all_prob_pairs, axis=1)
    # round the all length to the nearest integer
    all_length = np.round(all_length).astype(int)
    # sum the probability of duplicate length
    unique_length = np.unique(all_length)
    unique_prob = np.zeros(len(unique_length))
    for i, j in enumerate(unique_length):
        unique_prob[i] = np.sum(all_joint_prob[all_length == j])
    return unique_length, unique_prob


# calculate the joint probability of P(V=v,k=k)=P(k=k1│V=v)×P(V=v)
def vk_joint_prob(n, p):
    # get the unique set of AV-HV, HV-AV pairs and their probabilities
    unique, prob = platoon_arrangement(n, p)
    print(f"completed the platoon arrangement for n={n} and p={p}")
    # calculate the density for each speed
    hv_av_equi = equilibrium('hv_av')
    av_hv_equi = equilibrium('av_hv')
    hv_hv_equi = equilibrium('hv_hv')
    # iterate over each speed in the speed range
    all_ve = hv_av_equi[:, 0]
    # get the unique ve in all_ve with order
    unique_ve = np.unique(all_ve)
    # iterate over each platoon arrangement
    all_vkp = np.array([])
    for i in range(len(unique)):
        arrangement, arrange_prob = unique[i], prob[i]
        for ve in unique_ve:
            # get the equilibrium spacing and probability for the speed
            hv_av_data = hv_av_equi[hv_av_equi[:, 0] == ve]
            av_hv_data = av_hv_equi[av_hv_equi[:, 0] == ve]
            hv_hv_data = hv_hv_equi[hv_hv_equi[:, 0] == ve]
            # compute the Cartesian product of equilibrium spacings for given equilibrium speed and platoon arrangement
            all_length, all_joint_prob = cartesian(av_hv_data, hv_av_data, hv_hv_data, arrangement)
            # compute the density
            all_density = n / np.array(all_length)
            # compute the joint probability of P(V=v,k=k)=SUM[P(V,k,C)×P(C)] for each C
            final_joint_prob = np.array(all_joint_prob) * arrange_prob
            # stack the (V, k) pairs and their joint probability
            vkp = np.column_stack((np.ones(len(all_density)) * ve, all_density, final_joint_prob))
            all_vkp = np.vstack((all_vkp, vkp)) if all_vkp.size else vkp
    return all_vkp


# sample the (V, k) pairs with the joint probability
def sample_vk(n, p, sample_num=1000):
    all_vkp = vk_joint_prob(n, p)
    # normalize the joint probability for all density-speed pairs
    all_vkp[:, 2] = all_vkp[:, 2] / np.sum(all_vkp[:, 2])
    # convert m/s to km/h
    all_vkp[:, 0] = all_vkp[:, 0] * 3.6
    all_vkp[:, 1] = all_vkp[:, 1] * 1000
    return all_vkp


# plot all the ve, density with joint probability and mean density
if __name__ == '__main__':
    n = 40
    # # create an Excel file
    # with Workbook('platoon_arrangement_' + str(n) + '.xlsx') as workbook:
    #     # p is the number of AVs
    #     for p in range(n + 1):
    #         # sample the (V, k) pairs with the joint probability
    #         all_vkp = sample_vk(n, p)
    #         # save all_vkp to a sheet for each p in one csv file
    #         all_vkp_df = pl.from_numpy(all_vkp, schema=['speed', 'density', 'probability'])
    #         all_vkp_df.write_excel(workbook, worksheet=f'p={np.round(p, 2)}')
    #         print(f'p={np.round(p, 2)} is done')

    # get the CF sets only
    # store the unique set of (AV-HV, AV-AV, HV-AV, HV-HV) with prob and n,p in an Excel file
    with Workbook('CF_config' + str(n) + '.xlsx') as workbook:
        for p in range(n // 4 + 1):
            uniq, prob = platoon_arrangement(n, p)
            # add one more column for the total number of vehicles
            uniq = np.column_stack((uniq[:, 0] + uniq[:, 1], uniq))
            all_platoon_df = pl.from_numpy(np.column_stack((uniq, prob)),
                                           schema=['AV_follow', 'AV_HV', 'AV_AV', 'HV_AV', 'HV_HV', 'prob'])
            all_platoon_df.write_excel(workbook, worksheet=f'n={n}_p={p}')
            print(f'n={n}_p={p} is done')
