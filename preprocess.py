import pandas
import numpy
import os
from sklearn.ensemble import RandomForestRegressor

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

"""
Input data, only a list of integers, after the first case has been found in switzerland
"""
# timeline: from 25.02.2020 - now
input_data_CH = [1,1,8,8,18,27,42,56,90,114,214,268,337,374,491,652,652,1139,1359,2200,2200,2700,3028,4075,5294,6575,7474,8795,9877,10897,11811,12928]
# timeline: 22.01.2020 - now
input_data_USA = [1,1,2,2,5,5,5,5,5,7,8,8,11,11,11,11,11,11,11,11,12,12,13,13,13,13,13,13,13,13,15,15,15,51,51,57,58,60,68,74,98,118,149,217,262,402,518,583,959,1281,1663,2179,2727,3499,4632,6421,7783,13677,19100,25489,33276,43847,53740,65778,83836,101657]
# timeline: 22.01.2020 - now
input_data_hubei = [444,444,549,761,1058,1423,3554,3554,4903,5806,7153,11177,13522,16678,19665,22112,24953,27100,29631,31728,33366,33366,48206,54406,56249,58182,59989,61682,62031,62442,62662,64084,64084,64287,64786,65187,65596,65914,66337,66907,67103,67217,67332,67466,67592,67666,67707,67743,67760,67773,67781,67786,67790,67794,67798,67799,67800,67800,67800,67800,67800,67800,67801,67801,67801,67801]
# timeline: 22.01.2020 - now
input_data_southKorea = [1,1,2,2,3,4,4,4,4,11,12,15,15,16,19,23,24,24,25,27,28,28,28,28,28,29,30,31,31,104,204,433,602,833,977,1261,1766,2337,3150,3736,4335,5186,5621,6088,6593,7041,7314,7478,7513,7755,7869,7979,8086,8162,8236,8320,8413,8565,8652,8799,8961,8961,9037,9137,9241,9332]
# timeline: 01.02.2020 - now
input_data_spain = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,6,13,15,32,45,84,120,165,222,259,400,500,673,1073,1695,2277,2277,5232,6391,7798,9942,11748,13910,17963,20410,25374,28768,35136,39885,49515,57786,65719]
# timeline: 31.01.2020 - now
input_data_italy = [2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,20,62,155,229,322,453,655,888,1128,1694,2036,2502,3089,3858,4636,5883,7375,9172,10149,12462,12462,17660,21157,24747,27980,31506,35713,41035,47021,53578,59138,63927,69176,74386,80589,86498]
# timeline: 23.01.2020 - now
input_data_singapore = [1,3,3,4,5,7,7,10,13,16,18,18,24,28,28,30,33,40,45,47,50,58,67,72,75,77,81,84,84,85,85,89,89,91,93,93,93,102,106,108,110,110,117,130,138,150,150,160,178,178,200,212,226,243,266,313,345,385,432,455,509,558,631,683,732]


all_data = pandas.DataFrame(columns=['QUARANTÄNE_BESCHRÄNKUNG', 'GRUPPENLIMITIERUNG', 'SCHULE', 'ARBEIT', 'LÄDEN', 'REISEBESCHRÄNKUNG', 'TESTING'])

# all_data = pandas.concat([all_data, new_DF], axis=0, ignore_index=True)

""" ============================== SWITZERLAND ============================== """
rates_CH = []
for i in range(1, len(input_data_CH)):
    rate = input_data_CH[i] / input_data_CH[i-1]
    rates_CH.append(rate)
rates_CH.append(rates_CH[-1])
massnahmen_CH = pandas.read_csv(os.path.join('datasets_preprocessed', 'coronavirus_measures_CH.csv'))
# combine rates with measures
massnahmen_CH.insert(7, 'RATE', rates_CH, True)
# finish pre-processing
# shift rates by n
n = 10
for i in range(massnahmen_CH.shape[0] - n):
    massnahmen_CH.iloc[i, -1] = massnahmen_CH.iloc[i+n, -1]
new_massnahmen = massnahmen_CH.iloc[0:(massnahmen_CH.shape[0] - n), :]

all_data = pandas.concat([all_data, new_massnahmen], axis=0, ignore_index=True)


""" ============================== USA ============================== """
rates_USA = []
for i in range(1, len(input_data_USA)):
    rate = input_data_USA[i] / input_data_USA[i-1]
    rates_USA.append(rate)
rates_USA.append(rates_USA[-1])
massnahmen_USA = pandas.read_csv(os.path.join('datasets_preprocessed', 'coronavirus_measures_USA.csv'))
# combine rates with measures
massnahmen_USA.insert(7, 'RATE', rates_USA, True)
# finish pre-processing
# shift rates by n
n = 10
for i in range(massnahmen_USA.shape[0] - n):
    massnahmen_USA.iloc[i, -1] = massnahmen_USA.iloc[i+n, -1]
new_massnahmen = massnahmen_USA.iloc[0:(massnahmen_USA.shape[0] - n), :]

all_data = pandas.concat([all_data, new_massnahmen], axis=0, ignore_index=True)


""" ============================== HUBEI (CHINA) ============================== """
rates_hubei = []
for i in range(1, len(input_data_hubei)):
    rate = input_data_hubei[i] / input_data_hubei[i-1]
    rates_hubei.append(rate)
rates_hubei.append(rates_hubei[-1])
massnahmen_hubei = pandas.read_csv(os.path.join('datasets_preprocessed', 'coronavirus_measures_Hubei.csv'))
# combine rates with measures
massnahmen_hubei.insert(7, 'RATE', rates_hubei, True)
# finish pre-processing
# shift rates by n
n = 10
for i in range(massnahmen_hubei.shape[0] - n):
    massnahmen_hubei.iloc[i, -1] = massnahmen_hubei.iloc[i+n, -1]
new_massnahmen = massnahmen_hubei.iloc[0:(massnahmen_hubei.shape[0] - n), :]

all_data = pandas.concat([all_data, new_massnahmen], axis=0, ignore_index=True)


# """ ============================== SOUTH KOREA ============================== """
# rates_southKorea = []
# for i in range(1, len(input_data_southKorea)):
#     rate = input_data_southKorea[i] / input_data_southKorea[i-1]
#     rates_southKorea.append(rate)
# rates_southKorea.append(rates_southKorea[-1])
# massnahmen_southKorea = pandas.read_csv(os.path.join('datasets_preprocessed', 'coronavirus_measures_SouthKorea.csv'))
# # combine rates with measures
# massnahmen_southKorea.insert(7, 'RATE', rates_southKorea, True)
# # finish pre-processing
# # shift rates by n
# n = 10
# for i in range(massnahmen_southKorea.shape[0] - n):
#     massnahmen_southKorea.iloc[i, -1] = massnahmen_southKorea.iloc[i+n, -1]
# new_massnahmen = massnahmen_southKorea.iloc[0:(massnahmen_southKorea.shape[0] - n), :]

# all_data = pandas.concat([all_data, new_massnahmen], axis=0, ignore_index=True)


""" ============================== SPAIN ============================== """
rates_spain = []
for i in range(1, len(input_data_spain)):
    rate = input_data_spain[i] / input_data_spain[i-1]
    rates_spain.append(rate)
rates_spain.append(rates_spain[-1])
massnahmen_spain = pandas.read_csv(os.path.join('datasets_preprocessed', 'coronavirus_measures_Spain.csv'))
# combine rates with measures
massnahmen_spain.insert(7, 'RATE', rates_spain, True)
# finish pre-processing
# shift rates by n
n = 10
for i in range(massnahmen_spain.shape[0] - n):
    massnahmen_spain.iloc[i, -1] = massnahmen_spain.iloc[i+n, -1]
new_massnahmen = massnahmen_spain.iloc[0:(massnahmen_spain.shape[0] - n), :]

all_data = pandas.concat([all_data, new_massnahmen], axis=0, ignore_index=True)


""" ============================== ITALY ============================== """
rates_italy = []
for i in range(1, len(input_data_italy)):
    rate = input_data_italy[i] / input_data_italy[i-1]
    rates_italy.append(rate)
rates_italy.append(rates_italy[-1])
massnahmen_italy = pandas.read_csv(os.path.join('datasets_preprocessed', 'coronavirus_measures_Italy.csv'))
# combine rates with measures
massnahmen_italy.insert(7, 'RATE', rates_italy, True)
# finish pre-processing
# shift rates by n
n = 10
for i in range(massnahmen_italy.shape[0] - n):
    massnahmen_italy.iloc[i, -1] = massnahmen_italy.iloc[i+n, -1]
new_massnahmen = massnahmen_italy.iloc[0:(massnahmen_italy.shape[0] - n), :]

all_data = pandas.concat([all_data, new_massnahmen], axis=0, ignore_index=True)


# """ ============================== SINGAPORE ============================== """
# rates_singapore = []
# for i in range(1, len(input_data_singapore)):
#     rate = input_data_singapore[i] / input_data_singapore[i-1]
#     rates_singapore.append(rate)
# rates_singapore.append(rates_singapore[-1])
# massnahmen_singapore = pandas.read_csv(os.path.join('datasets_preprocessed', 'coronavirus_measures_Singapore.csv'))
# # combine rates with measures
# massnahmen_singapore.insert(7, 'RATE', rates_singapore, True)
# # finish pre-processing
# # shift rates by n
# n = 10
# for i in range(massnahmen_singapore.shape[0] - n):
#     massnahmen_singapore.iloc[i, -1] = massnahmen_singapore.iloc[i+n, -1]
# new_massnahmen = massnahmen_singapore.iloc[0:(massnahmen_singapore.shape[0] - n), :]

# all_data = pandas.concat([all_data, new_massnahmen], axis=0, ignore_index=True)

""" CODE """

all_data = all_data.reindex(['QUARANTÄNE_BESCHRÄNKUNG', 'GRUPPENLIMITIERUNG', 'SCHULE', 'ARBEIT', 'LÄDEN', 'REISEBESCHRÄNKUNG', 'TESTING', 'RATE'], axis=1)

all_data.to_csv(os.path.join("datasets_preprocessed", "all_data.csv"))

for i in range(all_data.shape[1]):
    all_data.iloc[:, i] = all_data.iloc[:, i].astype(float)


model = RandomForestRegressor(
    n_estimators = 500,
    max_depth = None,
)

model.fit(
    all_data.iloc[:, 0:7],
    all_data.iloc[:, 7]
)

# predict
predict_df = pandas.DataFrame(columns=['QUARANTÄNE_BESCHRÄNKUNG', 'GRUPPENLIMITIERUNG', 'SCHULE', 'ARBEIT', 'LÄDEN', 'REISEBESCHRÄNKUNG', 'TESTING'])
for q_i in [0, 1, 2]:
    for g_i in [5, 100, 1000, 1000000]:
        for s_i in [0, 1, 2]:
            for a_i in [0, 1, 2]:
                for l_i in [0, 1, 2]:
                    for r_i in [0, 1, 2]:
                        for t_i in [100, 200, 500, 1000]:
                            predict_df = predict_df.append({
                                'QUARANTÄNE_BESCHRÄNKUNG': q_i,
                                'GRUPPENLIMITIERUNG': g_i,
                                'SCHULE': s_i,
                                'ARBEIT': a_i,
                                'LÄDEN': l_i,
                                'REISEBESCHRÄNKUNG': r_i,
                                'TESTING': t_i,
                            }, ignore_index=True)

y = model.predict(predict_df)

number_min_indices = 5
indices = numpy.argpartition(y, number_min_indices)

k_smallest_parameters = predict_df.iloc[indices[:number_min_indices]]

print("===============================================")
print(k_smallest_parameters)
print(y[indices[:number_min_indices]])
print(y)
print(type(y))

