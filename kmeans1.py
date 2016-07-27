# coding=utf-8
#####################################################
# Tyler Hedegard
# 7/27/16
# Thinkful Data Science
# K-Nearest Means
#####################################################

import pandas as pd
import numpy as np
import scipy.cluster.vq


un = pd.read_csv('un.csv', sep=',', header=0, engine='python')

row_count = un.count().max()
row_count_per_column = un.count()

k = np.array(range(1, 11))
obs = un['GDPperCapita']
obs = obs.fillna(un['GDPperCapita'].median())
for each in k:
    out = scipy.cluster.vq.kmeans(obs, each)
    centroid = out[0]
    # print(out)
    """
    Didn't really understand the part to calculate the 'average within-cluster sum of squares' for each centroid section.
    I couldn't run the kmeans without handling the NaN values, and dropping, setting to zero or setting to median...
    I couldn't get the same values as the lecture, although looking at the latter plots, I think the centroids were the same.
    """

