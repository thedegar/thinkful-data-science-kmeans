# coding=utf-8
#####################################################
# Tyler Hedegard
# 7/27/16
# Thinkful Data Science
# K-Nearest Means
#####################################################

import pandas as pd
import scipy.cluster.vq


un = pd.read_csv('un.csv', sep=',', header=0, engine='python')

row_count = un.count().max()

obs = un['GDPperCapita']
obs = obs.fillna(un['GDPperCapita'].median())
kmeans = scipy.cluster.vq.kmeans(obs, 3)
centroids = kmeans[0]
centroids.sort()

un['cluster'] = 0
for j in range(0, row_count):
    min = False
    for i in range(0, len(centroids)):
        distance = abs(un['GDPperCapita'][j] - centroids[i])
        if not min:
            min = distance
            cluster = i
        else:
            if distance < min:
                min = distance
                cluster = i
    un['cluster'][j] = cluster

un.plot.scatter('GDPperCapita', 'infantMortality', c=un['cluster'])
un.plot.scatter('GDPperCapita', 'lifeMale', c=un['cluster'])
un.plot.scatter('GDPperCapita', 'lifeFemale', c=un['cluster'])

