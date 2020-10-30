from sklearn import cluster, decomposition, preprocessing, feature_selection
import pandas as pd
import numpy as np
from scipy.spatial import distance

centers = pd.read_csv('../../data/interim/Modelling/clusterProfilesTSVD.csv').iloc[:,1:]

dev = pd.read_csv('../../data/processed/DEVELOPERS_DATA.csv').iloc[:,1:]
cols = ['committer'] + list(centers.columns)
dev = dev.reindex(columns=cols)

dev2 = dev.iloc[:,1:]

kmeans = cluster.KMeans(n_clusters=5, init=centers, n_init=1, max_iter=1).fit(dev2)
kmeans.cluster_centers = np.array(centers)

clusters = kmeans.predict(dev2)

dev['cluster'] = clusters

# Within cluster variance

def WCV(dev, centers):
  WCV = np.zeros(5)
  for i in range(5):  # clusters
    X = dev[dev.cluster==i].iloc[:,1:-1]
    c = [np.array(centers)[i]]
    d = distance.cdist(X, c)
    WCV[i] = d.sum()/d.shape[0]
  return [WCV, WCV.sum()]

 cluster, total = WCV(dev, centers)

 # Between cluster variance

 def BCV(dev, centers):
  BCV = np.zeros(5)
  x = [np.array(dev.iloc[:,1:-1].mean())]
  for i in range(5):
    n = dev[dev.cluster==i].shape[0]
    c = [np.array(centers)[i]]
    d = distance.cdist(c, x)
    BCV[i] = n*d.sum()
  return [BCV, BCV.sum()]

 cluster, total = BCV(dev, centers)

# Davies–Bouldin index

def DB(dev, centers):
  wcv, _ = WCV(dev, centers) # mean distance of all elements in cluster to centroid
  DBC = np.zeros((5,5)) # distance between centroids
  DavisBouldin = 0
  for i in range(5):  # clusters
    max = 0
    for j in range(5):
      ci = [np.array(centers)[i]]
      cj = [np.array(centers)[j]]
      d = distance.cdist(ci, cj)
      DBC[i,j] = d.sum()

      if i != j:
        val = (wcv[i]+wcv[j])/DBC[i,j]
        if val > max:
          max = val
    DavisBouldin += max
  return DavisBouldin/5

DavisBouldinIndex = DB(dev, centers)

# Types of issues

centers[["codeBlockerViolations", "codeInfoViolations",	"codeMajorViolations", "codeBugs", "codeViolations", "codeVulnerabilities",	"codeCodeSmells", "codeCriticalViolations",	"codeMinorViolations", "inducedSZZIssues",	"inducedSonarIssues", ]]