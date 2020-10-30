
# Modelling developer profiles

!pip install altair

from sklearn import cluster, decomposition, preprocessing, feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import altair as alt

devs = pd.read_csv('../../data/processed/DEVELOPERS_DATA.csv')

# Simple feature selection

minMaxScaler = preprocessing.MinMaxScaler()
devsScaled = minMaxScaler.fit_transform(devs.drop(['committer', 'Unnamed: 0'], axis=1))

featureSelector = feature_selection.VarianceThreshold()
res = featureSelector.fit_transform(devsScaled)

# Dimensionality reduction with PCA

pca = decomposition.PCA(random_state=888)
devPCA = pca.fit(devsScaled)
comps = devPCA.components_
compsVariance = devPCA.explained_variance_
compsVarRatio = devPCA.explained_variance_ratio_

# plt.plot(compsVarRatio)

# Pick components that account for x% of variability
cumSum = 0
nComps = 0
x = 0.75
for varExplained in compsVarRatio:
  cumSum += varExplained
  if cumSum >= x:
    break
  else:
    nComps += 1

projectedPoints = devPCA.transform(devsScaled)
importantDims = projectedPoints[:,:nComps]

# Clustering over the Principal Components

Nc = range(1, 20)
kmeans = [cluster.KMeans(n_clusters=i, random_state=888) for i in Nc]
score = [kmeans[i].fit(importantDims).score(importantDims) for i in range(len(kmeans))]

def fitKMeansAndPlot(data, nClusters=3):
  """
  Fits the data to a KMeans model and plots the result over the principal 
  components that explain the most variance.
  """
  kmeans = cluster.KMeans(n_clusters=nClusters, random_state=888)
  fittedKMeans = kmeans.fit(importantDims) 
  centroids = kmeans.cluster_centers_
  labels = kmeans.predict(importantDims)

  colors=['red', 'green', 'blue', 'yellow', 'black', 'gray']
  colors = colors[:nClusters]
  toAsign=[]
  for row in labels:
      toAsign.append(colors[row])

  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(importantDims[:, 0], importantDims[:, 1], importantDims[:, 2], c=toAsign,s=60)
  ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c=colors, s=1000)

  return labels, kmeans

labels, kmeans = fitKMeansAndPlot(importantDims, nClusters=5)

nLabels = dict()
for label in labels:
  if label not in nLabels:
    nLabels[label] = 1
  nLabels[label] += 1 

labels, kmeans = fitKMeansAndPlot(importantDims, nClusters=6)

nLabels = dict()
for label in labels:
  if label not in nLabels:
    nLabels[label] = 1
  nLabels[label] += 1 

kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(importantDims)
centroids = kmeans.cluster_centers_

# Interpreting 'PCA' clusters

def n_important_components(n=None, point=None):
  """ 
  Returns the indices of the point's components and their percentage of 
  contribution to the norm.
  """
  if n is None:
    n = len(point)
  point = np.abs(point)
  percentages = np.array([])
  total = np.sum(point)
  if n > len(point):
    raise Exception('Value for n too big.')
  indices = np.argpartition(point, -n)[-n:]
  sortedIndices = indices[np.argsort(-point[indices])]
  percentages = np.round(point[sortedIndices] / total, 3)
  return sortedIndices, percentages

n_important_components(5, centroids[0])

def correlated_features(component, dfNames, nFeatures=None):
  """ 
  Returns a list with 'nFeatures' names of the most correlated features to the 
  specified principal component and their percentage of correlation associated.
  The percentage does not take into account if the correlation is positive or 
  negative. Instead, we just want to characterize what features are most important 
  to a component.
  """
  if nFeatures is None:
    nFeatures = len(dfNames)
  absComponent = np.abs(component)
  i = 0
  correlatedVars = []
  totalCorrelation = np.sum(absComponent)
  while i < nFeatures:
    mostCorrelated = np.argmax(absComponent)
    correlation = max(absComponent)
    absComponent[mostCorrelated] = -np.inf
    correlatedColumn = dfNames[mostCorrelated]
    correlatedVars.append((correlatedColumn, round(correlation / totalCorrelation, 3)))
    i += 1
  return correlatedVars

correlated_features(devPCA.components_[0], devs.columns[2:], 5)

def centroid_corr_with_features(centroid, dimRedObject, featureNames, nFeatures=None):
  """ 
  Returns the percentage of 'correlation' of a centroid in the dimensionality-reduced 
  space with the features of the original space. It does not take into account
  negative correlation, just indicates the magnitude of the correlation.
      for each PCi:
        get relevance
        get correlation with features (relevance * pctCorr(feature, PCi))
        add result to the correlation of the centroid with that feature
  """
  if nFeatures is None:
    nFeatures = len(featureNames)
  corrWithFeatures = {feature: 0 for feature in featureNames}

  indexes, importancePcts = n_important_components(len(centroid), centroid)
  for index, PCImportance in zip(indexes, importancePcts):
    featureCorrs = correlated_features(dimRedObject.components_[index], featureNames)
    for featureCorr in featureCorrs:
      corrWithFeatures[featureCorr[0]] += PCImportance * featureCorr[1]
  featureList = sorted(corrWithFeatures.items(), key=lambda k: k[1], reverse=True)[:nFeatures]
  featureArray = np.array([[feature, round(corr, 3)] for feature, corr in featureList])

  # plotting
  data = pd.DataFrame({'features': featureArray[:,0], 'correlation': featureArray[:,1]})
  chart = alt.Chart(data, title='Correlation of cluster with features').mark_bar().encode(
    x = alt.X('features:O', sort='-y'),
    y = alt.Y('correlation:Q')
  )
  return featureArray, chart

corrFeatures, chart = centroid_corr_with_features(centroids[0], devPCA, devs.columns[2:], 20)

def mean_developer(centroid, dimRedObject, scalerObject):
  """
  Returns the value of the features in the original space of the inversely transformed centroid.
  """
  # append 0s to conform with the number of components picked for the clutering
  appendedCentroid = np.append(centroid, np.linspace(0, 0, len(dimRedObject.components_)-len(centroid)))
  normOriginalPoint = dimRedObject.inverse_transform(appendedCentroid.reshape(1,-1))
  originalPoint = scalerObject.inverse_transform(normOriginalPoint.reshape(1,-1))
  meanDeveloper = {feature: round(value, 2) for feature, value in zip(devs.columns[2:], originalPoint[0])}
  return meanDeveloper

def important_values(centroid, dimRedObject, featureNames, scaler, nValues=None):
  """
  Returns the value of the nValues most important features of the mean developer for that cluster.
  """
  centroidCorrelatedFeatures, _ = centroid_corr_with_features(centroid, dimRedObject, featureNames, nValues)
  meanDeveloper = mean_developer(centroid, dimRedObject, scaler)
  if nValues is None:
    nValues = len(featureNames)
  # correlatedFeatures is sorted decreasingly
  return [[featureName, meanDeveloper[featureName]] for featureName in centroidCorrelatedFeatures[:,0]][:nValues]

important_values(centroids[0], devPCA, devs.columns[2:], minMaxScaler, 10)

clusterProfiles = []
for i in range(len(centroids)):
  clusterValues = np.array(important_values(centroids[i], devPCA, devs.columns[2:], minMaxScaler))
  clusterDf = pd.DataFrame(data=[clusterValues[:,1]], columns=clusterValues[:,0], index=['cluster'+str(i)])
  clusterProfiles.append(clusterDf)
clusterProfilesDf = pd.concat(clusterProfiles)
clusterProfilesDf

clusterProfilesDf.to_csv('../../data/interim/Modelling/clusterProfilesPCA.csv')

###############################################################################################################

# Dimensionality reduction with tSVD

tSVD = decomposition.TruncatedSVD(n_components=len(devs.columns[2:])-1, random_state=888)
devtSVD = tSVD.fit(devsScaled)
comps = devtSVD.components_
compsVariance = devtSVD.explained_variance_
compsVarRatio = devtSVD.explained_variance_ratio_

# Pick components that account for x% of variability
cumSum = 0
nComps = 0
x = 0.75
for varExplained in compsVarRatio:
  cumSum += varExplained
  if cumSum >= x:
    break
  else:
    nComps += 1

# Clustering over tSVD Principal Components

importantDims = tSVD.transform(devsScaled)[:,:nComps]

Nc = range(1, 20)
kmeans = [cluster.KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(importantDims).score(importantDims) for i in range(len(kmeans))]

labels, kmeans = fitKMeansAndPlot(importantDims, nClusters=5)

nLabels = dict()
for label in labels:
  if label not in nLabels:
    nLabels[label] = 1
  nLabels[label] += 1 

labels, kmeans = fitKMeansAndPlot(importantDims, nClusters=6)

nLabels = dict()
for label in labels:
  if label not in nLabels:
    nLabels[label] = 1
  nLabels[label] += 1

kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(importantDims)
centroids = kmeans.cluster_centers_

# Interpreting 'tSVD' clusters

centroids = kmeans.cluster_centers_
corrFeatures, chart = centroid_corr_with_features(centroids[0], devtSVD, devs.columns[2:], 20)

important_values(centroids[0], devtSVD, devs.columns[2:], minMaxScaler, 10)

clusterProfiles = []
for i in range(len(centroids)):
  clusterValues = np.array(important_values(centroids[i], devtSVD, devs.columns[2:], minMaxScaler))
  clusterDf = pd.DataFrame(data=[clusterValues[:,1]], columns=clusterValues[:,0], index=['cluster'+str(i)])
  clusterProfiles.append(clusterDf)
clusterProfilesDf = pd.concat(clusterProfiles)

clusterProfilesDf.to_csv('../../data/interim/Modelling/clusterProfilesTSVD.csv')

# Extra: comparing aggregated clusters vs. raw clusters

kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(importantDims)
centroids = kmeans.cluster_centers_

clusterProfiles = []
for i in range(len(centroids)):
  clusterValues = np.array(important_values(centroids[i], devtSVD, devs.columns[2:], minMaxScaler))
  clusterDf = pd.DataFrame(data=[clusterValues[:,1]], columns=clusterValues[:,0], index=['cluster'+str(i)])
  clusterProfiles.append(clusterDf)
clusterProfilesDf = pd.concat(clusterProfiles)

clusterProfilesDf.to_csv('../../data/interim/Modelling/3clusterProfilesTSVD.csv')