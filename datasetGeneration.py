from scipy.io import arff
from io import StringIO
import pandas as pd
import numpy as np
import pickle
from urllib.request import urlretrieve
import zipfile

rng = np.random.default_rng()

path, _ = urlretrieve('http://www.timeseriesclassification.com/aeon-toolkit/ArticularyWordRecognition.zip')
dataFiles = zipfile.ZipFile(path)

with dataFiles.open('ArticularyWordRecognition_TRAIN.arff') as arffFile, open('datasets/train.arff', 'wb') as f:
    f.write(arffFile.read())
with dataFiles.open('ArticularyWordRecognition_TEST.arff') as arffFile, open('datasets/test.arff', 'wb') as f:
    f.write(arffFile.read())

data = arff.loadarff('datasets/train.arff')[0]
X, y = [], []
ts = np.atleast_2d(np.arange(12)).T
fs = np.arange(9)
fs = np.hstack((np.nan, fs))
for i in range(len(data)):
    currX, currY = pd.DataFrame(data[i][0]).T.to_numpy()[::12], int(float(data[i][1]))
    currX = np.hstack((ts, currX))
    currX = np.vstack((fs, currX))
    X.append(currX)
    y.append(currY)
dataOut = (X, y)
pickle.dump(dataOut, open('datasets/trainWordRecognition.pickle', 'wb'))

for i in range(len(X)):
    tChoices = np.sort(rng.choice(10, rng.integers(5, 11), replace=False)) + 2
    tChoices = np.hstack((0, 1, tChoices, 12))
    fChoices = np.sort(rng.choice(9, rng.integers(4, 10), replace=False)) + 1
    fChoices = np.hstack((0, fChoices))
    X[i] = X[i][tChoices, :]
    X[i] = X[i][:, fChoices]

dataOut = (X, y)
pickle.dump(dataOut, open('datasets/trainWordRecognitionMissing.pickle', 'wb'))

data = arff.loadarff('datasets/test.arff')[0]
X, y = [], []
for i in range(len(data)):
    currX, currY = pd.DataFrame(data[i][0]).T.to_numpy()[::12], int(float(data[i][1]))
    currX = np.hstack((ts, currX))
    currX = np.vstack((fs, currX))
    X.append(currX)
    y.append(currY)
dataOut = (X, y)
pickle.dump(dataOut, open('datasets/testWordRecognitionReduced.pickle', 'wb'))

for i in range(len(X)):
    tChoices = np.sort(rng.choice(10, rng.integers(5, 11), replace=False)) + 2
    tChoices = np.hstack((0, 1, tChoices, 12))
    fChoices = np.sort(rng.choice(9, rng.integers(4, 10), replace=False)) + 1
    fChoices = np.hstack((0, fChoices))
    X[i] = X[i][tChoices, :]
    X[i] = X[i][:, fChoices]

dataOut = (X, y)
pickle.dump(dataOut, open('datasets/testWordRecognitionReducedMissing.pickle', 'wb'))

X, y = [], []
ts = np.atleast_2d(np.linspace(0, 12, 144, endpoint=False)).T
fs = np.arange(9)
fs = np.hstack((np.nan, fs))
for i in range(len(data)):
    currX, currY = pd.DataFrame(data[i][0]).T.to_numpy(), int(float(data[i][1]))
    currX = np.hstack((ts, currX))
    currX = np.vstack((fs, currX))
    X.append(currX)
    y.append(currY)

data = (X, y)
pickle.dump(dataOut, open('datasets/testWordRecognition.pickle', 'wb'))

for i in range(len(X)):
    tChoices = np.sort(rng.choice(142, rng.integers(1, 143), replace=False)) + 2
    tChoices = np.hstack((0, 1, tChoices, 144))
    fChoices = np.sort(rng.choice(9, rng.integers(2, 10), replace=False)) + 1
    fChoices = np.hstack((0, fChoices))
    X[i] = X[i][tChoices, :]
    X[i] = X[i][:, fChoices]

dataOut = (X, y)
pickle.dump(dataOut, open('datasets/testWordRecognitionMissing.pickle', 'wb'))
