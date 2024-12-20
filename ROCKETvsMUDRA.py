import numpy as np
import pickle
import tensorly as tf
import pandas as pd
from skfda.representation.basis import BSplineBasis
from scipy.linalg import solve_sylvester, sqrtm
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from sktime.transformations.panel.rocket import Rocket
from sktime.transformations.panel.padder import PaddingTransformer
import pandas as pd
import argparse

from MUDRA import MUDRA

rng = np.random.default_rng()

parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', default=500, type=int, help='Number of iterations for the ECM algorithms')
parser.add_argument('--avg', default=1000, type=int, help='Number of times to run the models to average over')
args = parser.parse_args()

def imputation_padding(X):
    paddedX = []
    for i in range(len(X)):
        currX = X[i]
        newX = currX[0,:]
        prevTPoint = -1
        for idx, tPoint in enumerate(currX[1:,0]):
            if tPoint - prevTPoint > 1:
                for j in range((tPoint - prevTPoint - 1).astype('int')):
                    newX = np.vstack((newX, np.zeros(currX.shape[1])))
            newX = np.vstack((newX, currX[idx+1,:]))
            prevTPoint = tPoint
        if prevTPoint < 11.:
            for j in range((11 - prevTPoint).astype('int')):
                newX = np.vstack((newX, np.zeros(currX.shape[1])))
        newX[1:,0] = np.arange(12)
        currX = newX
        newX = np.atleast_2d(currX[:,0]).T
        prevFeat = -1
        for idx, feat in enumerate(currX[0,1:]):
            if feat - prevFeat > 1:
                for j in range((feat - prevFeat - 1).astype('int')):
                    newX = np.hstack((newX, np.zeros((currX.shape[0], 1))))
            newX = np.hstack((newX, np.atleast_2d(currX[:,idx+1]).T))
            prevFeat = feat
        if prevFeat < 8:
            for j in range((8 - prevFeat).astype('int')):
                newX = np.hstack((newX, np.zeros((currX.shape[0], 1))))
        newX[0,1:] = np.arange(9)
        paddedX.append(newX)
    return paddedX

def end_padding(X):
    paddedX = []
    for i in range(len(X)):
        currX  = X[i]
        newX = np.atleast_2d(currX[:,0]).T
        prevFeat = -1
        for idx, feat in enumerate(currX[0,1:]):
            if feat - prevFeat > 1:
                for j in range((feat - prevFeat - 1).astype('int')):
                    newX = np.hstack((newX, np.zeros((currX.shape[0], 1))))
            newX = np.hstack((newX, np.atleast_2d(currX[:,idx+1]).T))
            prevFeat = feat
        if prevFeat < 8:
            for j in range((8 - prevFeat).astype('int')):
                newX = np.hstack((newX, np.zeros((currX.shape[0], 1))))
        newX[0,1:] = np.arange(9)
        paddedX.append(newX)
    return paddedX

def locf(X):
    
    for i in range(len(X)):
        currX = X[i].copy()
        newX = currX[0:2, :].copy()
        prevTPoint = 0
        for row in range(2, currX.shape[0]):
            if currX[row, 0] - prevTPoint > 1:
                for j in np.arange(currX[row, 0] - prevTPoint - 1):
                    newX = np.vstack((newX, newX[-1, :]))
                    newX[-1, 0] = newX[-1, 0] + 1
            newX = np.vstack((newX, currX[row, :]))
            prevTPoint = newX[-1, 0]
        currX = newX.copy()
        newX = currX[:,0].reshape(-1, 1).copy()
        prevCol = -1
        for col in range(1, currX.shape[1]):
            if currX[0, col] - prevCol > 1:
                for j in np.arange(prevCol + 1, currX[0, col]):
                    newCol = np.zeros((newX.shape[0], 1))
                    newCol[0, 0] = j
                    newX = np.hstack((newX, newCol))
            newX = np.hstack((newX, currX[:, col].reshape(-1, 1)))
            prevCol = newX[0, -1]
        if newX[0, -1] < 8:
            newX = np.hstack((newX, np.zeros((newX.shape[0], (8 - newX[0, -1]).astype('int')))))
        newX[0, 1:] = np.arange(9)
        X[i] = newX

    return X
    
def convertToDataFrame(X):
    X_df = []
    tmp = np.zeros(9)
    tmp[:] = np.nan
    for i in range(len(X)):
        X_df.append(tmp.tolist())
        for j in range(1, X[i].shape[1]):
            X_df[-1][X[i][0,j].astype('int')] = pd.Series(X[i][1:,j], index=X[i][1:,0])
    return pd.DataFrame(X_df)


X, y = pickle.load(open("datasets/trainWordRecognition.pickle", 'rb'))
X_test, y_test = pickle.load(open("datasets/testWordRecognitionReduced.pickle", 'rb'))
XMissing, yMissing = pickle.load(open('datasets/trainWordRecognitionMissing.pickle', 'rb'))
XMissing_test, yMissing_test = pickle.load(open('datasets/testWordRecognitionReducedMissing.pickle', 'rb'))
imputePaddedX = convertToDataFrame(imputation_padding(XMissing))
imputePaddedX_test = convertToDataFrame(imputation_padding(XMissing_test))
locfImputedX = convertToDataFrame(locf(XMissing))
locfImputedX_test = convertToDataFrame(locf(XMissing_test))
X = convertToDataFrame(X)
X_test = convertToDataFrame(X_test)
XMissing = convertToDataFrame(XMissing)
XMissing_test = convertToDataFrame(XMissing_test)


accTable = np.zeros((8, 8))
for r in range(2, 10):
    pureModelAccuracy = 0
    modelAccuracy = 0
    rocketAccuracy = 0
    pureModelAccuracyMissing = 0
    modelAccuracyMissing = 0
    rocketAccuracyMissing1 = 0
    rocketAccuracyMissing2 = 0
    rocketAccuracyMissing3 = 0
    modelPipeline = make_pipeline(MUDRA(r=r, n_iter=args.n_iter, nBasis=9), RidgeClassifierCV())
    rocketPipeline = make_pipeline(Rocket(num_kernels=(r**2 + 1)//2), RidgeClassifierCV())
    rocketPaddedPipeline = make_pipeline(PaddingTransformer(), Rocket(num_kernels=(r**2 + 1)//2), RidgeClassifierCV())
    for _ in tqdm(range(args.avg)):
        rocketPipeline.fit(X, y)
        accTable[r-2, 2] += f1_score(y_test, rocketPipeline.predict(X_test), average='weighted')
        modelPipeline.fit(X, y)
        accTable[r-2, 0] += f1_score(y_test, modelPipeline[0].predict(X_test), average='weighted')
        accTable[r-2, 1] += f1_score(y_test, modelPipeline.predict(X_test), average='weighted')
        
        rocketPipeline.fit(imputePaddedX, y)
        accTable[r-2, 5] += f1_score(yMissing_test, rocketPipeline.predict(imputePaddedX_test), average='weighted')
        modelPipeline.fit(XMissing, yMissing)
        accTable[r-2, 3] += f1_score(yMissing_test, modelPipeline[0].predict(XMissing_test), average='weighted')
        accTable[r-2, 4] += f1_score(yMissing_test, modelPipeline.predict(XMissing_test), average='weighted')
        
        rocketPaddedPipeline.fit(XMissing, yMissing)
        accTable[r-2, 6] += f1_score(yMissing_test, rocketPaddedPipeline.predict(XMissing_test), average='weighted')

        rocketPipeline.fit(locfImputedX, yMissing)
        accTable[r-2, 7] += f1_score(yMissing_test, rocketPipeline.predict(locfImputedX_test), average='weighted')
        
accTable/= args.avg
    
pd.DataFrame(accTable[:, -1], index=range(2, 10), columns=['ROCKET with LOCF imputation']).to_csv('rocket_vs_mflda.csv')