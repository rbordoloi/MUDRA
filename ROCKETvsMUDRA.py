import numpy as np
import pickle
import tensorly as tf
import pandas as pd
from skfda.representation.basis import BSplineBasis
from scipy.linalg import solve_sylvester, sqrtm
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from sktime.transformations.panel.rocket import Rocket
from sktime.transformations.panel.padder import PaddingTransformer
import pandas as pd
import argparse

rng = np.random.default_rng()

parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', default=500, type=int, help='Number of iterations for the ECM algorithms')
parser.add_argument('--avg', default=1000, type=int, help='Number of times to run the models to average over')
args = parser.parse_args()

def nearPSD(X):
    
    assert len(X.shape) == 2
    X = (X + X.T) / 2
    D, Q = np.linalg.eigh(X)
    D = np.clip(D, 0, np.inf)
    return Q @ np.diag(D) @ Q.T

def matSqrt(X):
    
    assert len(X.shape) == 2
    X = (X + X.T) / 2
    D, Q = np.linalg.eigh(X)
#     print(D)
    D = np.sqrt(np.clip(D, 0, np.inf))
#     print(D)
    return Q @ np.diag(D) @ Q.T

class MFLDA(BaseEstimator, TransformerMixin, ClassifierMixin):
    
    def __init__(self, r=2, nBasis=4, SigmaRank = None, PsiRank = None, splineOrder = 3, regularizer=1e-7, n_iter=1000, criterion='bic'):
        
        # Initializing all hyperparameters
        
        self.nBasis = nBasis #Number of spline basis functions used
        self.r = r #Number of dimensions to reduce to
        self.SigmaRank = SigmaRank #Rank of Sigma, or time autocovariance(used to avoid local maximas)
        self.PsiRank = PsiRank #Rank of Psi, or feature autocovariance(used to avoid local maximas)
        self.splineOrder = splineOrder #Order of the spline, deafult is 3
        self.regularizer = regularizer #Regularization value used to compute inverses of non-invertible matrices
        self.n_iter = n_iter #Number of iterations to run for the ECM algorithm
        self.criterion = 'bic'
        super().__init__()
        
    def preprocess(self, X):

        t = []
        f = []
        x = []
        for i in range(len(X)):
            t.append(X[i][1:, 0])
            f.append(X[i][0, 1:])
            x.append(X[i][1:,1:])
#                 t.append(X[i][1:, 0].get())
#                 f.append(X[i][0, 1:].get())
#                 x.append(X[i][1:,1:].get())

        return t, f, x

    def Estep(self, x, y, S, C):

        #Takes one sample and performs the E-Step for that sample to compute the corresponding gamma'
        
        #Set up the Sylvester equation
        beta = self.lambda0_ + self.Lambda_ @ np.diag(self.alpha_[y]) @ self.xi_
        PsiPrime = C.T @ self.thetaPsi_ @ np.diag(self.DPsi_) @ self.thetaPsi_.T @ C
        SigmaPrime = S @ self.thetaSigma_ @ np.diag(self.DSigma_) @ self.thetaSigma_.T @ S.T
        centeredX = (x - S @ beta @ C) @ PsiPrime.T
        
        #Solve the Sylvester equation
        gamma = solve_sylvester(self.sigma2_ * np.linalg.pinv(SigmaPrime), PsiPrime.T, centeredX)
        gamma = np.array(gamma)

        return gamma
        
    
    # Compute the parameters lambda0, Lambda, alpha, xi
    def MStepLinear(self, x, y, m, gammaPrime, S, C):
        
        #Define LHS of first equation as a function instead of a matrix multiply. This allows us to use GMRES
        #and solve the matrix equation without a large matrix inversion
        def mv(v, target):
            
            v = v.reshape(self.n_features, self.nBasis).T
            s = np.zeros((self.nBasis, self.n_features))
            impIdxs = np.where(y == target)[0] #Choose samples belonging to the current class
            for i in impIdxs:
                s += S[i].T @ S[i] @ v @ C[i] @ C[i].T #Multiply and add the corresponding S and C matrices
            s = s.T.flatten()
            return s
        
        #Convert the above function into a linear operator and solve using GMRES
        A = [LinearOperator((self.nBasis * self.n_features, self.nBasis * self.n_features), matvec=lambda x: mv(x, i)) for i in range(self.n_classes)]
        b = np.zeros((self.n_classes, self.nBasis, self.n_features))
        for i in range(len(x)):
            centeredX = x[i] - gammaPrime[i]
            b[y[i]] += S[i].T @ centeredX @ C[i].T
        beta = [gmres(A[i], b[i].T.flatten())[0].reshape(self.n_features, self.nBasis).T for i in range(self.n_classes)]
        lambda0 = np.average(beta, axis=0, weights=m) #Compute lambda0 by weighted average
        beta -= lambda0 #De-mean beta by subtracting lambda0
        
        #Compute the other linear params by a PARAFAC decompositon, similar to SVD
        weights, loadings = tf.decomposition.parafac(beta, self.r, normalize_factors=True)
        weights = np.diag(weights)
        alpha = np.zeros((self.n_classes, self.r))
        for i in range(self.n_classes):
            alpha[i] = weights @ loadings[0][i] #Convert the PARAFAC kernel to an alpha for each class
        Lambda = loadings[1]
        xi = loadings[2]
        
        
        return lambda0, Lambda, alpha, xi.T
    
    #Compute parameters Sigma, Psi. sigma2
    def MStepVar(self, x, y, gammaPrime, S, C, sigma2, lambda0, Lambda, alpha, xi):
        
        #Initialize some values in order to start iterations
        prevSigma2 = 1
        Sigma = np.eye(self.SigmaRank)
        Psi = np.eye(self.PsiRank)
        gamma = np.zeros((len(gammaPrime), self.nBasis, self.n_features))
        t = 0
        f = 0
        den = 0
        
        #Compute gammas to be used for computing Sigma and Psi
        for i in range(len(gammaPrime)):
            tmp = np.linalg.lstsq(C[i].T, gammaPrime[i].T, rcond=None)[0].T
            gamma[i] = np.linalg.lstsq(S[i], tmp, rcond=None)[0]
            t += x[i].shape[0]
            f += x[i].shape[1]
            den += x[i].flatten().shape[0]
        max_iter = 50
        currIter = 0
        
        #Run iterations until max_iter is reached, or convergence is achieved
        while ((np.abs(prevSigma2 - sigma2) / prevSigma2) > 1e-5) and (currIter < max_iter):
            prevSigma2 = sigma2 #Store current value of sigma2 to check for convergence
            
            # Estimate the autocovariance parameters
            SigmaNew = np.zeros((self.nBasis, self.nBasis))
            PsiNew = np.zeros((self.n_features, self.n_features))
            for i in range(len(X)):
#                 SigmaNew += gamma[i] @ np.linalg.lstsq(Psi + self.regularizer * np.eye(self.n_features), gamma[i].T)[0]
                SigmaNew += gamma[i] @ np.linalg.lstsq(Psi, gamma[i].T, rcond=1)[0]
            Sigma = nearPSD(SigmaNew / f)
            
            for i in range(len(X)):
#                 PsiNew += gamma[i].T @ np.linalg.lstsq(Sigma + self.regularizer * np.eye(self.nBasis), gamma[i])[0]
                PsiNew += gamma[i].T @ np.linalg.lstsq(Sigma, gamma[i], rcond=1)[0]
            Psi = nearPSD(PsiNew / t)
            
#             num = 0
#             for i in range(len(X)):
#                 num += np.trace(np.linalg.lstsq(Psi, gamma[i].T)[0] @ np.linalg.lstsq(Sigma, gamma[i])[0])
#             scale = num / den
#             Sigma *= np.sqrt(scale)
#             Psi *= np.sqrt(scale)
            
            #Compute sigma2
            num = 0
            for i in range(len(x)):
                num += np.linalg.norm(x[i] - S[i] @ (lambda0 + Lambda @ np.diag(alpha[y[i]]) @ xi) @ C[i] - gammaPrime[i]) + np.trace(C[i].T @ Psi @ C[i]) * np.trace(S[i] @ Sigma @ S[i].T)
#                 den += x[i].shape[0] * x[i].shape[1]
            sigma2 = np.clip(num / den, 1e-5, 1e5) #Ensure sigma2 is not too small or too big
            currIter += 1
        #Perform eigendecompositons in order to obtain reduced rank estimated via an Eckart-Young type theorem
#         print(scale)
#         print(DSigma, DPsi)
        
        return Sigma, Psi, sigma2        
    
    def fit(self, X, y):
                
        t, f, x = self.preprocess(X)
        
        #Initialize all parameters
        self.features_ = np.unique(np.concatenate(f))
        self.n_features = len(self.features_)
        self.tPoints = np.unique(np.concatenate(t))
        if self.SigmaRank == None:
            self.SigmaRank = self.nBasis
        if self.PsiRank == None:
            self.PsiRank = self.n_features
        self.splineDomain = (np.min(self.tPoints), np.max(self.tPoints))
        
        #Compute the spline basis functions
        self.basis = BSplineBasis(
            domain_range = self.splineDomain,
            n_basis = self.nBasis,
            order = self.splineOrder
        )
        
        #Compute spline and feature selection matrices S and C respectively
        self.CFull = np.eye(self.n_features)
        S = np.zeros(len(X)).tolist()
        C = np.zeros(len(X)).tolist()
        for i in range(len(f)):
            idx = np.array([]).astype('int')
            for j in f[i]:
                idx = np.append(idx, np.where(self.features_ == j))
            idx.sort()
            f[i] = idx.astype('int')
            S[i] = np.array(self.basis(t[i]).squeeze().T)
            C[i] = self.CFull[:,np.atleast_1d(f[i])]
        
        self.classes_, y, m = np.unique(y, return_inverse=True, return_counts=True)
        self.n_classes = len(self.classes_)
        self.lambda0_ = rng.random((self.nBasis, self.n_features))
        self.Lambda_ = rng.random((self.nBasis, self.r))
        self.alpha_ = rng.random((self.n_classes, self.r))
        self.xi_ = rng.random((self.r, self.n_features))
        self.sigma2_ = rng.random()
        self.thetaSigma_ = rng.random((self.nBasis, self.SigmaRank))
        self.thetaPsi_ = rng.random((self.n_features, self.PsiRank))
        self.DSigma_ = rng.random(self.SigmaRank)
        self.DPsi_ = rng.random(self.PsiRank)
        score = self.logLikelihood(t, f, x, y, self.lambda0_, self.Lambda_, self.alpha_, self.xi_, self.thetaSigma_ @ np.diag(self.DSigma_) @ self.thetaSigma_.T, self.thetaPsi_ @ np.diag(self.DPsi_) @ self.thetaPsi_.T, self.sigma2_) #Add a stopping criterion
        
        #Run the ECM algorithm for n_iter steps
        for iteration in range(self.n_iter):
            prevScore = score
            gammaPrime = []
            
            #E-Step
            #TODO: If possible parallelize to run for all samples at the same time
            for i in range(len(x)):
                gammaPrime.append(self.Estep(x[i], y[i], S[i], C[i]))

            #CM-Steps
            lambda0, Lambda, alpha, xi = self.MStepLinear(x, y, m, gammaPrime, S, C)
#             self.thetaSigma_, self.DSigma_, self.thetaPsi_, self.DPsi_, self.sigma2_ = self.MStepVar(x, y, gammaPrime, S, C, self.thetaSigma_ @ np.diag(self.DSigma_) @ self.thetaSigma_.T, self.thetaPsi_ @ np.diag(self.DPsi_) @ self.thetaPsi_.T, self.sigma2_)
            
            Sigma, Psi, sigma2 = self.MStepVar(x, y, gammaPrime, S, C, self.sigma2_, lambda0, Lambda, alpha, xi)
            
            #Compute the stopping criterion and stop if criterion is fulfilled
#             if score > prevScore and iteration >= 5:
#                 print(iteration)
#                 break
            
            score = self.logLikelihood(t, f, x, y, lambda0, Lambda, alpha, xi, Sigma, Psi, sigma2)
            if score > prevScore and iteration > 0:
                break
        
            try:
                DSigma, thetaSigma = np.linalg.eigh(Sigma)
                DPsi, thetaPsi = np.linalg.eigh(Psi)
            except np.linalg.LinAlgError:
                print(Sigma, '\n', Psi)
                raise ValueError
            self.thetaSigma_ = thetaSigma[:,-self.SigmaRank:]
            self.thetaPsi_ = thetaPsi[:,-self.PsiRank:]
    #         print(DSigma, DPsi)
            self.DSigma_ = DSigma[-self.SigmaRank:]
            self.DPsi_ = DPsi[-self.PsiRank:]
            self.lambda0_ = lambda0
            self.Lambda_ = Lambda
            self.alpha_ = alpha
            self.xi_ = xi
            self.sigma2_ = sigma2
        
        return self
    
    #Stopping criterion
    def logLikelihood(self, t, f, x, y, lambda0, Lambda, alpha, xi, Sigma, Psi, sigma2):
        r = 0
        for i in range(len(x)):
            S = self.basis(t[i]).squeeze().T
            C = self.CFull[:,f[i].astype('int')]
            SigmaPrime = S @ Sigma @ S.T
            PsiPrime = C.T @ Psi @ C
            currX = (x[i] - S @ (lambda0 + Lambda @ np.diag(alpha[y[i]]) @ xi) @ C).T.flatten()
            var = sigma2 * np.eye(t[i].shape[0] * f[i].shape[0]) + np.kron(PsiPrime, SigmaPrime)
            r += currX @ np.linalg.lstsq(var, currX, rcond=None)[0] + np.linalg.slogdet(var)[1]
        return r/2
    
    #Compute a low-dimensional representation of new data
    def transform(self, X):
        
        t, f, x = self.preprocess(X)
        Y = np.zeros((len(X), self.r * self.r)) #target low dimension matrix
        
        #Compute the variance parameters
        Sigma = self.thetaSigma_ @ np.diag(self.DSigma_) @ self.thetaSigma_.T
        Psi = self.thetaPsi_ @ np.diag(self.DPsi_) @ self.thetaPsi_.T
        
        for i in range(len(X)):
            #Compute S and C matrices
            S = self.basis(t[i]).squeeze().T
            idx = np.array([])
            for j in f[i]:
                idx = np.append(idx, np.where(self.features_ == j)).astype('int')
            idx.sort()
            C = self.CFull[:,np.atleast_1d(idx)]
            
            #Compute variances for the current sample
            SigmaPrime = np.atleast_2d(S @ Sigma @ S.T)
            PsiPrime = np.atleast_2d(C.T @ Psi @ C)
            
            #Compute \hat{alpha}_Y
            centeredX = x[i] - S @ self.lambda0_ @ C
#             print(S @ self.lambda0_ @ C)
            centeredX = np.linalg.lstsq(PsiPrime, centeredX.T, rcond=None)[0].T
            centeredX = solve_sylvester(SigmaPrime, self.sigma2_ * np.linalg.pinv(PsiPrime).T, centeredX)
            centeredX = self.Lambda_.T @ S.T @ centeredX @ C.T @ self.xi_.T
            M = self.sigma2_ * np.eye(t[i].shape[0] * f[i].shape[0]) + np.kron(PsiPrime, SigmaPrime)
            A = np.kron(C.T @ self.xi_.T, S @ self.Lambda_)
            var = A.T @ np.linalg.lstsq(M, A, rcond=None)[0]
            alpha = np.linalg.lstsq(matSqrt(var), centeredX.T.flatten(), rcond=None)[0]
            Y[i] = alpha
#             Y[i] = np.diag(alpha.reshape(self.r, self.r).T) #Add it to Y
            
        return Y
    
    def fit_transform(self, X, y):
        
        #Fit on new data and then transform
        self.fit(X, y)
        return self.transform(X)
    
    def predict_log_proba(self, X):
        
        #Compute partial log likelihood for new data
        
        t, f, x = self.preprocess(X)
        predProbs = np.zeros((len(X), self.n_classes))
        Sigma = self.thetaSigma_ @ np.diag(self.DSigma_) @ self.thetaSigma_.T
        Psi = self.thetaPsi_ @ np.diag(self.DPsi_) @ self.thetaPsi_.T
        
        for i in range(len(X)):
            
            #Compute S and C matrices
            S = self.basis(t[i]).squeeze().T
            idx = []
            for j in f[i]:
                idx = np.append(idx, np.where(self.features_ == j)).astype('int')
            idx.sort()
            C = self.CFull[:,np.atleast_1d(idx)]
            
            #Compute the variances for each sample
            SigmaPrime = np.atleast_2d(S @ Sigma @ S.T)
            PsiPrime = np.atleast_2d(C.T @ Psi @ C)
            centeredX = x[i] - S @ self.lambda0_ @ C
            centeredX = np.linalg.lstsq(PsiPrime, centeredX.T, rcond=None)[0].T
            centeredX = solve_sylvester(SigmaPrime, self.sigma2_ * np.linalg.pinv(PsiPrime).T, centeredX)
            centeredX = self.Lambda_.T @ S.T @ centeredX @ C.T @ self.xi_.T
            M = self.sigma2_ * np.eye(t[i].shape[0] * f[i].shape[0]) + np.kron(PsiPrime, SigmaPrime)
            A = np.kron(C.T @ self.xi_.T, S @ self.Lambda_)
            var = sqrtm(A.T @ np.linalg.lstsq(M, A, rcond=None)[0])
            #Compute \hat{\alpha}_Y
            alpha = np.linalg.lstsq(var, centeredX.T.flatten(), rcond=None)[0].reshape(self.r, self.r).T
            
            for j in range(self.n_classes):
                #Compute corresponding alpha_i values and find distance
                target = (var @ np.diag(self.alpha_[j]).T.flatten()).reshape(self.r, self.r).T
                predProbs[i, j] = -np.linalg.norm(alpha - target)
        
        return predProbs
    
    def predict_proba(self, X):
        
        #Compute the actual class likelihoods
        #This is done be getting the partial log-likelihoods and applying a softmax to normalize
        
        logProbs = self.predict_log_proba(X)
        return softmax(logProbs, axis=1)
    
    def predict(self, X):
        
        NegLogProbs = -self.predict_log_proba(X) #Get partial log likelihoods
        classIdx = np.argmin(NegLogProbs, axis=1) #Choose class with maximum log likelihood
        return self.classes_[classIdx]
    
    def ic(self, X, y):
        
        if self.criterion != 'bic':
            raise NotImplementedError
        
        t, f, x = self.preprocess(X)
        for i in range(len(y)):
            y[i] = np.where(y[i] == self.classes_)[0][0]
        l = self.logLikelihood(t, f, x, y)
        numLinearParams = self.nBasis * self.r + self.r * self.n_classes + self.r * self.n_features
        numVarParams = (self.nBasis + 1) * self.SigmaRank + (self.n_features + 1) * self.PsiRank + 1
        return ((numLinearParams + numVarParams) * len(y) + 2 * l)

X, y = pickle.load(open("trainWordRecognition.pickle", 'rb'))
X_test, y_test = pickle.load(open("testWordRecognitionReduced.pickle", 'rb'))
XMissing, yMissing = pickle.load(open('trainWordRecognitionMissing.pickle', 'rb'))
XMissing_test, yMissing_test = pickle.load(open('testWordRecognitionReducedMissing.pickle', 'rb'))

paddedX = []
for i in range(len(X)):
    currX = XMissing[i]
    newX = currX[0,:]
    prevTPoint = -1
    for idx, tPoint in enumerate(currX[1:,0]):
        if tPoint - prevTPoint > 1:
            for j in range((tPoint - prevTPoint - 1).astype('int')):
                newX = np.vstack((newX, np.zeros(currX.shape[1])))
        newX = np.vstack((newX, currX[idx+1,:]))
    #             print(idx, tPoint, currX[idx+1,:])
        prevTPoint = tPoint
    if prevTPoint < 11.:
        for j in range((11 - prevTPoint).astype('int')):
            newX = np.vstack((newX, np.zeros(currX.shape[1])))
    newX[1:,0] = np.arange(12)
    # print(newX)
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
    
paddedX_test = []
for i in range(len(X_test)):
    currX = XMissing_test[i]
    newX = currX[0,:]
    prevTPoint = -1
    for idx, tPoint in enumerate(currX[1:,0]):
        if tPoint - prevTPoint > 1:
            for j in range((tPoint - prevTPoint - 1).astype('int')):
                newX = np.vstack((newX, np.zeros(currX.shape[1])))
        newX = np.vstack((newX, currX[idx+1,:]))
    #             print(idx, tPoint, currX[idx+1,:])
        prevTPoint = tPoint
    if prevTPoint < 11.:
        for j in range((11 - prevTPoint).astype('int')):
            newX = np.vstack((newX, np.zeros(currX.shape[1])))
    newX[1:,0] = np.arange(12)
    # print(newX)
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
    paddedX_test.append(newX)
    
paddedX2 = []
for i in range(len(XMissing)):
    currX  = XMissing[i]
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
    paddedX2.append(newX)
    
paddedX2_test = []
for i in range(len(XMissing_test)):
    currX  = XMissing_test[i]
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
    paddedX2_test.append(newX)

accTable = np.zeros((8, 7))
for r in range(2, 10):
    pureModelAccuracy = 0
    modelAccuracy = 0
    rocketAccuracy = 0
    pureModelAccuracyMissing = 0
    modelAccuracyMissing = 0
    rocketAccuracyMissing1 = 0
    rocketAccuracyMissing2 = 0
#     model = MFLDA(r=r, n_iter=arg.n_iter, nBasis=9)
    modelPipeline = make_pipeline(MFLDA(r=r, n_iter=args.n_iter, nBasis=9), RidgeClassifierCV())
    rocketPipeline = make_pipeline(Rocket(num_kernels=(r**2 + 1)//2), RidgeClassifierCV())
    rocketPaddedPipeline = make_pipeline(PaddingTransformer(), Rocket(num_kernels=(r**2 + 1)//2), RidgeClassifierCV())
    for _ in tqdm(range(args.avg)):
        x = pd.DataFrame([[pd.Series(X[j][1:,i]) for i in range(1, 10)] for j in range(275)])
        rocketPipeline.fit(x, y)
        x = pd.DataFrame([[pd.Series(X_test[j][1:,i]) for i in range(1, 10)] for j in range(300)])
        rocketAccuracy += classification_report(y_test, rocketPipeline.predict(x), output_dict=True)['weighted avg']['f1-score']
        modelPipeline.fit(X, y)
        pureModelAccuracy += classification_report(y_test, modelPipeline[0].predict(X_test), output_dict=True)['weighted avg']['f1-score']
        modelAccuracy += classification_report(y_test, modelPipeline.predict(X_test), output_dict=True)['weighted avg']['f1-score']
        x = pd.DataFrame([[pd.Series(paddedX[j][1:,i]) for i in range(1, 10)] for j in range(275)])
        rocketPipeline.fit(x, y)
        x = pd.DataFrame([[pd.Series(paddedX_test[j][1:,i]) for i in range(1, 10)] for j in range(300)])
        rocketAccuracyMissing1 += classification_report(yMissing_test, rocketPipeline.predict(x), output_dict=True)['weighted avg']['f1-score']
        modelPipeline.fit(XMissing, yMissing)
        pureModelAccuracyMissing += classification_report(yMissing_test, modelPipeline[0].predict(XMissing_test), output_dict=True)['weighted avg']['f1-score']
        modelAccuracyMissing += classification_report(yMissing_test, modelPipeline.predict(XMissing_test), output_dict=True)['weighted avg']['f1-score']
        
        x = pd.DataFrame([[pd.Series(paddedX2[j][1:,i]) for i in range(1, 10)] for j in range(275)])
        rocketPaddedPipeline.fit(x, y)
        x = pd.DataFrame([[pd.Series(paddedX2_test[j][1:,i]) for i in range(1, 10)] for j in range(300)])
        rocketAccuracyMissing2 += classification_report(yMissing_test, rocketPaddedPipeline.predict(x), output_dict=True)['weighted avg']['f1-score']
        
    pureModelAccuracy /= args.avg
    modelAccuracy /= args.avg
    rocketAccuracy /= args.avg
    pureModelAccuracyMissing /= args.avg
    modelAccuracyMissing /= args.avg
    rocketAccuracyMissing1 /= args.avg
    rocketAccuracyMissing2 /= args.avg
    
    accTable[r - 2,:] = [pureModelAccuracy, modelAccuracy, rocketAccuracy, pureModelAccuracy, modelAccuracyMissing, rocketAccuracyMissing1, rocketAccuracyMissing2]
pd.DataFrame(accTable, index=range(2, 10), columns=['MFLDA classifier', 'MFLDA with RidgeClassifierCV', 'ROCKET', 'MFLDA classifier on missing data', 'MFLDA with RidgeClassifierCV on missing data', 'ROCKET with imputation padding', 'ROCKET with end padding']).to_csv('rocket_vs_mflda3.csv')