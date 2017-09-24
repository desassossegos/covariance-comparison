from sklearn.covariance import empirical_covariance
import pandas as pd
import numpy as np

def GarciaTest(D):
    """
    Source: Garcia, Carlos. “A Simple Procedure for the Comparison of Covariance Matrices.” BMC Evolutionary Biology 12 (2012): 222. PMC. Web. 24 Sept. 2017.
    """
    covs = []
    for ma in D:
        #get covariance matrix for each sample matrix
        covs.append(empirical_covariance(ma))
    egvectors = []
    var = []
    for c in covs:
        egvectors.append(np.linalg.eig(c)[1]) #get eigenvectors
        var.append(np.diagonal(c)) #get variance in each sample for later use in defining S1_max
    X = np.array(egvectors)
    S1_max = 2*sum(2*var[0]**2+2*var[1]**2)
    
    # delete lists that are no longer needed
    del covs[:]
    del egvectors[:]
    del var[:]
    
    #define v's (not sure if these are right!)
    v11 = np.diagonal(empirical_covariance(np.dot(D[0],X[0])))
    v12 = np.diagonal(empirical_covariance(np.dot(D[0],X[1])))
    v21 = np.diagonal(empirical_covariance(np.dot(D[1],X[0])))
    v22 = np.diagonal(empirical_covariance(np.dot(D[1],X[1])))
    
    #calculate S's and S1 max (not sure if these are right!)
    S1 = 2 * sum(((v11-v21)**2)+((v12-v22)**2))
    S2 = sum(((v11+v22)-(v12+v21))**2)
    S3 = sum(((v11+v12)-(v21+v22))**2)
    S1_ratio = S1/S1_max
    
    result = [S1, S2, S3, S1_max, S1_ratio]
    return result    
