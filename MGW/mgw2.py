import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl
import scipy.spatial as sps
import ot
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import sklearn.mixture as sklmi
import sklearn.manifold as sklma
from tqdm import tqdm
import warnings
import os, sys
import itertools
import scipy as sp
from sklearn.datasets import make_blobs
from scipy.stats import special_ortho_group
from sklearn.neighbors import NearestNeighbors



### utils ###

class Gaussian:
    ### class encoding a Gaussian distribution ###
    
    def __init__(self,mean,cov):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.dim = len(mean)

    def sample(self,n_samples):
        return np.random.multivariate_normal(self.mean,self.cov,n_samples)
    
    def pdf(self,x):
        return multivariate_normal.pdf(x,mean=self.mean,cov=self.cov)



class GaussianMixture:
    ### Encode a Gaussian distribution ###
    
    def __init__(self,weights_list,mean_list,cov_list):
        self.comp = [Gaussian(mean_list[k],cov_list[k]) for k in range(len(mean_list))]
        self.comp_mean = np.array(mean_list)
        self.comp_cov = np.array(cov_list)
        self.weights = np.array(weights_list)
        self.dim = len(mean_list[0])
        self.K = len(mean_list)

    
    def sample(self,n_samples,return_K=False):
        """
        if return_K, return the list of component indice for each sample
        """
        idx_list = np.random.choice(np.arange(self.K),n_samples,p=self.weights)
        samples_list = [np.random.multivariate_normal(self.comp_mean[i],self.comp_cov[i]) for i in idx_list]
        if return_K:
            return np.array(samples_list), idx_list
        else:
            return np.array(samples_list)

    def mean(self):
        ### return the mean of the Gaussian mixture
        return np.average(self.comp_mean,axis=0,weights=self.weights)
    
    def cov(self):
        ### return the covariance matrix of the Gaussian mixture
        A = sum([self.weights[k]*(self.comp_cov[k]+np.outer(self.comp_mean[k],self.comp_mean[k])) for k in range(self.K)])
        mean = self.mean()
        return A - np.outer(mean,mean)


    def pdf(self,x):
        return sum([self.weights[k]*multivariate_normal.pdf(x,mean=self.comp_mean[k],cov=self.comp_cov[k]) for k in range(self.K)])


    def plot_scatter_2d(self,n_samples,T=None,*args):
        ### utils for Figures 3 and 4
        X, idx_list = self.sample(n_samples,True)
        grad_color = np.array([(x/np.max(X[:,0]),y/np.max(X[:,1]),x/np.max(X[:,0]),1) for x,y in X])
        if T is not None: 
            Y = np.array([T(X[k],idx_list[k],*args) for k in range(len(X))])
        else:
            Y = X
        cmap_list = ['spring', 'summer', 'autumn', 'winter', 'cool',
                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','viridis', 'plasma', 'inferno', 'magma', 'cividis']
        for k in range(self.K):
            Z = Y[idx_list==k]
            plt.scatter(Z[:,0],Z[:,1],color=X[idx_list==k][:,0],s=10)
        plt.axis('equal')



    def pdist(self,f): 
        ### compute pairwise distances matrix 
        dist_list = [[f(self.comp_mean[k],self.comp_mean[l],self.comp_cov[k],self.comp_cov[l]) for l in range(k+1,self.K)] for k in range(self.K)][:-1]
        dist_matrix = np.zeros((self.K,self.K))
        for k in range(self.K):
            for l in range(k+1,self.K):
                dist_matrix[k,l] = dist_list[k][l-k-1]
                dist_matrix[l,k] = dist_matrix[k,l]
        return dist_matrix

    





def gmm_transform(mu,P=None,b=None):
    ### Apply a linear transformation x -> Px + b to the Gaussian mixture
    if P is None:
        P = np.eye(mu.dim)
    if b is None:
        b = np.zeros(P.shape[0])
    mean_list = [P.dot(m) + b for m in mu.comp_mean]
    cov_list = [P@A@P.T for A in mu.comp_cov] 
    return GaussianMixture(mu.weights,mean_list,cov_list)


def I(m,n):
    ### rectangular matrix (I_n  0)^T 
    I = np.zeros((m,n))
    I[:n,:n] = np.eye(n)
    return I


def proj_stiefel(A):
    ### projection on Stiefel Manifold
    m,n = A.shape
    U, s, VT = spl.svd(A)
    return U @ I(m,n) @ VT


def GaussianW2(mean_0,mean_1,Cov_0,Cov_1):
    ### W_2 between Gaussian distributions
    sqCov_1 = spl.sqrtm(Cov_1).real
    return spl.norm(mean_0 - mean_1)**2 + np.trace(Cov_0) + np.trace(Cov_1) - 2*np.trace(spl.sqrtm(sqCov_1@Cov_0@sqCov_1).real)






def grad_gauss(P,Cov_0,Cov_1):
    ### grad of P -> tr(Cov_1^{\frac{1}{2}}P^TCov_0PCov_1^{\frac{1}{2})
    sqCov_1 = spl.sqrtm(Cov_1).real
    A = Cov_0@P@sqCov_1 
    B = spl.inv(spl.sqrtm(sqCov_1@P.T@A)).real 
    return A @ B @ sqCov_1 

    

def grad(P,weights,mu,nu):
    ### grad of P -> \sum_{k,l} weights[k,l]W_2^2(\mu_k,P_{\#}\nu_l)
    return 2*sum([weights[k,l]*(-np.outer(mu.comp_mean[k],nu.comp_mean[l]) 
                + P@np.outer(nu.comp_mean[l],nu.comp_mean[l]) 
                - grad_gauss(P,mu.comp_cov[k],nu.comp_cov[l])) for k in range(mu.K) for l in range(nu.K)])




def proj_gradient_descent(P,weights,mu,nu,alpha,n_iter=150):
    ### projected gradient descent on the Stiefel manifold
    loss = [sum([weights[k,l]*GaussianW2(0,0,mu.comp_cov[k],P@nu.comp_cov[l]@P.T) for k in range(mu.K) for l in range(nu.K)])]
    for k in range(n_iter):
        P = proj_stiefel(P-alpha*grad(P,weights,mu,nu))
        loss.append(sum([weights[k,l]*GaussianW2(0,0,mu.comp_cov[k],P@nu.comp_cov[l]@P.T) for k in range(mu.K) for l in range(nu.K)]))
    return P, loss



def T_map_Gaussian(x,mu,nu):
    ### Transport map between two Gaussian distributions mu and nu 
    invcov0 = spl.inv(spl.sqrtm(mu.cov).real)
    sqcov0 = spl.sqrtm(mu.cov).real
    A = invcov0@spl.sqrtm(sqcov0@nu.cov@sqcov0)@invcov0
    if len(x.shape)==1:
        return nu.mean + A@(x - mu.mean)
    else:
        return nu.mean + np.einsum('ij,bj -> bi',A,x - mu.mean)


def T_map_Gaussian_rand(x,mu,nu,idx_list): 
    ### Transport map between two Gaussian distributions mu and nu (utils for T_rand)
    invcov0 = [spl.inv(spl.sqrtm(mu.comp[k].cov).real) for k in range(mu.K)]
    sqcov0 = [spl.sqrtm(mu.comp[k].cov).real for k in range(mu.K)]
    A_list = [[invcov0[k]@spl.sqrtm(sqcov0[k]@nu.comp[l].cov@sqcov0[k])@invcov0[k] for l in range(nu.K)] for k in range(mu.K)]
    return np.array([nu.comp[idx[1]].mean + A_list[idx[0]][idx[1]]@(x[i] - mu.comp[idx[0]].mean) for i, idx in enumerate(idx_list)])

 

def T_mean(X,mu,nu,P,weights):
    mu_P_centered = gmm_transform(mu,P=P.T,b=-P.T@mu.mean())
    nu_centered = gmm_transform(nu,b=-nu.mean())
    pdf_comp = [mu.comp[k].pdf(X) for k in range(mu.K)]
    ipdf_mu = 1./mu.pdf(X)
    if len(X.shape)==1:
        X_P = P.T@(X - mu.mean())
        Y = sum([weights[k,l]*pdf_comp[k]*T_map_Gaussian(X_P,mu_P_centered.comp[k],nu_centered.comp[l]) for k in range(mu.K) for l in range(nu.K)])*ipdf_mu
    else:
        X_P = np.einsum('ij,bj->bi',P.T,X - mu.mean())
        Y = np.einsum('bi,b -> bi',sum([weights[k,l]*np.einsum('bi,b -> bi',T_map_Gaussian(X_P,mu_P_centered.comp[k],nu_centered.comp[l]),pdf_comp[k]) for k in range(mu.K) for l in range(nu.K)]),ipdf_mu)
    return Y + nu.mean()


def T_rand(x,mu,nu,P,weights):
    mu_P_centered = gmm_transform(mu,P=P.T,b=-P.T@mu.mean())
    nu_centered = gmm_transform(nu,b=-nu.mean())
    rng = np.random.default_rng()
    probs = np.array([weights[k,l]*mu.comp[k].pdf(x)/mu.pdf(x) for k in range(mu.K) for l in range(nu.K)])
    idx_list = [rng.choice([[k,l] for k in range(mu.K) for l in range(nu.K)],p=prob) for prob in probs.T]
    return T_map_Gaussian_rand(x,mu_P_centered,nu_centered,idx_list)
    




## Main algo



def MGW2_GM(mu,nu):
    ### core method between two Gaussian mixtures (returns a distance)
    return ot.gromov.gromov_wasserstein2(mu.pdist(GaussianW2),nu.pdist(GaussianW2),mu.weights,nu.weights,loss_fun='square_loss')



def aMGW2_GM(mu,nu,n_step=10,reg_init=1,beta=0.95,verbose=False):
    ### core method with annealed scheme 
    reg = reg_init
    weights = np.outer(mu.weights,nu.weights)
    for k in range(n_step):
        weights = ot.gromov.entropic_gromov_wasserstein(mu.pdist(GaussianW2),nu.pdist(GaussianW2),mu.weights,nu.weights,loss_fun='square_loss',epsilon=reg,G0=weights)
        reg *= beta
    return ot.gromov.gromov_wasserstein2(mu.pdist(GaussianW2),nu.pdist(GaussianW2),mu.weights,nu.weights,loss_fun='square_loss',G0=weights)
            



def MGW2(X,Y,n_components=20,annealing=False):
    ### MGW2 between two clouds of points (returns a distance)
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_,mix.means_,mix.covariances_)
    del mix
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_,mix.means_,mix.covariances_)
    del mix
    if annealing:
        return aMGW2_GM(mu,nu)
    else: 
        return MGW2_GM(mu,nu)

    

def MGW2_GM_coup(mu,nu):
    ### core method between two Gaussian Mixtures (return a coupling between Gaussian components)
    return ot.gromov.gromov_wasserstein(mu.pdist(GaussianW2),nu.pdist(GaussianW2),mu.weights,nu.weights,loss_fun='square_loss')



def aMGW2_GM_coup(mu,nu,n_step=10,reg_init=1,beta=0.95,verbose=False):
    ### core method with annealed scheme
    reg = reg_init
    weights = np.outer(mu.weights,nu.weights)
    for k in range(n_step):
        weights = ot.gromov.entropic_gromov_wasserstein(mu.pdist(GaussianW2),nu.pdist(GaussianW2),mu.weights,nu.weights,loss_fun='square_loss',epsilon=reg,G0=weights)
        reg *= beta
    return ot.gromov.gromov_wasserstein(mu.pdist(GaussianW2),nu.pdist(GaussianW2),mu.weights,nu.weights,loss_fun='square_loss',G0=weights)
            


def MGW2_coup(X,Y,n_components=20,annealing=False,method='T_rand',points=True,return_both=False,verbose=False):
    """ 
    MGW2 between two clouds of points (return a map between the points)
    If points = False, return the direct output of the T_mean or T_rand map (not in Y), 
    if points = True, return the idxs of the points in Y where the points in X are transported at. 
    """
    if verbose:
        print('fitting mixture 1')
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_,mix.means_,mix.covariances_)
    del mix
    if verbose: 
        print('fitting mixture 2')
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_,mix.means_,mix.covariances_)
    del mix
    if verbose: 
        print('deriving coupling between GMMs')
    if annealing:
        weights = aMGW2_GM_coup(mu,nu)
    else: 
        weights = MGW2_GM_coup(mu,nu)
    if verbose: 
        print('deriving map from coupling')
    P = proj_stiefel(sum([weights[k,l]*np.outer(mu.comp_mean[k],nu.comp_mean[l]) for k in range(mu.K) for l in range(nu.K)]))
    P, loss  = proj_gradient_descent(P,weights,mu,nu,1)
    if method=='T_mean':
        Z = T_mean(X,mu,nu,P,weights)
    elif method=='T_rand':
        Z = T_rand(X,mu,nu,P,weights)
    if return_both: 
        idx = nbrs.kneighbors(Z,return_distance=False).ravel()
        return idx, Z
    elif points:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Y)
        idx = nbrs.kneighbors(Z,return_distance=False).ravel()
        return idx
    else: 
        return Z


### MEW2 ###


def MEW2_GM(mu,nu,alpha,n_iter_P=150,eps=1e-3,n_iter_max=1000,init='Gaussian',initw=None,init_phase=True,symmetry=True,verbose=False):
    
    ### core method between two Gaussian mixtures mu and nu

    def twistedGaussianW2(mean_0,mean_1,Cov_0,Cov_0_full,Cov_1):
        sqCov_1 = spl.sqrtm(Cov_1).real
        return spl.norm(mean_0 - mean_1)**2 + np.trace(Cov_0_full) + np.trace(Cov_1) - 2*np.trace(spl.sqrtm(sqCov_1@Cov_0@sqCov_1).real)


    def compute_weights_loss(P,mu,nu,loss_only=False,w=None):
        mu_P = gmm_transform(mu,P=P.T)
        M = np.array([[twistedGaussianW2(mu_P.comp_mean[k],nu.comp_mean[l],mu_P.comp_cov[k],mu.comp_cov[k],nu.comp_cov[l]) for l in range(nu.K)] for k in range(mu.K)])
        del mu_P
        if not loss_only:
            weights = ot.emd(mu.weights,nu.weights,M) 
        else:
            weights = w
        loss = np.sum(weights*M)
        if loss_only:
            return loss
        else:
            return weights, loss


    def initialize(P,mu,nu,symmetry=False):
        """ enumerate all possible initializations matrices \tilde{\Id}_{m,n} 
        and choose the better (only possible in low dimensions)
        """
        P_list = P@[np.diag(a) for a in itertools.product([1,-1],repeat=P.shape[1])]
        weights_list, loss_list = [], []
        for k in range(len(P_list)):
            weights, loss = compute_weights_loss(P_list[k],mu,nu)
            weights_list.append(weights)
            loss_list.append(loss)
        idx = np.argmin(loss_list)
        P = P_list[idx]
        weights = weights_list[idx]
        loss = loss_list[idx]
        if symmetry: 
            P_list = [P[:,a] for a in itertools.permutations(range(P.shape[1]))]
            weights_list, loss_list = [], []
            for k in range(len(P_list)):
                weights, loss = compute_weights_loss(P_list[k],mu,nu)
                weights_list.append(weights)
                loss_list.append(loss)
            idx = np.argmin(loss_list)
            P = P_list[idx]
            weights = weights_list[idx]
            loss = loss_list[idx]
            
        return P, weights, loss


    if type(init) == str:
        s0, P0 = spl.eigh(mu.cov())
        s1, P1 = spl.eigh(nu.cov())
        P0 = P0[:,::-1]
        P1 = P1[:,::-1]    


    mu = gmm_transform(mu,b=-mu.mean())
    nu = gmm_transform(nu,b=-nu.mean())
    
    

    
    
    if type(init) == str:
        if init=='Gaussian':        
            P = P0@I(mu.dim,nu.dim)@P1.T
        elif init=='random':
            P = proj_stiefel(np.random.rand(mu.dim,nu.dim))
    else:
        P = init
    if initw is None:
        if init_phase:
            P, weights, loss = initialize(P,mu,nu,symmetry)
        else:
            weights, loss = compute_weights_loss(P,mu,nu)
    else:
        weights = initw 
        loss = compute_weights_loss(P,mu,nu,True,weights)
    loss_old = 0
    #print('initialization, loss = ' + str(loss))
    n_iter = 0
    while n_iter < n_iter_max and np.abs(loss - loss_old) > eps:
        if verbose:
            print('iteration ' + str(n_iter) + ': loss = ' + str(loss))
        loss_old = loss
        P = proj_gradient_descent(P,weights,mu,nu,alpha,n_iter = n_iter_P)
        weights, loss = compute_weights_loss(P,mu,nu)
        #print('iteration: ' + str(k+1) + ', loss = ' + str(loss))
        n_iter += 1

    return  P, weights, loss 



def EW2(X,Y,a=None,b=None,eps=1e-3,n_iter_max=10000,verbose=False):
    ### EW2 between two clouds of points (https://arxiv.org/pdf/1806.09277.pdf)
    m = X.shape[1]
    n = Y.shape[1]
    if a is None:
        a = ot.unif(X.shape[0])
    if b is None:
        b = ot.unif(Y.shape[0])
    P = proj_stiefel(np.random.rand(m,n))
    M = ot.dist(X,np.einsum('mn,bn->bm',P,Y))
    weights = ot.emd(a,b,M,numItermax=1000000)
    loss = np.trace(weights.T@M)
    loss_old = 0
    n_iter = 0
    while n_iter < n_iter_max and np.abs(loss-loss_old) > eps:
        if verbose:
            print('iteration ' + str(n_iter) + ': loss = ' + str(loss))
        loss_old = loss
        P = proj_stiefel(X.T@weights@Y)
        M = ot.dist(X,np.einsum('mn,bn->bm',P,Y))
        weights = ot.emd(a,b,M,numItermax=1000000)
        loss = np.trace(weights.T@M)
        n_iter += 1
    return P, weights, loss


def aEW2(X,Y,a=None,b=None,eps=1e-3,n_iter_max=10000,reg_init=1,beta=0.95,verbose=False):
    ### EW2 with annealed scheme (https://arxiv.org/pdf/1806.09277.pdf)
    m = X.shape[1]
    n = Y.shape[1]
    P = proj_stiefel(np.random.rand(m,n))
    M = ot.dist(X,np.einsum('mn,bn->bm',P,Y))
    if a is None:
        a = ot.unif(X.shape[0])
    if b is None:
        b = ot.unif(Y.shape[0])
    reg = reg_init
    weights = ot.bregman.sinkhorn_stabilized(a,b,M,reg)
    loss = np.trace(weights.T@M)
    loss_old = 0
    n_iter = 0
    while n_iter < n_iter_max and np.abs(loss-loss_old) > eps:
        if verbose:
            print('iteration ' + str(n_iter) + ': loss = ' + str(loss))
        loss_old = loss
        reg *= beta
        P = proj_stiefel(X.T@weights@Y)
        M = ot.dist(X,np.einsum('mn,bn->bm',P,Y))
        a = ot.unif(X.shape[0])
        b = ot.unif(Y.shape[0])
        weights = ot.bregman.sinkhorn_stabilized(a,b,M,reg)
        loss = np.trace(weights.T@M)
        n_iter += 1
    return P, weights, loss


def MEW2(X,Y,n_components=20,annealing=True,n_iter_annealing=10,beta=0.99):
    ### MEW2 between two clouds of points (returns a distance)
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_,mix.means_,mix.covariances_)
    del mix
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_,mix.means_,mix.covariances_)
    del mix
    if annealing:
        P, weights, loss =  aEW2(mu.comp_mean,nu.comp_mean,n_iter_max=n_iter_annealing,beta=0.99,verbose=False)
        P, weights, loss = MEW2_GM(mu,nu,alpha=0.01,eps=1e-3,n_iter_P=150,init=P,init_phase=False,symmetry=False,verbose=False)
    else:
        P, weights, loss = MEW2_GM(mu,nu,alpha=0.01,eps=1e-3,n_iter_P=150,init_phase=True,symmetry=True,verbose=False)
    return loss     


def MEW2_coup(X,Y,n_components=20,annealing=True,n_iter_annealing=10,beta=0.99,method='T_rand',points=True,return_both=False):
    """ 
    MEW2 between two clouds of points (return a map between the points)
    If points = False, return the direct output of the T_mean or T_rand map (not in Y), 
    if points = True, return the idxs of the points in Y where the points in X are transported at. 
    """
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_,mix.means_,mix.covariances_)
    del mix
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_,mix.means_,mix.covariances_)
    del mix
    if annealing:
        P, weights, loss =  aEW2(mu.comp_mean,nu.comp_mean,n_iter_max=n_iter_annealing,beta=0.99,verbose=False)
        P, weights, loss = MEW2_GM(mu,nu,alpha=0.01,eps=1e-3,n_iter_P=150,init=P,init_phase=False,symmetry=False,verbose=False)
    else:
        P, weights, loss = MEW2_GM(mu,nu,alpha=0.01,eps=1e-3,n_iter_P=150,init_phase=True,symmetry=True,verbose=False)
    if method=='T_mean':
        Z = T_mean(X,mu,nu,P,weights)
    elif method=='T_rand':
        Z = T_rand(X,mu,nu,P,weights)
    if return_both: 
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Y)
        idx = nbrs.kneighbors(Z,return_distance=False).ravel()
        return idx, Z
    elif points:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Y)
        idx = nbrs.kneighbors(Z,return_distance=False).ravel()
        return idx
    else: 
        return Z





    
