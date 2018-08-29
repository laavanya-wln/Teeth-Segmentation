import math
import numpy as np
from bisect import bisect_left

__author__ = "Laavanya Wudali Lakshmi Narsu"

"""
PCA and PCA based models
"""


class PCAModel:
    """ A Principal Component Analysis Model to perform
        dimensionality reduction. Implementation is based on
        Appendix B and C of

        Cootes, Tim, E. R. Baldock, and J. Graham.
        "An introduction to active shape models."
        Image processing and analysis (2000): 223-248.

        Attributes:
        _x  The centered data matrix (Nxd)
        _mean The mean of the input data (1xd)
        _lambdas The eigenvalues of the PCA Matrix (1xd)
        _eigenvectors The eigenvectors of the PCA Matrix (Nxd)

        Authors: Laavanya Wudali Lakshmi Narsu
    """

    def _matrix(self):
        """
        Returns the matrix to use to perform
        PCA depending on the shape of the input
        matrix
        :return: The Matrix to be eigendecomposed
        """
        n, d = self._x.shape
        if n < d:
            return np.dot(self._x, self._x.T) / float(n - 1)
        return np.dot(self._x.T, self._x) / float(n - 1)

    def _eigen(self):
        """
        Perform the eigendecomposition and persist ALL the
        eigenvalues and eigenvectors
        """
        n, d = self._x.shape
        if self._useSVD:
            U,S,Vt = np.linalg.svd(self._x)
            l = (S**2)/float(n-1);
            self._w = Vt.T
        else:            
            c = self._matrix()
            nc, _ = c.shape
            l, w = np.linalg.eigh(c)
            if nc == n:
                w = np.dot(self._x.T, w)    
            # Normalize the eigenvectors
            self._w = w / np.linalg.norm(w, axis=0)    
            # Sort the eigenvectors according to the largest
            # eigenvalues
        indices = np.argsort(l)[::-1][:d]
        self._lambdas = l[indices]
        self._w = self._w[:, indices]

    def __init__(self, x,use_SVD=True):
        """
        Constructs and fits the PCA Model with the given data matrix x
        :param x: The input data matrix
        """
        # Compute the mean and center the matrix
        self._mean = np.mean(x, axis=0)
        self._x = x - self._mean
        self._useSVD = use_SVD
        # Perform the fit
        self._eigen()

    def get_mean(self):
        """
        Returns the mean of the input matrix
        :return: The vector containing the mean
        """
        return self._mean

    def get_eigenvectors(self):
        """
        Returns the eigenvectors of the covariance matrix.
        They are scaled to have norm 1
        :return: The matrix containing the eigenvalues
        """
        return self._w

    def get_eigenvalues(self):
        """
        Returns the eigenvalues of the covariance matrix
        :return:  The vector of the eigenvalues
        """
        return self._lambdas
    
    def get_varfrac(self):
        """
        Returns the fraction of variance captured by each PC
        :return:  The vector of the variance captured by each component
        """
        return self._lambdas / np.sum(self._lambdas)

    def get_k_cutoff(self, variance_captured=0.9):
        """
        Returns the number of principal components that have to be used
        in order for the model to capture the given fraction
        of total variance
        :param variance_captured: The fraction of the total variance

        :return: The number of components
        """
        return bisect_left(np.cumsum(self.get_varfrac()), variance_captured)

    def project(self, x=None, k=0):
        """
        Use the fitted model to project a set of points
        :param x: The data that has to be projected.
         Defaults to the data used to fit the model.
        :param k: The number of principal components to be used.
        Defaults to all the principal components.
        :return: The matrix of projections of the input data
        """
        _, d = self._x.shape
        if k < 1 or k > d:
            k = d
        if x is not None:
            return np.dot(x - np.mean(x, axis=0), self._w[0:k])
        return np.dot(self._x, self._w[0:k])

    def reconstruct(self, y=None, k=0):
        """
        Reconstructs the input projections by mapping back to the
        original space.
        :param y: The projections produced by the model.
        Defaults to the projections of the data used to fit
        the model
        :param k: The number of components to be used. This
        must be compatible with the input matrix y. Defaults to all
        the principal components
        :return: The matrix of reconstructed points
        """
        _, d = self._x.shape
        if k < 1 or k > d:
            k = d
        if y is None:
            y = self.project(k)
        return np.dot(self._w[0:k], y.T) + self._mean