# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree as KDTree

__all__ = ["LocalLinearApproximation", "LocalQuadraticApproximation"]


class LocalLinearApproximation(object):

    def __init__(self, theta, f_theta, nfactor=None):
        self.theta = np.atleast_2d(theta)
        self.f_theta = np.atleast_2d(f_theta)
        self.nsamples, self.ndim = self.theta.shape
        _, self.nout = self.f_theta.shape
        if self.f_theta.shape[0] != self.nsamples:
            raise ValueError("dimension mismatch; there must be a realization "
                             "for every sample")
        self.tree = KDTree(self.theta)
        self.ndef = self.get_ndef(self.ndim)
        if nfactor is None:
            nfactor = np.sqrt(self.ndim)
        self.ntot = max(int(nfactor * self.ndef), self.ndef + 2)

    def evaluate(self, theta, cross_validate=False):
        theta = np.atleast_1d(theta)
        if theta.shape != (self.ndim, ):
            raise ValueError("dimension mismatch; theta must have shape {0}"
                             .format((self.ndim, )))

        dists, inds = self.tree.query(theta, self.ntot)
        if len(inds) != self.ntot:
            raise ValueError("could not construct list of {0} neighbors"
                             .format(self.ntot))

        # Compute the weights.
        rout = dists[-1]
        rdef = dists[self.ndef-1]
        weights = (1.0 - ((dists-rdef)/(rout-rdef))**3)**3
        weights *= (dists <= rout) * (dists >= rdef)
        weights += 1.0 * (dists < rdef)

        # Construct the regression matrices.
        A = self.get_design_matrix((self.theta[inds] - theta) / rout)
        AT = A.T
        Aw = A * weights[:, None]
        yw = self.f_theta[inds] * weights[:, None]
        Apred = self.get_design_matrix([theta])

        # Evaluate the model at the requested point.
        ATA = np.dot(AT, Aw)
        ATy = np.dot(AT, yw)
        w = np.linalg.solve(ATA, ATy)
        pred = np.dot(Apred, w)[0]
        if not cross_validate:
            return pred

        # Leave-one-out cross validation scheme.
        preds = []
        rng = np.arange(self.ntot)
        for i in rng:
            m = rng != i
            ATA = np.dot(AT[:, m], Aw[m])
            ATA = np.dot(AT[:, m], Aw[m])
            ATy = np.dot(AT[:, m], yw[m])
            w = np.linalg.solve(ATA, ATy)
            preds.append(np.dot(Apred, w))
        return pred, np.concatenate(preds, axis=0)

    def find_refinement_coords(self, theta):
        theta = np.atleast_1d(theta)
        if theta.shape != (self.ndim, ):
            raise ValueError("dimension mismatch; theta must have shape {0}"
                             .format((self.ndim, )))
        dists, _ = self.tree.query(theta, self.ntot)
        R = dists[-1]
        inds = self.tree.query_ball_point(theta, 3*R)
        thetas = self.theta[inds]

        def cost(t):
            if np.sum((t - theta)**2) > R**2:
                return 1e12
            return np.min(np.sum((t[None, :] - thetas)**2, axis=1))

        def grad_cost(t):
            r = t[None, :] - thetas
            i = np.argmin(np.sum(r**2, axis=1))
            return 2 * r[i]

        v = minimize(cost, theta, method="L-BFGS-B", jac=grad_cost,
                     bounds=[(t-R, t+R) for t in theta])
        return v.x

    def get_ndef(self, ndim):
        return ndim + 1

    def get_design_matrix(self, x):
        return np.concatenate((x, np.ones((len(x), 1))), axis=1)


class LocalQuadraticApproximation(LocalLinearApproximation):

    def get_ndef(self, ndim):
        return ((ndim + 1) * (ndim + 2)) // 2

    def get_design_matrix(self, x):
        x = np.atleast_2d(x)
        cols = [np.ones(len(x))]
        for i in range(self.ndim):
            cols.append(x[:, i])
            for j in range(i, self.ndim):
                cols.append(x[:, i] * x[:, j])
        return np.vstack(cols).T


if __name__ == "__main__":
    x = np.random.rand(100, 2)
    y = np.vstack((np.sin(np.sum(x, 1)), np.cos(x[:, 0]) + np.sin(x[:, 1]))).T
    approx = LocalQuadraticApproximation(x, y)
    # approx = LocalLinearApproximation(x, y)
    print(approx.evaluate([0.5, 0.5], cross_validate=True))
    print(approx.find_refinement_coords([0.5, 0.5]))
