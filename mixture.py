import numpy as np
import pandas as pd
import tqdm

import scipy.stats as stats
import sklearn.cluster as cluster

import cvxpy as cp


class MultiGauss:
    
    def init_responsibilities(X, weights,n_components):
        # initialize weights with KMeans
        n_samples, _ = X.shape
        labels = cluster.KMeans(n_clusters=n_components, n_init=10).fit_predict(
            X, sample_weight=weights
        )
        resp = np.zeros((n_samples, n_components))
        resp[np.arange(n_samples), labels] = 1
        return resp
    
    def fit(self, X, w):
        self.mean = np.average(X, weights=w, axis=0)
        self.cov = np.nan_to_num(np.cov(X.T, aweights=w), 0)
        np.fill_diagonal(self.cov, self.cov.diagonal() + 1e-6)

        self._n_parameters = len(self.mean) + len(self.cov.flatten())
        return self

    def pdf(self, X):
        return np.nan_to_num(
            stats.multivariate_normal.pdf(
                X, mean=self.mean, cov=self.cov
            )
        )

class Gauss: 
   
    def fit(self, X, w):
        self.mean = np.average(X, weights=w, axis=0)
        self.std = np.sqrt(np.cov(X, aweights=w))
        self.std += 1e-6

        self._n_parameters = 2
        return self

    def pdf(self, X):
        return np.nan_to_num(
            stats.norm.pdf(
                X, loc=self.mean, scale=self.std
            )
        )

class VonMises:
    
    def init_responsibilities(alpha, weights,n_components):
        # initialize weights with KMeans
        n_samples, _ = alpha.shape
        X = np.concatenate([np.sin(alpha),np.cos(alpha)],axis=1)
        labels = cluster.KMeans(n_clusters=n_components, n_init=10).fit_predict(
            X, sample_weight=weights
        )
        resp = np.zeros((n_samples, n_components))
        resp[np.arange(n_samples), labels] = 1
        return resp
    
    def fit(self, alpha, w):
        sin = np.average(np.sin(alpha), weights=w, axis=0)
        cos = np.average(np.cos(alpha), weights=w, axis=0)

        self.loc = np.arctan2(sin, cos)
        self.R = np.sqrt(sin ** 2 + cos ** 2)  # mean resultant length
        # self.R = min(self.R,0.99)

        maxR = np.empty_like(self.R)
        maxR[0] = 0.99
        self.R = min(self.R, maxR)

        self.kappa = (
            self.R * (2 - self.R ** 2) / (1 - self.R ** 2)
        )  # approximation for kappa

        self._n_parameters = 2
        return self

    def pdf(self, alpha):
        return np.nan_to_num(
            stats.vonmises.pdf(alpha, kappa=self.kappa, loc=self.loc).flatten()
        )


class CategoricalModel:
    
    def __init__(self, tol=1e-6):
        self.tol = tol

    def fit(self, X, weights=None):
        if weights:
            X = X[weights > self.tol]
        self.categories = set(X)
        return self

    def predict_proba(self, X, weights=None):
        p = pd.DataFrame()
        if weights is None:
            weights = np.zeros(len(X)) + 1
        for c in self.categories:
            p[str(c)] = ((X == c) & (weights > self.tol)).apply(float)
        return p


class MixtureModel:
    def __init__(self, n_components, distribution=MultiGauss, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.distribution = distribution
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, weights=None, verbose=False):

        # handle sparsity
        if weights is None:
            weights = np.zeros(len(X)) + 1
        pos_weights_idx = weights > self.tol
        X = X[pos_weights_idx]
        weights = weights[pos_weights_idx]

        self.weight_total = weights.sum()
        self.loglikelihood = -np.inf
        self.submodels = list(self.distribution() for _i in range(self.n_components))

        if len(X) < self.n_components:
            return None

        responsibilities = self.distribution.init_responsibilities(X, weights,self.n_components)

        # learn models on initial weights
        self.priors = responsibilities.sum(axis=0) / responsibilities.sum()
        # invalid model if less clusters found than given components
        if any(self.priors < self.tol):
            return None
            
        for i in range(self.n_components):
            self.submodels[i].fit(X, weights * responsibilities[:, i])

        iterations = (
            range(self.max_iter) if not verbose else tqdm.tqdm(range(self.max_iter))
        )

        for self._n_iter in iterations:
            # Expectation
            for i in range(self.n_components):
                responsibilities[:, i] = self.priors[i] * self.submodels[i].pdf(X)

            # enough improvement or not?
            new_loglikelihood = (weights * np.log(responsibilities.sum(axis=1))).sum()

            if new_loglikelihood > self.loglikelihood + self.tol:
                self.loglikelihood = new_loglikelihood
                # self.responsibilities = responsibilities
                # self.weights = weights
            else:
                break

            # normalize responsibilities such that each data point occurs with P=1
            responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]

            # Maximalization
            self.priors = responsibilities.sum(axis=0) / responsibilities.sum()
            for i in range(self.n_components):
                self.submodels[i].fit(X, weights * responsibilities[:, i])

        if np.isinf(self.loglikelihood):
            return None

        return self

    def predict_proba(self, X, weights=None):
        p = np.zeros((len(X), self.n_components))

        # handle sparsity
        if weights is None:
            weights = np.zeros(len(X)) + 1
        pos_weights_idx = weights > self.tol
        X = X[pos_weights_idx]
        weights = weights[pos_weights_idx]

        pdfs = np.vstack([m.pdf(X) for m in self.submodels]).T
        resp = self.priors * pdfs
        probs = resp / resp.sum(axis=1)[:, np.newaxis]

        p[pos_weights_idx, :] = (weights * probs.T).T
        return p

    def params(self):
        return list(m.__dict__ for m in self.submodels)

    def _n_parameters(self):
        return (
            sum(m._n_parameters for m in self.submodels)
            - self.submodels[0]._n_parameters
        )


def ilp_select_models_max(models, max_components, verbose=False):
    x = cp.Variable(len(models), boolean=True)
    c = np.array(list(m.loglikelihood for m in models))
    n_components = np.array(list(m.n_components for m in models))

    objective = cp.Maximize(cp.sum(c * x))
    constraints = []
    constraints += [n_components * x <= max_components]
    for name in set(m.name for m in models):
        name_idx = np.array(list(int(m.name == name) for m in models))
        constraints += [name_idx * x == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    idx, = np.where(x.value > 0.3)
    return list(models[i] for i in idx)


def ilp_select_models_bic_triangle(models, verbose=False):
    x = cp.Variable(len(models), boolean=True)
    c = np.array(list(m.loglikelihood for m in models))
    n_parameters = np.array(list(m.n_components for m in models))
    dataweights = {}
    for m in models:
        if m.name not in dataweights:
            dataweights[m.name] = m.weight_total
    n_data = sum(dataweights.values())

    n = cp.sum(n_parameters * x)
    para = n + (cp.square(n) + n) / 2
    objective = cp.Minimize(np.log(n_data) * para - 2 * cp.sum(c * x))

    constraints = []
    for name in set(m.name for m in models):
        name_idx = np.array(list(int(m.name == name) for m in models))
        constraints += [name_idx * x == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    idx, = np.where(x.value > 0.3)
    return list(models[i] for i in idx)


def ilp_select_models_bic(models, verbose=False):
    x = cp.Variable(len(models), boolean=True)
    c = np.array(list(m.loglikelihood for m in models))
    n_parameters = np.array(list(m._n_parameters() for m in models))
    dataweights = {}
    for m in models:
        if m.name not in dataweights:
            dataweights[m.name] = m.weight_total
    n_data = sum(dataweights.values())

    objective = cp.Minimize(
        np.log(n_data) * cp.sum(n_parameters * x) - 2 * cp.sum(c * x)
    )

    constraints = []
    for name in set(m.name for m in models):
        name_idx = np.array(list(int(m.name == name) for m in models))
        constraints += [name_idx * x == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    idx, = np.where(x.value > 0.3)
    return list(models[i] for i in idx)


def select_models_solo_bic(models):
    for m in models:
        m.solo_bic = np.log(m.weight_total) * m._n_parameters() - 2 * m.loglikelihood

    ms = []
    for name in set(m.name for m in models):
        bestm = min([m for m in models if m.name == name], key=lambda m: m.solo_bic)
        ms.append(bestm)
    return ms


def probabilities(models, X, W):
    weights = []
    for model in models:
        probs = model.predict_proba(X, W[model.name].values)
        nextlevel_columns = list(f"{model.name}_{i}" for i in range(model.n_components))
        weights.append(pd.DataFrame(probs, columns=nextlevel_columns))
    return pd.concat(weights, axis=1)

