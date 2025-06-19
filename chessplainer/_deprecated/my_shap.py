import shap
from shap.utils._legacy import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data
from shap.utils._legacy import convert_to_instance_with_index, convert_to_link, IdentityLink, convert_to_data, \
    DenseData, SparseData
from shap.utils import safe_isinstance
from scipy.special import binom
from scipy.sparse import issparse
import numpy as np
import pandas as pd
import scipy as sp
import logging
import copy
import itertools
import warnings
import gc
from sklearn.linear_model import LassoLarsIC, Lasso, lars_path
from tqdm.auto import tqdm
from shap.explainers._explainer import Explainer

log = logging.getLogger('shap')


class MyKernelExplainer(shap.KernelExplainer):

    def __init__(self, model, data, link=IdentityLink(), **kwargs):

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.model = convert_to_model(model)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        self.data = convert_to_data(data, keep_index=self.keep_index)
        model_null = np.array([0])  # self.model.f(np.ones_like(self.data.data))

        # enforce our current input type limitations
        assert isinstance(self.data, DenseData) or isinstance(self.data, SparseData), \
            "Shap explainer only supports the DenseData and SparseData input currently."
        assert not self.data.transposed, "Shap explainer does not support transposed DenseData or SparseData currently."

        # warn users about large background data sets
        if len(self.data.weights) > 100:
            log.warning("Using " + str(len(self.data.weights)) + " background data samples could cause " +
                        "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to " +
                        "summarize the background as K samples.")

        # init our parameters
        self.N = self.data.data.shape[0]
        self.P = self.data.data.shape[1]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = np.squeeze(model_null.values)
        if safe_isinstance(model_null, "tensorflow.python.framework.ops.EagerTensor"):
            model_null = model_null.numpy()
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]

    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :]
        if self.keep_index:
            index = self.synth_data_index[self.nsamplesRun * self.N:self.nsamplesAdded * self.N]
            index = pd.DataFrame(index, columns=[self.data.index_name])
            data = pd.DataFrame(data, columns=self.data.group_names)
            data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
            if self.keep_index_ordered:
                data = data.sort_index()
        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        not_nan_idxs = ~np.isnan(modelOut)
        self.y = self.y[not_nan_idxs]
        self.maskMatrix = self.maskMatrix[not_nan_idxs]
        self.lastMask = self.lastMask[not_nan_idxs]
        self.fnull = np.array([0])
        self.kernelWeights = self.kernelWeights[not_nan_idxs]
        modelOut = modelOut[not_nan_idxs]
        num_to_run = len(modelOut)
        self.nsamplesAdded = num_to_run
        self.nsamples = num_to_run
        self.max_samples = num_to_run
        self.y[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))
        self.synth_data_valid = data[not_nan_idxs]

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1
        self.ey = self.ey[:num_to_run]


    def solve(self, fraction_evaluated, dim):
        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])
        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        log.debug("fraction_evaluated = {0}".format(fraction_evaluated))
        # if self.l1_reg == "auto":
        #     warnings.warn(
        #         "l1_reg=\"auto\" is deprecated and in the next version (v0.29) the behavior will change from a " \
        #         "conditional use of AIC to simply \"num_features(10)\"!"
        #     )
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
            log.info("np.sum(w_aug) = {0}".format(np.sum(w_aug)))
            log.info("np.sum(self.kernelWeights) = {0}".format(np.sum(self.kernelWeights)))
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix, self.maskMatrix - 1))))
            #var_norms = np.array([np.linalg.norm(mask_aug[:, i]) for i in range(mask_aug.shape[1])])

            # select a fixed number of top features
            if isinstance(self.l1_reg, str) and self.l1_reg.startswith("num_features("):
                r = int(self.l1_reg[len("num_features("):-1])
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1]

            # use an adaptive regularization method
            elif self.l1_reg == "auto" or self.l1_reg == "bic" or self.l1_reg == "aic":
                c = "aic" if self.l1_reg == "auto" else self.l1_reg
                nonzero_inds = np.nonzero(LassoLarsIC(criterion=c).fit(mask_aug, eyAdj_aug).coef_)[0]

            # use a fixed regularization coeffcient
            else:
                nonzero_inds = np.nonzero(Lasso(alpha=self.l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
                    self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))
        etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) - self.maskMatrix[:, nonzero_inds[-1]])
        log.debug("etmp[:4,:] {0}".format(etmp[:4, :]))

        # solve a weighted least squares equation to estimate phi
        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
        etmp_dot = np.dot(np.transpose(tmp), etmp)
        try:
            tmp2 = np.linalg.inv(etmp_dot)
        except np.linalg.LinAlgError:
            tmp2 = np.linalg.pinv(etmp_dot)
            warnings.warn(
                "Linear regression equation is singular, Moore-Penrose pseudoinverse is used instead of the regular inverse.\n"
                "To use regular inverse do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
        w = np.dot(tmp2, np.dot(np.transpose(tmp), eyAdj2))
        log.debug("np.sum(w) = {0}".format(np.sum(w)))
        log.debug("self.link(self.fx) - self.link(self.fnull) = {0}".format(
            self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])))
        log.debug("self.fx = {0}".format(self.fx[dim]))
        log.debug("self.link(self.fx) = {0}".format(self.link.f(self.fx[dim])))
        log.debug("self.fnull = {0}".format(self.fnull[dim]))
        log.debug("self.link(self.fnull) = {0}".format(self.link.f(self.fnull[dim])))
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)
        log.info("phi = {0}".format(phi))

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))
