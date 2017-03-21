"""This module contains Module implementations to compute cost metrics.
"""
import math
import numpy as np
import scipy.linalg as spl
from modprop.core.modules_core import ModuleBase, InputPort, OutputPort

def cho_determinant(factor):
    """Computes the determinant of a matrix from its Cholesky factor.

    Produces a more stable result than np.linalg.det for ill-conditioned matrices,
    assuming that they have a Cholesky decomposition.

    Parameters
    ----------
    factor : Tuple of (matrix, bool) produced by scipy.linalg.cho_factor
    """
    D = np.diag(factor[0])
    d = np.prod(D)
    return d * d

def cho_logdet(factor):
    """Computes the log-determinant of a matrix from its Cholesky factor.

    Produces a more stable result than np.linalg.det for ill-conditioned matrices,
    assuming that they have a Cholesky decomposition.
    
    Parameters
    ----------
    factor : Tuple of (matrix, bool) produced by scipy.linalg.cho_factor
    """
    d = np.diag(factor[0])
    return 2 * np.sum(np.log(d))

class LogLikelihoodModule(ModuleBase):
    """Computes the log-likelihood of a multivariate Gaussian.

    Input Ports
    -----------
    x_in : Sample with mean removed
    S_in : Gaussian covariance matrix

    Output Ports
    ------------
    ll_out : Log-likelihood output
    """
    def __init__(self):
        ModuleBase.__init__(self)
        self._x_in = InputPort(self)
        self._S_in = InputPort(self)
        self._ll_out = OutputPort(self)

        ModuleBase.register_inputs(self, self._x_in, self._S_in)
        ModuleBase.register_outputs(self, self._ll_out)

        self._cho_L = None
        self._S_inv = None
        self._x_inv = None

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        k = len(self._x_in.value)
        self._cho_L = spl.cho_factor(self._S_in.value)
        self._S_inv = spl.cho_solve(self._cho_L, np.identity(k))
        logdet = cho_logdet(self._cho_L)

        reg_term = -0.5 * (k*math.log(2*math.pi) + logdet)

        self._x_inv = spl.cho_solve(self._cho_L, self._x_in.value)
        exp_term = -0.5 * np.dot(self._x_in.value, self._x_inv)

        return self._ll_out.foreprop(reg_term + exp_term)

    def backprop(self):
        if not self._ll_out.backprop_ready:
            return

        # do_dll = self._ll_out.backprop_value

        dll_dxin = -np.atleast_2d(self._x_inv)
        # do_dx = np.dot(do_dll, dll_dx)
        do_dxin = self._ll_out.chain_backprop(dy_dx=dll_dxin)

        dll_dSin = -0.5 * self._S_inv.flatten('F') + 0.5 * np.kron(self._x_inv, self._x_inv)
        dll_dSin = np.atleast_2d(dll_dSin)
        # do_dS = np.dot(do_dll, dll_dS)
        do_dSin = self._ll_out.chain_backprop(dy_dx=dll_dSin)


        ret = self._x_in.backprop(do_dxin)
        ret += self._S_in.backprop(do_dSin)
        return ret

    @property
    def x_in(self):
        return self._x_in

    @property
    def S_in(self):
        return self._S_in

    @property
    def ll_out(self):
        return self._ll_out