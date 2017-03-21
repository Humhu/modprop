"""This module contains Module implementations that reshape their inputs.
"""
import numpy as np
from modprop.core.modules_core import ModuleBase, InputPort, OutputPort

class DiagonalReshapeModule(ModuleBase):
    """Remaps a vector input into a diagonal matrix.

    Input Ports
    -----------
    vec_in : Vector input of length N

    Output Ports
    ------------
    diag_out : Diagonal output of size (N, N)
    """
    def __init__(self):
        ModuleBase.__init__(self)

        self._d_in = InputPort(self)
        self._D_out = OutputPort(self)

        ModuleBase.register_inputs(self, self._d_in)
        ModuleBase.register_outputs(self, self._D_out)

        self._D_inds = None
        self._D = None

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        N = len(self._d_in.value)
        if self._D_inds is None:
            D_i_inds, D_j_inds = np.diag_indices(N)
            self._D_inds = np.ravel_multi_index((D_i_inds, D_j_inds), (N, N))

        self._D = np.zeros((N, N))
        self._D.flat[self._D_inds] = self._d_in.value
        return self._D_out.foreprop(self._D)

    def backprop(self):
        if not self.backprop_ready():
            return []

        # do_dDout = self._D_out.backprop_value

        N = self._D_out.value.shape[0]
        dD_dd = np.zeros((N*N, N))
        dD_dd[self._D_inds, :] = np.identity(N)

        do_dd = self._D_out.chain_backprop(dy_dx=dD_dd)
        # do_dd = np.dot(do_dDout, dD_dd)

        return self._d_in.backprop(do_dd)

    @property
    def vec_in(self):
        return self._d_in

    @property
    def diag_out(self):
        return self._D_out


class CholeskyReshapeModule(ModuleBase):
    """Remaps two vector inputs into a Cholesky factor (lower triangular matrix) and recombines them into a matrix.

    Input Ports
    -----------
    d_in : Vector input of length N that maps to factor diagonal
    l_in : Vector input of length (N*(N-1))/2 that column-major maps to factor lower-triangular

    Output Ports
    ------------
    S_out : Matrix output of size (N, N) that equals factor * factor.T
    """
    def __init__(self):
        ModuleBase.__init__(self)

        self._d_in = InputPort(self)
        self._l_in = InputPort(self)
        self._S_out = OutputPort(self)

        ModuleBase.register_inputs(self, self._d_in, self._l_in)
        ModuleBase.register_outputs(self, self._S_out)

        self._D_inds = None
        self._L_inds = None
        self._L_T_inds = None
        self._L = None

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        N = len(self._d_in.value)
        if self._D_inds is None or self._L_inds is None:
            D_i_inds, D_j_inds = np.diag_indices(N)
            self._D_inds = np.ravel_multi_index((D_i_inds, D_j_inds), (N, N))

            L_i_inds, L_j_inds = np.tril_indices(N, k=-1)
            self._L_inds = np.ravel_multi_index((L_i_inds, L_j_inds), (N, N))
            self._L_T_inds = np.ravel_multi_index((L_i_inds, L_j_inds), (N, N), order='F')

        self._L = np.zeros((N, N))
        self._L.flat[self._D_inds] = self._d_in.value
        self._L.flat[self._L_inds] = self._l_in.value

        return self._S_out.foreprop(np.dot(self._L, self._L.T))

    def backprop(self):
        if not self._S_out.backprop_ready:
            return []

        # do_dSout = self._S_out.backprop_value

        #dS_dL = self._L.flatten()
        N = self._S_out.value.shape[0]
        dS_dL = np.kron(np.identity(N), self._L)
        dST_dL = np.kron(self._L, np.identity(N))

        dS_dl = dS_dL[:, self._L_inds] + dST_dL[:, self._L_T_inds]
        do_dl = self._S_out.chain_backprop(dy_dx=dS_dl)
        # do_dl = np.dot(do_dSout, dS_dl)

        dS_dd = dS_dL[:, self._D_inds] + dST_dL[:, self._D_inds]
        do_dd = self._S_out.chain_backprop(dy_dx=dS_dd)
        # do_dd = np.dot(do_dSout, dS_dd)

        ret = self._l_in.backprop(do_dl)
        ret += self._d_in.backprop(do_dd)
        return ret

    @property
    def d_in(self):
        return self._d_in

    @property
    def l_in(self):
        return self._l_in

    @property
    def S_out(self):
        return self._S_out
