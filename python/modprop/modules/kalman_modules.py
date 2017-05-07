"""This module contains Module implementations for Kalman filtering.
"""

import numpy as np
import scipy.linalg as spl
from modprop.core.modules_core import ModuleBase, InputPort, OutputPort, iterative_invalidate
from modprop.core.backprop import sum_accumulators


def transpose_matrix(m, n):
    '''Generate the vectorized transpose matrix for column-major flattening.
    '''
    d = m * n
    inds = np.arange(start=0, stop=d)
    mat_trans = np.reshape(inds, (m, n), 'F').T
    T = np.zeros((d, d))
    T[inds, mat_trans.flatten('F')] = 1
    return T


def cho_solve_right(Lfactor, B):
    """Solves a right-inverse system B = XA for X = BA^-1.
    """
    return spl.cho_solve(Lfactor, B.T).T


class PredictModule(ModuleBase):
    """Performs a Kalman filter predict step.

    Input Ports
    -----------
    x_in : State estimate mean
    P_in : State estimate covariance
    Q_in : Transition covariance

    Output Ports
    ------------
    x_out : Post-prediction state estimate mean
    P_out : Post-prediction state estimate covariance

    Parameters
    ----------
    A : Transition matrix
    """

    def __init__(self, A):
        ModuleBase.__init__(self)
        self._A = A

        self._x_in = InputPort(self)
        self._P_in = InputPort(self)
        self._Q_in = InputPort(self)

        self._x_out = OutputPort(self)
        self._P_out = OutputPort(self)

        ModuleBase.register_inputs(self, self._x_in, self._P_in, self._Q_in)
        ModuleBase.register_outputs(self, self._x_out, self._P_out)

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        next_x = np.dot(self._A, self._x_in.value)
        next_P = np.dot(np.dot(self._A, self._P_in.value),
                        self._A.T) + self._Q_in.value
        ret = self._x_out.foreprop(next_x)
        ret += self._P_out.foreprop(next_P)

        return ret

    def backprop(self):
        if not self.backprop_ready():
            return []

        do_dxin = self._backprop_x_out()
        do_dPin, do_dQ = self._backprop_P_out()

        ret = self._x_in.backprop(do_dxin)
        ret += self._P_in.backprop(do_dPin)
        ret += self._Q_in.backprop(do_dQ)
        return ret

    def _backprop_x_out(self):
        dxout_dxin = self._A

        do_dxin = self._x_out.chain_backprop(dy_dx=dxout_dxin)
        return do_dxin

    def _backprop_P_out(self):
        '''Perform backpropagation on the P_out port.

        Each ith column of do_dPout is the gradient of element i of
        the final output w.r.t. the column unrolled P_out
        '''
        dPout_dPin = np.kron(self._A, self._A)
        do_dPin = self._P_out.chain_backprop(dy_dx=dPout_dPin)
        do_dQ = self._P_out.chain_backprop()
        return do_dPin, do_dQ

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, a):
        # TODO Make A an input?
        self._A = a
        iterative_invalidate(self)

    @property
    def x_in(self):
        return self._x_in

    @property
    def P_in(self):
        return self._P_in

    @property
    def Q_in(self):
        return self._Q_in

    @property
    def x_out(self):
        return self._x_out

    @property
    def P_out(self):
        return self._P_out


class UpdateModule(ModuleBase):
    """Performs a Kalman filter update step.

    # TODO Extend this logic to require correct pairings?
    This module does not require that all of its outputs be connected in order to backpropagate.

    Input Ports
    -----------
    x_in : Input filter estimate mean
    P_in : Input filter estimate covariance
    R_in : Input observation covariance

    Output Ports
    ------------
    x_out : Post-update estimate mean
    P_out : Post-update estimate covariance
    v_out : Update innovation
    S_out : Update innovation covariance

    Parameters
    ----------
    y : Observation vector
    C : Observation matrix
    """

    def __init__(self, y, C):
        ModuleBase.__init__(self)

        self._y = y
        self._C = C

        self._x_in = InputPort(self)
        self._P_in = InputPort(self)
        self._R_in = InputPort(self)

        self._x_out = OutputPort(self)
        self._P_out = OutputPort(self)
        self._v_out = OutputPort(self)
        self._S_out = OutputPort(self)

        ModuleBase.register_inputs(self, self._x_in, self._P_in, self._R_in)
        ModuleBase.register_outputs(
            self, self._x_out, self._P_out, self._v_out, self._S_out)

        # Cached variables
        self._S_chol = None
        self._K = None

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        P_in = self._P_in.value
        ypred = np.dot(self._C, self._x_in.value)
        v = self._y - ypred
        S = np.dot(np.dot(self._C, P_in), self._C.T) + self._R_in.value
        #self._S_inv = np.linalg.inv(S)
        self._S_chol = spl.cho_factor(S)
        #self._K = np.dot(np.dot(P_in, self._C.T), self._S_inv)
        self._K = cho_solve_right(self._S_chol, np.dot(P_in, self._C.T))

        x_next = self._x_in.value + np.dot(self._K, v)
        P_next = P_in - np.dot(np.dot(self._K, self._C), P_in)

        ret = self._x_out.foreprop(x_next)
        ret += self._P_out.foreprop(P_next)
        ret += self._v_out.foreprop(v)
        ret += self._S_out.foreprop(S)
        return ret

    def backprop(self):
        if not self.backprop_ready():
            return []

        do_dxin_x, do_dPin_x, do_dR_x = self._backprop_x_out()
        do_dPin_P, do_dRin_P = self._backprop_P_out()
        do_dxin_v = self._backprop_v_out()
        do_dPin_S, do_dRin_S = self._backprop_S_out()

        ret = self._x_in.backprop(sum_accumulators((do_dxin_x, do_dxin_v)))
        ret += self._P_in.backprop(sum_accumulators((do_dPin_x,
                                                     do_dPin_P, do_dPin_S)))
        ret += self._R_in.backprop(sum_accumulators((do_dR_x,
                                                     do_dRin_P, do_dRin_S)))
        return ret

    def _backprop_x_out(self):
        N = len(self._x_in.value)
        dxout_dxin = np.identity(N) - np.dot(self._K, self._C)
        do_dxin = self._x_out.chain_backprop(dy_dx=dxout_dxin)

        Sv = spl.cho_solve(self._S_chol, self._v_out.value)
        # Sv = np.dot(self._S_inv, self._v_out.value)
        CTSv = np.dot(self._C.T, Sv)
        KC = np.dot(self._K, self._C)
        dxout_dPin = np.kron(CTSv.T, np.identity(N)) - np.kron(CTSv.T, KC)
        do_dPin = self._x_out.chain_backprop(dy_dx=dxout_dPin)

        dxout_dR = -np.kron(Sv.T, self._K)
        do_dR = self._x_out.chain_backprop(dy_dx=dxout_dR)

        return do_dxin, do_dPin, do_dR

    def _backprop_P_out(self):
        N = self._P_in.value.shape[0]
        KC = np.dot(self._K, self._C)

        I = np.identity(N)
        II = np.identity(N * N)
        T = transpose_matrix(N, N)
        dPout_dPin = II - np.dot(II + T, np.kron(I, KC)) + np.kron(KC, KC)

        do_dPin = self._P_out.chain_backprop(dy_dx=dPout_dPin)

        dPout_dRin = np.kron(self._K, self._K)
        do_dRin = self._P_out.chain_backprop(dy_dx=dPout_dRin)

        return do_dPin, do_dRin

    def _backprop_v_out(self):
        dvout_dxin = -self._C
        do_dxin = self._v_out.chain_backprop(dy_dx=dvout_dxin)

        return do_dxin

    def _backprop_S_out(self):
        dSout_dPin = np.kron(self._C, self._C)
        do_dPin = self._S_out.chain_backprop(dy_dx=dSout_dPin)

        do_dRin = self._S_out.chain_backprop()

        return do_dPin, do_dRin

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = v
        iterative_invalidate(self)

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, v):
        self._C = v
        iterative_invalidate(self)

    @property
    def x_in(self):
        return self._x_in

    @property
    def P_in(self):
        return self._P_in

    @property
    def R_in(self):
        return self._R_in

    @property
    def x_out(self):
        return self._x_out

    @property
    def P_out(self):
        return self._P_out

    @property
    def v_out(self):
        return self._v_out

    @property
    def S_out(self):
        return self._S_out
