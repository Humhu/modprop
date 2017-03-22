import numpy as np
import scipy.linalg as spl
from itertools import izip

from modprop import *
import utils
import pdb

import matplotlib.pyplot as plt

class ConstantPSD(object):
    def __init__(self, init_value):
        L = spl.cho_factor(init_value)[0]
        d = np.diag(L)
        log_d = np.log(d)

        li, lj = np.tril_indices(len(d), k=-1)
        l_inds = np.ravel_multi_index((li, lj), (len(d), len(d)))
        l = L.flat[l_inds] # Note that CholeskyReshape uses default flattening

        self._d_source = ConstantModule(log_d)
        self._d_exp = ExponentialModule()
        link_ports(in_port=self._d_exp.in_port, out_port=self._d_source.out_port)

        self._l_source = ConstantModule(l)
        self._psd_source = CholeskyReshapeModule()
        link_ports(in_port=self._psd_source.l_in, out_port=self._l_source.out_port)
        link_ports(in_port=self._psd_source.d_in, out_port=self._d_exp.out_port)

    def foreprop(self):
        ret = self._d_source.foreprop()
        ret += self._l_source.foreprop()
        return ret

    def invalidate(self):
        ret = self._d_source.invalidate()
        ret += self._l_source.invalidate()
        return ret

    @property
    def out_port(self):
        return self._psd_source.S_out

    @property
    def theta(self):
        return np.hstack((self._d_source.value,
                          self._l_source.value))

    @theta.setter
    def theta(self, th):
        n_d = len(self._d_source.value)
        n_l = len(self._l_source.value)
        self._d_source.value = th[:n_d]
        self._l_source.value = th[n_d:]

    @property
    def backprop_value(self):
        return np.hstack((self._d_source.backprop_value,
                          self._l_source.backprop_value))

def transition_matrix(dt, dim, order):
    A = utils.integrator_matrix(dt=dt, dim=dim, order=order)
    B = utils.deriv_control_matrix(dim=dim, order=order, gain=dt)
    K = utils.position_gain_matrix(dim=dim, order=order, gain=0.1)
    return utils.closed_loop_matrix(A=A, B=B, K=K)
    # return A

if __name__ == '__main__':

    dt = 0.01
    dim = 2
    order = 1
    trial_steps = 100
    full_state_dim = dim*(order+1)

    A = transition_matrix(dt, dim=dim, order=order)

    # C matrix observes only the position
    C = np.zeros((dim, full_state_dim))
    C[:, :dim] = np.identity(dim)
    # C = np.identity(full_state_dim)

    Q_true = 1E-6 * np.identity(full_state_dim)
    R_true = 0.03 * np.identity(C.shape[0])
    x0 = np.zeros(full_state_dim)
    x0[0:2:dim] = 1
    x0[1:2:dim] = -1
    trial = utils.rollout(dt=dt, A=A, C=C, tf=dt*trial_steps, x0=x0,
                          Q=Q_true, R=R_true)

    Q_init = 0.1 * np.identity(full_state_dim)
    R_init = 0.1 * np.identity(C.shape[0])
    P0 = np.identity(full_state_dim)

    # TODO: Also optimize initialization?
    x0_source = ConstantModule(x0)
    P0_source = ConstantPSD(P0)
    Q_source = ConstantPSD(Q_init)
    R_source = ConstantPSD(R_init)

    chain = ChainConstructor(x0_src=x0_source, P0_src=P0_source,
                             Q_src=Q_source, R_src=R_source)

    print 'Constructing chain for %d steps...' % trial_steps
    for t, y in izip(trial['t'], trial['y']):
        A = transition_matrix(dt, dim=dim, order=order)
        chain.add_predict(A)
        chain.add_update(C, y)

    print 'Chain construction complete'

    # Test derivatives
    print 'Testing derivatives...'
    P_dim = len(P0_source.theta)
    Q_dim = len(Q_source.theta)
    R_dim = len(R_source.theta)
    Q_deltas = utils.vector_deltas(P_dim, 1E-6)
    R_deltas = utils.vector_deltas(R_dim, 1E-6)

    all_deltas = utils.merge_deltas((Q_deltas, R_deltas))
    chain.foreprop()
    chain_grad = chain.backprop(-1)
    def eval_func(inval):
        acc = 0
        Q = inval[acc:acc+Q_dim]
        acc += Q_dim
        R = inval[acc:acc+R_dim]

        Q_source.theta = Q
        R_source.theta = R
        ll = chain.foreprop()
        return np.atleast_1d(ll)
    utils.test_derivs(eval_func, chain_grad, chain.get_theta(), all_deltas, mode='relative', tol=1E-4)

    init_theta = chain.get_theta()
    alpha = 1E-2
    min_norm = 1E-6
    max_norm = 1

    def test_backprop_depth(depth, num_iters):
        print 'Testing backprop depth %d for %d iterations...' % (depth, num_iters)
        chain.set_theta(init_theta)
        zlls = []
        for i in range(num_iters):
            zll = chain.foreprop()
            # Min value is 8
            grad = chain.backprop(depth)[0]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_norm:
                print 'Gradient norm exceeds max!'
                print grad
                grad = grad / grad_norm

            R = R_source.out_port.value
            Q = Q_source.out_port.value

            print 'Iteration %d, zll %f' % (i+1, zll)
            print 'Q: %s' % str(Q)
            print 'R: %s' % str(R)
            print 'Gradient: %s' % str(grad)

            zlls.append(zll)
            th = chain.get_theta()
            chain.set_theta(th + alpha*grad)

            # if np.linalg.norm(grad) < min_norm:
            #     break

            chain.invalidate()
        return zlls

    test_depths = [8, 16, -1]
    num_iters = 5000
    test_zlls = [test_backprop_depth(depth=d, num_iters=num_iters) for d in test_depths]

    # Visualize results
    plt.ion()
    plt.figure()
    cmap = plt.cm.jet
    def get_color(k):
        return cmap(float(k) / (len(test_zlls)-1))

    for i, tz in enumerate(test_zlls):
        plt.plot(tz, label='d=%d' % test_depths[i], color=get_color(i))
    plt.xlabel('Step Iteration')
    plt.ylabel('Mean Observation LL')
    plt.legend(loc='best')