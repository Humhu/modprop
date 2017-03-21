"""Tools for generating test data and testing derivatives
"""
import math
import numpy as np
from numpy.random import multivariate_normal as mvn
from itertools import izip

def integrator_matrix(dt, dim, order, int_order=None):
    """Produces a discrete time dynamics integrator matrix.
    """
    N = dim*(order+1)
    A = np.identity(N)
    if int_order >= order:
        raise ValueError('int_order can be at most order')
    if int_order is None:
        int_order = order

    acc = 1
    for i in range(int_order):
        iinds = range(0, dim*(order-i))
        jinds = range(dim*(i+1), A.shape[1])
        acc = acc * dt / (i+1)
        A[iinds, jinds] = acc
    return A

def deriv_control_matrix(dim, order, gain=1):
    N = dim * (order+1)
    B = np.zeros((N, dim))
    B[dim*order:, :] = np.identity(dim) * gain
    return B

def position_gain_matrix(dim, order, gain):
    N = dim * (order+1)
    K = np.zeros((dim, N))
    K[:, 0:dim] = gain * np.identity(dim)
    return K

def closed_loop_matrix(A, B, K):
    return A - np.dot(B,K)

def rollout(dt, A, C, tf, x0, Q, R):
    t = np.linspace(start=0, stop=tf, num=math.ceil(tf/dt)+1)
    xs = [x0]
    ys = []
    x_errs = []
    y_errs = []

    x_dim = len(x0)
    y_dim = R.shape[0]

    for ti in t[:-1]:
        x_noise = mvn(np.zeros(x_dim), Q)
        x_errs.append(x_noise)
        xi = np.dot(A, xs[-1]) + x_noise
        xs.append(xi)

        y_noise = mvn(np.zeros(y_dim), R)
        ys.append(np.dot(C, xi) + y_noise)
        y_errs.append(y_noise)
    return {'t' : np.array(t), 'x' : np.array(xs), 'y' : np.array(ys),
            'x_noise' : np.array(x_errs), 'y_noise' : np.array(y_errs)}

def vector_deltas(n, delta=1E-3):
    out = []
    for i in range(n):
        v = np.zeros(n)
        v[i] = delta
        out.append(v)
    return out

def symmetric_deltas(n, delta=1E-3):
    out = []
    for i in range(n):
        m = np.zeros((n,n))
        m[i,i] = delta
        out.append(m)

    for i in range(1,n):
        for j in range(0,i):
            m = np.zeros((n,n))
            m[i,j] = delta
            m[j,i] = delta
            out.append(m)
    return out

def merge_deltas(dels):
    '''Converts a list of lists of deltas into a list of deltas.
    '''
    flattened = [[np.asarray(i).flatten('F') for i in d] for d in dels]
    zeros = [np.zeros(len(f[0])) for f in flattened]

    deltas = []
    for i, flat in enumerate(flattened):
        for f in flat:
            temp = np.copy(zeros)
            temp[i] = f
            deltas.append(np.hstack(temp))

    return deltas

def test_derivs(func, grad, x0, deltas, mode='absolute', tol=1E-6):
    '''Tests derivatives by applying a set of delta inputs.
    '''
    N = len(x0)
    y_0 = func(x0)

    if mode == 'absolute':
        err_func = lambda x, y: np.any(np.abs(x - y) > tol)
    elif mode == 'relative':
        err_func = lambda x, y: np.any((np.abs(x[x != 0] - y[x != 0]) / x[x != 0]) > tol)
    else:
        raise ValueError('mode must be absolute or relative')

    for i, delta in enumerate(deltas):
        y_real = func(x0 + delta)
        real_delta_y = y_real - y_0

        pred_delta_y = np.dot(grad, delta)
        y_pred = y_0 + pred_delta_y

        if err_func(real_delta_y, pred_delta_y):
            print 'Delta %d: %s' % (i, str(delta))
            print 'Real delta: %s' % str(y_real - y_0)
            print 'Pred delta: %s' % str(pred_delta_y)
            print 'ERROR!'
        else:
            print 'PASS!'