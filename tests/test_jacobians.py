import numpy as np
import scipy as sp
from itertools import izip

from modprop import *
from utils import *

if __name__ == '__main__':

    N_x = 4
    N_y = 2

    A = np.random.rand(N_x, N_x)
    C = np.random.rand(N_y, N_x)
    x = np.random.rand(N_x)
    P = np.random.rand(N_x, N_x)
    P = np.dot(P, P.T) + 1E-1*np.identity(N_x)
    Q = np.random.rand(N_x, N_x)
    Q = np.dot(Q, Q.T) + 1E-1*np.identity(N_x)
    R = np.random.rand(N_y, N_y)
    R = np.dot(R, R.T) + 1E-1*np.identity(N_y)
    y = np.dot(C, x) + np.random.multivariate_normal(mean=np.zeros(N_y), cov=R)

    N_P = N_x * N_x
    N_Q = N_P
    N_R = N_y * N_y
    N_v = N_y
    N_S = N_v * N_v
    P_shape = (N_x, N_x)
    Q_shape = P_shape
    R_shape = (N_y, N_y)

    # Test predict module
    x_pred = ConstantModule(x)
    P_pred = ConstantModule(P)
    Q_pred = ConstantModule(Q)
    predict_1 = PredictModule(A=A)
    xo_pred = SinkModule()
    Po_pred = SinkModule()
    link_ports(predict_1.x_in, x_pred.out_port)
    link_ports(predict_1.P_in, P_pred.out_port)
    link_ports(predict_1.Q_in, Q_pred.out_port)
    link_ports(xo_pred.in_port, predict_1.x_out)
    link_ports(Po_pred.in_port, predict_1.P_out)
    def foreprop_predict():
        iterative_invalidate(x_pred)
        iterative_invalidate(P_pred)
        iterative_invalidate(Q_pred)

        iterative_foreprop(x_pred)
        iterative_foreprop(P_pred)
        iterative_foreprop(Q_pred)

    # Test update module
    x_up = ConstantModule(x)
    P_up = ConstantModule(P)
    R_up = ConstantModule(R)
    update_1 = UpdateModule(y=y, C=C)
    xo_up = SinkModule()
    Po_up = SinkModule()
    vo_up = SinkModule()
    So_up = SinkModule()
    link_ports(update_1.x_in, x_up.out_port)
    link_ports(update_1.P_in, P_up.out_port)
    link_ports(update_1.R_in, R_up.out_port)
    link_ports(xo_up.in_port, update_1.x_out)
    link_ports(Po_up.in_port, update_1.P_out)
    link_ports(vo_up.in_port, update_1.v_out)
    link_ports(So_up.in_port, update_1.S_out)
    def foreprop_update():
        iterative_invalidate(x_up)
        iterative_invalidate(P_up)
        iterative_invalidate(R_up)

        iterative_foreprop(x_up)
        iterative_foreprop(P_up)
        iterative_foreprop(R_up)

    # Test predict-update chain
    x_co = ConstantModule(x)
    P_co = ConstantModule(P)
    Q_co = ConstantModule(Q)
    R_co = ConstantModule(R)
    predict_2 = PredictModule(A=A)
    update_2 = UpdateModule(y=y, C=C)
    xo_co = SinkModule()
    Po_co = SinkModule()
    link_ports(predict_2.x_in, x_co.out_port)
    link_ports(predict_2.P_in, P_co.out_port)
    link_ports(predict_2.Q_in, Q_co.out_port)
    link_ports(update_2.x_in, predict_2.x_out)
    link_ports(update_2.P_in, predict_2.P_out)
    link_ports(update_2.R_in, R_co.out_port)
    link_ports(xo_co.in_port, update_2.x_out)
    link_ports(Po_co.in_port, update_2.P_out)
    def foreprop_co():
        iterative_invalidate(x_co)
        iterative_invalidate(P_co)
        iterative_invalidate(Q_co)
        iterative_invalidate(R_co)

        iterative_foreprop(x_co)
        iterative_foreprop(P_co)
        iterative_foreprop(Q_co)
        iterative_foreprop(R_co)

    # Test log-likelihood
    ll_dim = 3
    test_cov = np.random.rand(ll_dim, ll_dim) + np.identity(ll_dim)
    test_cov = np.dot(test_cov, test_cov.T)
    test_sample = np.random.rand(ll_dim)
    x_ll = ConstantModule(test_sample)
    S_ll = ConstantModule(test_cov)
    loglike_1 = LogLikelihoodModule()
    llo_ll = SinkModule()
    link_ports(loglike_1.x_in, x_ll.out_port)
    link_ports(loglike_1.S_in, S_ll.out_port)
    link_ports(llo_ll.in_port, loglike_1.ll_out)
    def foreprop_ll():
        iterative_invalidate(x_ll)
        iterative_invalidate(S_ll)
        iterative_foreprop(x_ll)
        iterative_foreprop(S_ll)

    # Test Cholesky
    chol_dim = 3
    chol_l_dim = (chol_dim*(chol_dim-1))/2
    l = np.random.rand(chol_l_dim)
    d = np.random.rand(chol_dim)
    l_src = ConstantModule(l)
    d_src = ConstantModule(d)
    chol = CholeskyReshapeModule()
    chol_out = SinkModule()
    link_ports(chol.l_in, l_src.out_port)
    link_ports(chol.d_in, d_src.out_port)
    link_ports(chol_out.in_port, chol.S_out)

    def foreprop_chol():
        iterative_invalidate(l_src)
        iterative_invalidate(d_src)

        iterative_foreprop(l_src)
        iterative_foreprop(d_src)

    # Generate deltas
    x_deltas = vector_deltas(N_x, 1E-6)
    P_deltas = symmetric_deltas(N_x, 1E-6)
    Q_deltas = P_deltas
    R_deltas = symmetric_deltas(N_y, 1E-6)

    ll_x_deltas = vector_deltas(ll_dim)
    ll_S_deltas = symmetric_deltas(ll_dim, 1E-3)

    chol_l_deltas = vector_deltas(chol_l_dim, 1E-6)
    chol_d_deltas = vector_deltas(chol_dim, 1E-6)

    pred_deltas = merge_deltas((x_deltas, P_deltas, Q_deltas))
    up_deltas = merge_deltas((x_deltas, P_deltas, R_deltas))
    co_deltas = merge_deltas((x_deltas, P_deltas, Q_deltas, R_deltas))
    ll_deltas = merge_deltas((ll_x_deltas, ll_S_deltas))
    chol_deltas = merge_deltas((chol_l_deltas, chol_d_deltas))

    def foreprop_all():
        foreprop_predict()
        foreprop_update()
        foreprop_co()
        foreprop_ll()
        foreprop_chol()

    foreprop_all()
    up_x_out = xo_up.value
    up_P_out = Po_up.value.flatten('F')
    up_v_out = vo_up.value
    up_S_out = So_up.value.flatten('F')

    co_x_out = xo_co.value
    co_P_out = Po_co.value.flatten('F')

    pred_init = np.hstack([x_pred.value.flatten('F'),
                           P_pred.value.flatten('F'),
                           Q_pred.value.flatten('F')])
    up_init = np.hstack([x_up.value.flatten('F'),
                         P_up.value.flatten('F'),
                         R_up.value.flatten('F')])
    co_init = np.hstack([x_co.value.flatten('F'),
                         P_co.value.flatten('F'),
                         Q_co.value.flatten('F'),
                         R_co.value.flatten('F')])
    ll_init = np.hstack([x_ll.value.flatten('F'),
                         S_ll.value.flatten('F')])
    chol_init = np.hstack([l, d])

    def pred_func(inval):
        acc = 0
        x = inval[acc:acc+N_x]
        acc += N_x
        P = inval[acc:acc+N_P]
        acc += N_P
        Q = inval[acc:acc+N_Q]

        x_pred.value = x
        P_pred.value = np.reshape(P, P_shape, 'F')
        Q_pred.value = np.reshape(Q, P_shape, 'F')
        foreprop_all()

        x_ret = xo_pred.value
        P_ret = Po_pred.value.flatten('F')
        return np.hstack((x_ret, P_ret))

    def pred_grad():
        N_all = N_x + N_P

        do_x = np.zeros((N_all, N_x))
        do_x[0:N_x, :] = np.identity(N_x)
        x_acc = AccumulatedBackprop(do_x)

        do_P = np.zeros((N_all, N_P))
        do_P[N_x:, :] = np.identity(N_P)
        P_acc = AccumulatedBackprop(do_P)

        xo_pred.backprop_value = x_acc
        Po_pred.backprop_value = P_acc
        iterative_backprop(xo_pred)
        iterative_backprop(Po_pred)

        grad_x = x_pred.out_port.backprop_value
        grad_P = P_pred.out_port.backprop_value
        grad_Q = Q_pred.out_port.backprop_value
        return np.hstack((grad_x, grad_P, grad_Q))

    def up_func(inval):
        acc = 0
        x_in = inval[acc:acc+N_x]
        acc += N_x
        P_in = inval[acc:acc+N_P]
        acc += N_P
        R_in = inval[acc:acc+N_R]

        x_up.value = x_in
        P_up.value = np.reshape(P_in, P_shape, 'F')
        R_up.value = np.reshape(R_in, R_shape, 'F')
        foreprop_all()

        x_ret = xo_up.value
        P_ret = Po_up.value.flatten('F')
        v_ret = vo_up.value
        S_ret = So_up.value.flatten('F')
        # x_ret = up_x_out
        # P_ret = up_P_out
        # v_ret = up_v_out
        # S_ret = up_S_out
        return np.hstack((x_ret, P_ret, v_ret, S_ret))

    def up_grad():
        N_all = N_x + N_P + N_v + N_S
        acc = 0

        do_x = np.zeros((N_all, N_x))
        do_x[acc:acc+N_x, :] = np.identity(N_x)
        xo_up.backprop_value = AccumulatedBackprop(do_x)
        acc += N_x

        do_P = np.zeros((N_all, N_P))
        do_P[acc:acc+N_P, :] = np.identity(N_P)
        Po_up.backprop_value = AccumulatedBackprop(do_P)
        acc += N_P

        do_v = np.zeros((N_all, N_v))
        do_v[acc:acc+N_v, :] = np.identity(N_v)
        vo_up.backprop_value = AccumulatedBackprop(do_v)
        acc += N_v

        do_S = np.zeros((N_all, N_S))
        do_S[acc:acc+N_S, :] = np.identity(N_S)
        So_up.backprop_value = AccumulatedBackprop(do_S)
        acc += N_S

        iterative_backprop(xo_up)
        iterative_backprop(Po_up)
        iterative_backprop(vo_up)
        iterative_backprop(So_up)

        grad_x = x_up.out_port.backprop_value
        grad_P = P_up.out_port.backprop_value
        grad_R = R_up.out_port.backprop_value
        return np.hstack((grad_x, grad_P, grad_R))

    def co_func(inval):
        acc = 0
        x = inval[acc:acc+N_x]
        acc += N_x
        P = inval[acc:acc+N_P]
        acc += N_P
        Q = inval[acc:acc+N_Q]
        acc += N_Q
        R = inval[acc:acc+N_R]

        x_co.value = x
        P_co.value = np.reshape(P, P_shape, 'F')
        Q_co.value = np.reshape(Q, Q_shape, 'F')
        R_co.value = np.reshape(R, R_shape, 'F')
        foreprop_all()

        # x_ret = xo_co.value
        x_ret = co_x_out
        P_ret = Po_co.value.flatten('F')
        return np.hstack((x_ret, P_ret))

    def co_grad():
        N_all = N_x + N_P

        do_x = np.zeros((N_all, N_x))
        do_x[0:N_x, :] = 0*np.identity(N_x)
        x_acc = AccumulatedBackprop(do_x)

        do_P = np.zeros((N_all, N_P))
        do_P[N_x:, :] = np.identity(N_P)
        P_acc = AccumulatedBackprop(do_P)

        xo_co.backprop_value = x_acc
        Po_co.backprop_value = P_acc
        iterative_backprop(xo_co)
        iterative_backprop(Po_co)

        grad_x = x_co.out_port.backprop_value
        grad_P = P_co.out_port.backprop_value
        grad_Q = Q_co.out_port.backprop_value
        grad_R = R_co.out_port.backprop_value
        return np.hstack((grad_x, grad_P, grad_Q, grad_R))

    def ll_func(inval):
        acc = 0
        x_in = inval[acc:acc+ll_dim]
        acc += ll_dim
        S_in = inval[acc:acc+ll_dim*ll_dim]

        x_ll.value = x_in
        S_ll.value = np.reshape(S_in, (ll_dim, ll_dim), 'F')
        foreprop_all()

        return np.array([llo_ll.value])

    def ll_grad():
        do_ll = np.identity(1)
        llo_ll.backprop_value = AccumulatedBackprop(do_ll)
        iterative_backprop(llo_ll)

        grad_x = x_ll.out_port.backprop_value
        grad_P = S_ll.out_port.backprop_value
        return np.hstack((grad_x, grad_P))

    def chol_func(inval):
        acc = 0

        l_in = inval[acc:acc+chol_l_dim]
        acc += chol_l_dim
        d_in = inval[acc:acc+chol_dim]

        l_src.value = l_in
        d_src.value = d_in
        foreprop_all()

        return chol_out.value.flatten()

    def chol_grad():
        do_c = np.identity(chol_dim*chol_dim)
        c_acc = AccumulatedBackprop(do_c)
        chol_out.backprop_value = c_acc
        iterative_backprop(chol_out)

        grad_l = l_src.out_port.backprop_value
        grad_d = d_src.out_port.backprop_value
        return np.hstack((grad_l, grad_d))

    dpred = pred_grad()
    dup = up_grad()
    dco = co_grad()
    dll = ll_grad()
    cll = chol_grad()

    # print 'Testing predict module...'
    # test_derivs(pred_func, dpred, pred_init, pred_deltas)

    # print 'Testing update module...'
    # test_derivs(up_func, dup, up_init, up_deltas, mode='relative', tol=1E-3)

    # test_derivs(co_func, dco, co_init, co_deltas)

    print 'Testing log-likelihood module...'
    test_derivs(ll_func, dll, ll_init, ll_deltas)

    # print 'Testing cholesky module...'
    # test_derivs(chol_func, cll, chol_init, chol_deltas)