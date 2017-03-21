"""This module contains classes to help in construction of Kalman filter chains.
"""
import numpy as np
import modprop.core.modules_core as core
import modprop.core.backprop as bp
import modprop.modules.cost_modules as cost
import modprop.modules.basic_modules as basic
import modprop.modules.reduce_modules as redu
import modprop.modules.kalman_modules as kalman

class ChainConstructor(object):
    """Sequentially constructs and links Kalman filter modules into a chain structure.

    Parameters
    ----------
    x0_src : Module with out_port to provide initial state
    P0_src : Module with out_port to provide initial covariance
    Q_src  : Module with out_port to provide transition covariance
    R_src  : Module with out_port to provide observation covariance
    """
    def __init__(self, x0_src, P0_src, Q_src, R_src):
        self.mean_reductor = redu.MeanModule()
        self.mean_zll_sink = basic.SinkModule()
        core.link_ports(in_port=self.mean_zll_sink.in_port,
                        out_port=self.mean_reductor.mean_out)

        self.x0_source = x0_src
        self.P0_source = P0_src
        self.last_x_port = self.x0_source.out_port
        self.last_P_port = self.P0_source.out_port

        self.Q_src = Q_src
        self.R_src = R_src

        self.predicts = []
        self.updates = []
        self.likelihoods = []

    def add_predict(self, A):
        predict = kalman.PredictModule(A)
        core.link_ports(in_port=predict.x_in, out_port=self.last_x_port)
        core.link_ports(in_port=predict.P_in, out_port=self.last_P_port)
        core.link_ports(in_port=predict.Q_in, out_port=self.Q_src.out_port)

        self.last_x_port = predict.x_out
        self.last_P_port = predict.P_out
        self.predicts.append(predict)
        return predict

    def add_update(self, C, y):
        update = kalman.UpdateModule(y=y, C=C)
        core.link_ports(in_port=update.x_in, out_port=self.last_x_port)
        core.link_ports(in_port=update.P_in, out_port=self.last_P_port)
        core.link_ports(in_port=update.R_in, out_port=self.R_src.out_port)

        log_likelihood = cost.LogLikelihoodModule()
        core.link_ports(in_port=log_likelihood.x_in, out_port=update.v_out)
        core.link_ports(in_port=log_likelihood.S_in, out_port=update.S_out)
        core.link_ports(in_port=self.mean_reductor.create_input(),
                        out_port=log_likelihood.ll_out)

        self.last_x_port = update.x_out
        self.last_P_port = update.P_out
        self.updates.append(update)
        self.likelihoods.append(log_likelihood)
        return update, log_likelihood

    def foreprop(self):
        core.iterative_foreprop(self.x0_source)
        core.iterative_foreprop(self.P0_source)
        core.iterative_foreprop(self.Q_src)
        core.iterative_foreprop(self.R_src)
        return self.mean_zll_sink.value

    def invalidate(self):
        core.iterative_invalidate(self.x0_source)
        core.iterative_invalidate(self.P0_source)
        core.iterative_invalidate(self.Q_src)
        core.iterative_invalidate(self.R_src)

    def backprop(self, max_depth=-1):
        if max_depth < 0:
            acc = bp.AccumulatedBackprop(do_dx=np.identity(1))
        else:
            acc = bp.TruncatedBackprop(do_dx=np.identity(1), depths=max_depth)
        self.mean_zll_sink.backprop_value = acc
        core.iterative_backprop(self.mean_zll_sink)
        return np.hstack((self.Q_src.backprop_value,
                          self.R_src.backprop_value))

    def get_theta(self):
        return np.hstack((self.Q_src.theta,
                          self.R_src.theta))

    def set_theta(self, th):
        n_Q = len(self.Q_src.theta)
        n_R = len(self.R_src.theta)
        if n_Q + n_R != len(th):
            raise ValueError('Theta dimension mismatch')

        self.Q_src.theta = th[:n_Q]
        self.R_src.theta = th[n_Q:]

    @property
    def mean_observation_likelihood(self):
        return self.mean_zll_sink.value

    @property
    def latest_x(self):
        return self.last_x_port.value

    @property
    def latest_P(self):
        return self.last_P_port.value
