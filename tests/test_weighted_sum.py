import numpy as np
import scipy as sp
from itertools import izip

from modprop import *
from utils import *

if __name__ == '__main__':

    dim = 3

    outs = []
    weights = []
    sum_mod = WeightedSumModule()
    sum_val = SinkModule()
    link_ports(in_port=sum_val.in_port, out_port=sum_mod.out_port)

    def init():
        A = ConstantModule(np.random.rand(dim))
        outs.append(A)

        w = ConstantModule(1.0)
        weights.append(w)

        ina, wa = sum_mod.create_inputs()
        link_ports(in_port=ina, out_port=A.out_port)
        link_ports(in_port=wa, out_port=w.out_port)

    def foreprop():
        iterative_invalidate(sum_val)
        for o, w in izip(outs, weights):
            iterative_invalidate(o)
            iterative_invalidate(w)

        for o, w in izip(outs, weights):
            iterative_foreprop(o)
            iterative_foreprop(w)
    
    def func(x):
        acc = 0
        for o, w in izip(outs, weights):
            o.value = x[acc:acc + dim]
            acc += dim
            w.value = float(x[acc:acc + 1])
            acc += 1
        
        foreprop()
        return sum_val.value

    for i in range(3):
        init()
    
    sum_val.backprop_value = AccumulatedBackprop(np.identity(dim))
    foreprop()
    iterative_backprop(sum_val)

    init_value = []
    grad = []
    for o, w in izip(outs, weights):
        grad.append(o.out_port.backprop_value)
        grad.append(w.out_port.backprop_value)
        #grad = np.hstack((grad, o.out_port.backprop_value))
        #grad = np.hstack((grad, w.out_port.backprop_value))
        init_value = np.hstack((init_value, o.value, w.value))
    grad = np.concatenate(grad, axis=0).T

    # Form deltas and get initial value
    deltas = []
    for o, w in izip(outs, weights):
        deltas.append(vector_deltas(dim, 1E-6))
        deltas.append(vector_deltas(1, 1E-6))

    merged_deltas = merge_deltas(deltas)

    test_derivs(func, grad, init_value, merged_deltas, mode='relative', tol=1E-3)