"""This module contains Module implementations that perform basic mathematical operations.
"""
import numpy as np
from modprop.core.modules_core import ModuleBase, InputPort, OutputPort

class AdditionModule(ModuleBase):
    """A module that computes a sum.

    Input Ports
    -----------
    left_port  : ND-array left-hand-side term
    right_port : ND-array right-hand-side term

    Output Ports
    ------------
    out_port  : ND-array sum, computed as left + right
    """

    def __init__(self):
        ModuleBase.__init__(self)
        self._left_port = InputPort(self)
        self._right_port = InputPort(self)
        self._out_port = OutputPort(self)

        ModuleBase.register_inputs(self, self._left_port)
        ModuleBase.register_inputs(self, self._right_port)
        ModuleBase.register_outputs(self, self._out_port)

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        out = self._left_port.value + self._right_port.value
        return self._out_port.foreprop(out)

    def backprop(self):
        if not self.backprop_ready():
            return []

        back = []

        dim = len(self._right_port.value.flat)
        dout_dright = np.identity(dim)
        do_dright = self._out_port.chain_backprop(dy_dx=dout_dright)
        back += self._right_port.backprop(do_dright)

        dout_dleft = np.identity(dim)
        do_dleft = self._out_port.chain_backprop(dy_dx=dout_dleft)
        back += self._left_port.backprop(do_dleft)

        return back

    @property
    def left_port(self):
        return self._left_port

    @property
    def right_port(self):
        return self._right_port

    @property
    def out_port(self):
        return self._out_port

class DifferenceModule(ModuleBase):
    """A module that computes a difference.

    Input Ports
    -----------
    left_port  : ND-array left-hand-side term
    right_port : ND-array right-hand-side term

    Output Ports
    ------------
    out_port  : ND-array difference, computed as left - right
    """

    def __init__(self):
        ModuleBase.__init__(self)
        self._left_port = InputPort(self)
        self._right_port = InputPort(self)
        self._out_port = OutputPort(self)

        ModuleBase.register_inputs(self, self._left_port)
        ModuleBase.register_inputs(self, self._right_port)
        ModuleBase.register_outputs(self, self._out_port)

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        out = self._left_port.value - self._right_port.value
        return self._out_port.foreprop(out)

    def backprop(self):
        if not self.backprop_ready():
            return []

        back = []

        dim = len(self._right_port.value.flat)
        dout_dright = - np.identity(dim)
        do_dright = self._out_port.chain_backprop(dy_dx=dout_dright)
        back += self._right_port.backprop(do_dright)

        dout_dleft = np.identity(dim)
        do_dleft = self._out_port.chain_backprop(dy_dx=dout_dleft)
        back += self._left_port.backprop(do_dleft)

        return back

    @property
    def left_port(self):
        return self._left_port

    @property
    def right_port(self):
        return self._right_port

    @property
    def out_port(self):
        return self._out_port


class MatrixProductModule(ModuleBase):
    """A module that computes a matrix product.

    Input Ports
    -----------
    left_port  : ND-array left-hand-side factor
    right_port : ND-array right-hand-side factor

    Output Ports
    ------------
    out_port  : ND-array product, computed as np.dot(left, right)
    """

    def __init__(self):
        ModuleBase.__init__(self)
        self._left_port = InputPort(self)
        self._right_port = InputPort(self)
        self._out_port = OutputPort(self)

        ModuleBase.register_inputs(self, self._left_port)
        ModuleBase.register_inputs(self, self._right_port)
        ModuleBase.register_outputs(self, self._out_port)

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        out = np.dot(self._left_port.value, self._right_port.value)
        return self._out_port.foreprop(out)

    def backprop(self):
        if not self.backprop_ready():
            return []

        back = []

        if len(self._right_port.value.shape) == 1:
            n = 1
        elif len(self._right_port.value.shape) == 2:
            n = self._right_port.value.shape[1]
        else:
            raise ValueError('Right input must be at most 2D')

        dout_dright = np.kron(np.identity(n), self._left_port.value)
        do_dright = self._out_port.chain_backprop(dy_dx=dout_dright)
        back += self._right_port.backprop(do_dright)

        m = self._left_port.value.shape[0]
        dout_dleft = np.kron(self._right_port.value.T, np.identity(m))
        do_dleft = self._out_port.chain_backprop(dy_dx=dout_dleft)
        back += self._left_port.backprop(do_dleft)

        return back

    @property
    def left_port(self):
        return self._left_port

    @property
    def right_port(self):
        return self._right_port

    @property
    def out_port(self):
        return self._out_port


class ExponentialModule(ModuleBase):
    """A module that exponentiates its input.

    Input Ports
    -----------
    in_port : ND-array input to be exponentiated

    Output Ports
    ------------
    out_port : ND-array exponentiated output, same size as input
    """

    def __init__(self):
        ModuleBase.__init__(self)
        self._in_port = InputPort(self)
        self._out_port = OutputPort(self)

        ModuleBase.register_inputs(self, self._in_port)
        ModuleBase.register_outputs(self, self._out_port)

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        out = np.exp(self._in_port.value)
        return self._out_port.foreprop(out)

    def backprop(self):
        if not self.backprop_ready():
            return []

        # do_dout = self._out_port.backprop_value
        dout_din = np.diag(self._out_port.value.flatten('F'))
        # do_din = np.dot(do_dout, dout_din)
        do_din = self._out_port.chain_backprop(dy_dx=dout_din)
        return self._in_port.backprop(do_din)

    @property
    def in_port(self):
        return self._in_port

    @property
    def out_port(self):
        return self._out_port
