"""This module contains basic Module implementations.
"""
import numpy as np
from modprop.core.modules_core import ModuleBase, InputPort, OutputPort, iterative_invalidate

class ConstantModule(ModuleBase):
    """Outputs a constant value.

    Like all other modules, this one must be invalidated in order to foreprop properly.

    Input Ports
    -----------
    None

    Output Ports
    ------------
    out_port : The constant output

    Parameters
    ----------
    value : The output value (default: None)
    """
    def __init__(self, value):
        ModuleBase.__init__(self)
        self._out = OutputPort(self)
        self._value = value

        ModuleBase.register_outputs(self, self._out)

    def foreprop(self):
        if not self.foreprop_ready():
            return []
        return self._out.foreprop(self._value)

    def backprop(self):
        return []

    @property
    def out_port(self):
        return self._out

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
        iterative_invalidate(self)

    @property
    def backprop_value(self):
        return self._out.backprop_value

class SinkModule(ModuleBase):
    """A module that terminates a Module tree.

    Input Ports
    -----------
    in_port : The input to terminate

    Output Ports
    ------------
    None
    """
    def __init__(self):
        ModuleBase.__init__(self)
        self._in = InputPort(self)
        self._do_dx = None
        ModuleBase.register_inputs(self, self._in)

    def foreprop(self):
        pass

    def backprop(self):
        if not self.backprop_ready():
            return []

        return self._in.backprop(self._do_dx.copy())

    @property
    def in_port(self):
        return self._in

    @property
    def value(self):
        return self._in.value

    @property
    def backprop_value(self):
        return self._do_dx

    @backprop_value.setter
    def backprop_value(self, do_dx):
        self._do_dx = do_dx
