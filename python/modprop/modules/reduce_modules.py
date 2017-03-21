"""This module contains Module implementations that preform reduction operations on multiple inputs.
"""
import numpy as np
from modprop.core.modules_core import ModuleBase, InputPort, OutputPort

class MeanModule(ModuleBase):
    """Returns the average of its inputs.

    This module implements a more efficient but less robust foreprop_ready which
    assumes that foreprop_ready is only called by input ports upon readying.

    Input Ports
    -----------
    Call create_input() to get a new input port

    Output Ports
    ------------
    mean_out : Output mean
    """
    def __init__(self):
        ModuleBase.__init__(self)

        self._mean_inputs = []
        self._valid_count = 0
        self._mean_out = OutputPort(self)

        ModuleBase.register_outputs(self, self._mean_out)

    def is_invalid(self):
        #return not any(self._mean_inputs) and not any(self._outputs)
        return self._valid_count == 0 and not self._mean_out

    def invalidate(self):
        self._valid_count = 0
        return ModuleBase.invalidate(self)

    def foreprop_ready(self):
        """Efficient override of ModuleBase.foreprop_ready

        Assumes that foreprop_ready is only called by member InputPorts when
        they are ready to foreprop. Each call to foreprop_ready increments an
        internal counter, allowing for faster checking of all input readiness.
        """
        #return all(self._mean_inputs) and not all(self._outputs)
        self._valid_count += 1
        return not self._mean_out and self._valid_count >= len(self._mean_inputs)

    def foreprop(self):
        if not self.foreprop_ready():
            return []

        acc = 0
        for i in self._mean_inputs:
            acc += i.value
        mean_val = np.array([acc / len(self._mean_inputs)])
        return self._mean_out.foreprop(mean_val)

    def backprop_ready(self):
        return ModuleBase.backprop_ready(self)

    def backprop(self):
        if not self._mean_out.backprop_ready():
            return []

        # do_dmean = self._mean_out.backprop_value
        dmean_din = 1.0 / len(self._mean_inputs)
        ret = []
        for i in self._mean_inputs:
            do_di = self._mean_out.chain_backprop(dy_dx=dmean_din)
            ret += i.backprop(do_di)
        return ret

    def create_input(self):
        """Creates a new input port to this module.

        Should be used when adding more sources to the mean.

        Returns
        -------
        port : The newly created port.
        """
        new_port = InputPort(self)
        self._mean_inputs.append(new_port)
        ModuleBase.register_inputs(self, new_port)
        return new_port

    @property
    def mean_out(self):
        return self._mean_out