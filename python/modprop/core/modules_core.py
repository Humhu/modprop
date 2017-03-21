"""This module contains base classes and types for creating new Modules and using module trees.
"""
import abc
from collections import deque
import numpy as np

class ModuleBase(object):
    """The base interface for all modules. Modules must inherit from this interface.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._inputs = []
        self._outputs = []

    def register_inputs(self, *args):
        """Registers inputs to this module.

        Parameters
        ----------
        inputs : Variable number of inputs to register.
        """
        for arg in args:
            if not isinstance(arg, InputPort):
                raise ValueError('All inputs must be InputPort type')
            self._inputs.append(arg)

    def register_outputs(self, *args):
        """Registers outputs to this module.

        Parameters
        ----------
        outputs : Variable number of outputs to register.
        """
        for arg in args:
            if not isinstance(arg, OutputPort):
                raise ValueError('All outputs must be OutputPort type')
            self._outputs.append(arg)

    def foreprop_ready(self):
        """Returns if the module is ready to forward-propagate.

        Default implementation returns true when all inputs are ready and
        not all outputs are set.

        Returns
        -------
        ready : Boolean denoting if the module is ready to foreprop
        """
        return all(self._inputs) and not all(self._outputs)

    @abc.abstractmethod
    def foreprop(self):
        """Perform forward-propagation for this module.

        Returns
        -------
        ready : The aggregated return list from all forepropped output ports.
        """
        return []

    def backprop_ready(self):
        """Returns if the module is ready to backward-propagate.

        Typically this is when all outputs have received all backprops.
        Default implementation checks to see if all outputs are ready to backprop.

        Returns
        -------
        ready : Boolean denoting if the module is ready to backprop
        """
        return all([o.backprop_ready() for o in self._outputs])

    @abc.abstractmethod
    def backprop(self):
        """Perform backward-propagation for this module.

        Returns
        -------
        ready : The aggregated return list from all backpropped input ports.
        """
        return []

    def is_invalid(self):
        """Returns if the module is fully invalidated.

        Typically this is when all ports are invalidated.
        Default implementation checks to see if all ports are invalidated.

        Returns
        -------
        invalid : Boolean denoting if this module is fully invalid
        """
        return not any(self._inputs) and not any(self._outputs)

    def invalidate(self):
        """Invalidate this modules' inputs and outputs.

        Default implementation first checks to see if the module is already invalid.
        If it is not, it calls invalidate on all inputs and outputs.

        Returns
        -------
        ready : List of modules to invalidate next.
        """
        if self.is_invalid():
            return []

        ready = []
        for i in self._inputs:
            ready += i.invalidate()
        for o in self._outputs:
            ready += o.invalidate()
        return ready

# TODO Ways to unregister port connections
class InputPort(object):
    """An input to a module. Ideally instantiated as a member of the module.

    Parameters
    ----------
    module : The owning module. Must implement the ModuleBase interface.
    """
    def __init__(self, module):
        if not isinstance(module, ModuleBase):
            raise ValueError('module must implement ModuleBase')

        self._module = module
        self._value = None
        self._source = None

    def __nonzero__(self):
        """Override of Python boolean test operator to return if the port has a value.

        Returns
        -------
        ready : Boolean denoting if the port has a valid value.
        """
        return self._value is not None

    def invalidate(self):
        """Invalidate this input port and propagate to the module and source.

        Returns
        -------
        valid : List of modules to invalidate next.
        """
        # If we're already invalidated, there's nothing for us to do here
        if not self:
            return []

        self._value = None
        valid = []

        # If the owning module is not invalid, return it
        if not self._module.is_invalid():
            valid.append(self._module)

        # Propagate invalidation to source
        if self._source is not None:
            valid += self._source.invalidate()

        return valid

    def foreprop(self, v):
        """Set this port's value and forward-propagate.

        Typically only called by OutputPorts.

        Parameters
        ----------
        v : The value to set the port to.

        Returns
        -------
        ready : List of modules to foreprop next.
        """
        self._value = v

        if self._module.foreprop_ready():
            return [self._module]
        else:
            return []

    def backprop(self, do_dx):
        """Give this port a backpropagation accumulator to pass on.

        Typically called by the owning module.

        Parameters
        ----------
        do_dx : Numpy 2D array Jacobian[i,j] of tree outputs[i] w.r.t. this input port elements[j].

        Returns
        -------
        ready : List of modules to backprop next.
        """
        if self._source is not None:
            return self._source.backprop(do_dx)
        else:
            return []

    def register_source(self, src):
        """Register an OutputPort source for this port.

        Parameters
        ----------
        src : OutputPort to take as the source of this port.
        """
        if not isinstance(src, OutputPort):
            raise ValueError('src must be OutputPort')
        self._source = src

    @property
    def value(self):
        return self._value

class OutputPort(object):
    """An output from a module. Typically instantiated as a module member.

    Parameters
    ----------
    module : The owning module. Must implement the ModuleBase interface.
    """
    def __init__(self, module):
        if not isinstance(module, ModuleBase):
            raise ValueError('module must implement ModuleBase')

        self._module = module

        self._backprop_acc = None
        self._num_backs = 0

        self._value = None
        self._consumers = []

    def __nonzero__(self):
        """Override of Python boolean test operator to return whether this port has a value.
        """
        return self.value is not None

    @property
    def num_consumers(self):
        """Return the number of registered consumers.
        """
        return len(self._consumers)

    @property
    def value(self):
        return self._value

    def register_consumer(self, con):
        """Register an InputPort consumer to this port.
        """
        if not isinstance(con, InputPort):
            raise ValueError('Consumer must be InputPort')
        self._consumers.append(con)

    def invalidate(self):
        """Invalidate this port and propagate.

        Returns
        -------
        valid : List of modules to invalidate next
        """
        # If we're already invalid, there's nothing to do
        if not self:
            return []

        self._backprop_acc = None
        self._num_backs = 0
        self._value = None

        valid = []
        if not self._module.is_invalid():
            valid.append(self._module)

        for con in self._consumers:
            valid += con.invalidate()
        return valid

    def foreprop(self, v):
        """Perform forward-propagation through this output.

        Typically called by the owning module.

        Parameters
        ----------
        v : The value to set this port to.

        Returns
        -------
        ready : List of modules to foreprop next.
        """
        self._value = v
        ready = []
        for con in self._consumers:
            ready += con.foreprop(self._value)
        return ready

    def backprop(self, do_dx):
        """Perform backward-propagation through this output.

        Typically called by a connected InputPort.
        Only propagates when data from all registered consumers is received.

        Parameters
        ----------
        do_dx : Numpy 2D array Jacobian[i,j] of tree outputs[i] w.r.t. this input port elements[j]

        Returns
        -------
        ready : List of modules to backprop next
        """
        if do_dx is None:
            raise RuntimeError('OutputPort received None backprop value.')

        do_dx.tick_descent()
        if self._backprop_acc is None:
            self._backprop_acc = do_dx
        else:
            self._backprop_acc += do_dx
        self._num_backs += 1

        # Check for backprop errors
        if self._num_backs > len(self._consumers):
            errstr = 'Received %d backprops for %d consumers!' % (self._num_backs, len(self._consumers))
            raise RuntimeError(errstr)

        # If we've heard from every consumer and our module is ready
        if self.backprop_ready() and self._module.backprop_ready():
            return [self._module]
        else:
            return []

    def backprop_ready(self):
        """Returns if this port has heard from all its consumers.
        """
        return self._num_backs == self.num_consumers

    def chain_backprop(self, dy_dx=None):
        """Returns a copy of this port's backprop accumulator right-multiplied by the
        given gradient. If the port has not received a backprop, returns None.
        """
        if self._backprop_acc is None:
            return None
            #raise RuntimeError('Cannot chain backprop! Port has not received do_dx.')

        out_acc = self._backprop_acc.copy()
        if dy_dx is not None:
            out_acc = out_acc * dy_dx
        return out_acc

    @property
    def backprop_accumulator(self):
        """Returns the port's backprop accumulator.
        """
        return self._backprop_acc

    @property
    def backprop_value(self):
        if self._backprop_acc is None:
            return 0
        else:
            return self._backprop_acc.retrieve()

def link_ports(in_port, out_port):
    """Join an input and output port together.

    Parameters
    ----------
    in_port  : InputPort to join
    out_port : OutputPort to join
    """
    if not isinstance(in_port, InputPort):
        raise ValueError('in_port must be an InputPort.')
    if not isinstance(out_port, OutputPort):
        raise ValueError('out_port must be an OutputPort.')

    in_port.register_source(out_port)
    out_port.register_consumer(in_port)

# @profile
def iterative_operation(init_module, op):
    # TODO Allow taking list of initial modules
    """Iteratively perform an operation on a module tree.

    This function should be used instead of recursive calls, which do not scale
    to deep trees very well.

    Parameters
    ----------
    init_module : Module to begin iteration on
    op          : Function that takes a module and returns a list of modules to operate on next
    """
    to_prop = deque()
    to_prop.append(init_module)
    while len(to_prop) > 0:
        current = to_prop.popleft()
        ready_children = op(current)
        for c in ready_children:
            to_prop.append(c)

def iterative_foreprop(init_module):
    """Iterative forward-pass propagation on a module tree.
    """
    op = lambda x: x.foreprop()
    iterative_operation(init_module, op)

def iterative_backprop(init_module):
    """Iterative backward-pass propagation on a module tree.
    """
    op = lambda x: x.backprop()
    iterative_operation(init_module, op)

def iterative_invalidate(init_module):
    """Iterative invalidation on a module tree.
    """
    op = lambda x: x.invalidate()
    iterative_operation(init_module, op)
