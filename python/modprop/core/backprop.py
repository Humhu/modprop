"""This module contains specialized backpropagation-related classes and interfaces.
"""
import abc
import numpy as np

def sum_accumulators(accs):
    """Takes a list of accumulators or Nones and adds them together.
    """
    valid = [acc for acc in accs if acc]
    if len(valid) == 0:
        return None

    ret = valid[0]
    for v in valid[1:]:
        ret += v
    return ret

class BackpropInterface(object):
    """Interface for all backprop accumulators.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def clear(self):
        """Resets the accumulator to zero.
        """
        pass

    @abc.abstractmethod
    def copy(self):
        """Returns a deep copy of this accumulator.
        """
        pass

    @abc.abstractmethod
    def tick_descent(self):
        """Signal that the accumulator has descended one level deeper into the chain.

        This is called upon entry into an OutputPort's backprop method.
        """
        pass

    @abc.abstractmethod
    def __add__(self, other):
        """Returns a copy of this accumulator with the merged contents of another
        accumulator.
        """
        pass

    @abc.abstractmethod
    def __mul__(self, A):
        """Returns a copy of this accumulator with the contents right-multiplied
        by the specified object.

        Typically this operator will be used to chain backprop gradients, ie.
        do_dx = do_dy * dy_dx, where the accumulator stores do_dy.
        """
        pass

    @abc.abstractmethod
    def __rmul__(self, A):
        """Returns a copy of this accumulator with the contents right-multiplied
        by the specified object.

        Typically this operator will be used to scale backprop gradients, ie.
        scaled = 2 * do_dx, where the accumulator stores do_dx.
        """
        pass

    @abc.abstractmethod
    def retrieve(self):
        """Returns this accumulator's contents as a single gradient/Jacobian matrix.
        """
        pass


class TruncatedBackprop(BackpropInterface):
    """Tracks a list of gradients and expires them after they exceed their lifespan
    in descents (max depth truncation).

    Parameters
    ----------
    do_dx     : iterable convertable to numpy ND-array [default None]
        The initial accumulated value(s)
    max_depth : Positive numeric or iterable convertable to numpy ND-array [default None]
        Descents after which the respective gradient expires and is removed
    """
    def __init__(self, do_dx=None, depths=None):
        # Fast return for operators
        # NOTE No properties are assigned to default values here!
        if do_dx is None and depths is None:
            return

        dodx_only = do_dx is None and depths is not None
        depths_only = do_dx is not None and depths is None
        if dodx_only or depths_only:
            raise ValueError('Cannot specify only one of do_dx and depths!')

        do_dx = np.array(do_dx, copy=True)
        if len(do_dx.shape) == 2:
            self._accs = np.expand_dims(do_dx, axis=0)
        elif len(do_dx.shape) == 3:
            self._accs = do_dx
        else:
            raise ValueError('Could not interpret do_dx.')

        self._depths = np.atleast_1d(depths).copy()

        if len(self._accs) != len(self._depths):
            raise ValueError('do_dx has len %d while depths has len %d!'
                             % (len(self._accs), len(self._depths)))

    def __len__(self):
        if self._accs is None:
            return 0
        else:
            return len(self._accs)

    def copy(self):
        ret = TruncatedBackprop()
        ret._accs = self._accs
        ret._depths = self._depths
        return ret

    def clear(self):
        self._accs = None
        self._depths = None

    def __add__(self, other):
        if not isinstance(other, TruncatedBackprop):
            raise NotImplementedError('Can only add TruncatedBackprops together!')

        ret = TruncatedBackprop()
        if self._depths is None or self._accs is None:
            ret._depths = other._depths
            ret._accs = other._accs
        elif other._depths is None or other._accs is None:
            ret._depths = self._depths
            ret._accs = self._accs
        else:
            ret._depths = np.concatenate((self._depths, other._depths), axis=0)
            ret._accs = np.concatenate((self._accs, other._accs), axis=0)
        return ret

    def __mul__(self, A):
        ret = TruncatedBackprop()
        if len(self) > 0:
            # np.dot will raise NotImplemented as appropriate
            ret._accs = np.dot(self._accs, A)
            ret._depths = self._depths
        return ret

    def __rmul__(self, A):
        ret = TruncatedBackprop()
        if len(self) > 0:
            # np.dot will raise NotImplemented as appropriate
            ret._accs = np.rollaxis(np.dot(A, self._accs), axis=1)
            ret._depths = self._depths
        return ret

    def tick_descent(self):
        if len(self) == 0:
            return

        self._depths = self._depths - 1
        not_expired = self._depths > 0
        self._depths = self._depths[not_expired]
        self._accs = self._accs[not_expired]

        # Catch for empty array
        if len(self._depths) == 0:
            self._depths = None
            self._accs = None

    def retrieve(self):
        if len(self) == 0:
            return 0
        else:
            return np.sum(self._accs, axis=0)

class AccumulatedBackprop():
    """Accumualtes gradients without truncation.

    Parameters
    ----------
    do_dx : iterable convertable to numpy ND-array [default None]
        The initial accumulated value(s)
    """
    def __init__(self, do_dx=None):
        if do_dx is None:
            self._acc = 0
        else:
            self._acc = np.atleast_2d(do_dx)

    def __len__(self):
        return 1

    def copy(self):
        return AccumulatedBackprop(do_dx=self._acc)

    def clear(self):
        self._acc = 0

    def __add__(self, other):
        if not isinstance(other, AccumulatedBackprop):
            raise NotImplementedError('Can only add AccumulatedBackprops together!')

        return AccumulatedBackprop(do_dx=self._acc + other._acc)

    def __mul__(self, A):
        return AccumulatedBackprop(do_dx=np.dot(self._acc, A))

    def __rmul__(self, A):
        return AccumulatedBackprop(do_dx=np.dot(A, self._acc))

    def tick_descent(self):
        pass

    def retrieve(self):
        return self._acc