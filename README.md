# modprop
Modprop (modular backpropagation) is a lightweight library for building and using computation graphs. It has pure Python and C++ implementations with similar interfaces, allowing fast prototyping in Python and deployment in C++.

## Why modprop?
* Lightweight, compact, and intended for integration into other codebases unlike many other alternatives
* Easy to extend with well-defined interfaces, helper functions, and automatic bookkeeping to simplify usage

## Dependencies
* Eigen (cpp)
* boost (cpp)
* numpy (Python)

# Usage
Modprop focuses on constructing and connecting **modules** into structures called **graphs**. Each module may have multiple **input** and **output** ports at which connections can be made. An input port can be linked to only one output port, but an output port may be linked to any number of input ports.

## Connecting and Disconnecting Ports
Connecting ports is referred to as 'linking' in the code. Modules should implement methods to return references to their ports to allow for linking.

In C++ you will call `void link_ports(OutputPort& out, InputPort& in)` to connect ports, and `void unlink_ports(OutputPort& out, InputPort& in)` to disconnect ports. Note that since InputPorts can only have one corresponding OutputPort, linking an already connected InputPort will replace the previously connected OutputPort. Unregistration with the previous OutputPort will happen automatically, so don't worry about that bookkeeping!

Linking in Python is analogous to C++ with `def link_ports(in_port, out_port)`. Unlinking ports is not yet supported in Python.

**NOTE:** Currently modules and ports do not unlink themselves when destructed. This means that you must make sure you unlink modules if you are deleting portions of your graph! In C++ you can make use of the `ModuleBase::UnregisterAllSources()` and `ModuleBase::UnregisterAllConsumers()` functions to remove all incoming and outgoing connections, respectively. This will be superceded in the future by having modules self-unlink upon destruction.

## Specifying Graph Inputs and Outputs
Two special modules are provided to 'cap' graphs:

The **ConstantModule** provides a fixed value output and serves as a graph input. ConstantModule has only one output port and no input ports, and also has a method to retrieve accumulated backpropagation information.

The **SinkModule** provides a single input port and serves as a graph output. SinkModule also provides methods to specify backpropagation starting information.

## Resetting Graph State
Modules and ports contain state to allow for separation between forward and backward passes. This state must be reset when performing new forward or backward passes. Resetting all modules in a graph is referred to as 'invalidating' the graph. 

ConstantModule will automatically begin invalidation if its output value is changed. You can also manually invalidate a graph by calling `iterative_invalidate(any_module)` in Python or the `ModuleBase::Invalidate()` method of any module in the graph in C++. Invalidation will then proceed both forward and backwards from the starting module, which will invalidate the entire connected graph. Calling invalidation on already invalidated modules returns with very little overhead, so feel free to do so if you want to be really sure the graph is invalidated.

## Performing Graph Forward Passes
Computing a forward pass, or 'forepropping', is used to 'execute' the computation graph. It is performed differently in C++ and Python owing to Python's smaller recursion limit. Remember to invalidate the graph before forepropping.

In Python, you will call `iterative_foreprop(starting_module)` on each ConstantModule. If everything works correctly, you can then retrieve your outputs from your SinkModules. If retrieving the output throws an exception, you may be missing some connections in your graph or may have forgotten to foreprop an input (see the debugging section for more details).

In C++ you can simply call the `ModuleBase::Foreprop()` method for each ConstantModule. Note that for extremely deep graphs this may result in a stack overflow due to the recursion. An iterative method like the Python implementation will be provided in the future.

## Performing Graph Backward Passes
Computing a backward pass, or 'backpropping', is used to efficiently compute derivatives in a computation graph. Like forepropping, backpropping has slightly different syntax in C++ versus Python, but both share the same conventions:

Backpropagation relies on computing the derivative of a multivariate output vector with respect to each module's inputs and outputs, and then passing this information backwards in the graph to compute the calculus chain rule derivative. In modprop's convention, the derivatives are represented by a Jacobian matrix with each row corresponding to an output dimension and each column corresponding to an input dimension.

To backprop, you will first need to foreprop the graph. After that, you will set the backprop value at each SinkModule. All SinkModule backprop values should have the same number of rows, representing the overall graph output dimensionality, while the number of columns should match the SinkModule's value's dimensionality. If the values of a SinkModule are irrelevant to the graph output, you can set the backprop value to all zeros. Then once all values are set, call the `backprop` method of each SinkModule and retrieve derivatives at each ConstantModule. 

For efficiency, modules will only backprop once their OutputPorts have received a backprop value from *all of their inputs*. This makes it extremely efficient to backprop on deep, branching graph structures, but also means that if not all SinkModules are backpropped, the backprop pass will not reach the start of the graph! ConstantModules will throw an exception if you try to retrieve their derivatives when they have not been reached in the backprop pass.

# Writing Custom Modules
Writing a new modprop module class is straightforward. First, all modules must inherit from ModuleBase, which provides a set of functions to greatly simplify writing derived module classes.

## Input and Output Ports
Input and output ports should be declared as member fields/variables in your module. Ports will require a reference to the owning module upon construction and will additionally need to be registered with the module itself in the constructor. Note that since ports rely on references to the owning module, and vice versa, assignment/moving of these objects is disallowed.

In C++, you will have to call RegisterInput once for each input and RegisterOutput once for each output:
```cpp
class MyModule
: public ModuleBase
{
public:

  ModuleBase()
  : _myInput( *this ), _myOutput( *this )
  {
    RegisterInput( &_myInput );
    RegisterOutput( &_myOutput );
  }

private:

  InputPort _myInput;
  OutputPort _myOutput;
};
```

Python makes variadic arguments easy so you can call register_inputs with all inputs and register_outputs with all outputs:
```python
class MyModule(ModuleBase):
  def __init__(self):
    self._myInput = InputPort(self)
    self._myOutput = OutputPort(self)
    ModuleBase.register_inputs(self, self._myInput)
    ModuleBase.register_outputs(self, self._myOutput)
```

## Required Methods
All modules must implement a forward-pass **foreprop** method and a backward-pass **backprop** method. 

**Foreprop** is called when all of the module's input ports have received input, and should compute outputs to send out along the module's output ports. 

**Backprop** is called when all of the module's output ports have received backpropagation information from all of their connected input ports, and should compute backpropagation derivatives to send out along the module's input ports.

# TODO
1. Implement unlinking in Python
2. Implement automatic unlinking in ModuleBase destructor
