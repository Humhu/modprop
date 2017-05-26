# modprop
Modprop (modular backpropagation) is a lightweight library for building and using computation graphs. It has pure Python and C++ implementations with similar interfaces, allowing fast prototyping in Python and deployment in C++.

## Dependencies
* Eigen (cpp)
* boost (cpp)
* numpy (Python)

# Usage
Modprop focuses on constructing and connecting *modules* into structures called *graphs*. Each module may have multiple *input* and *output* ports at which connections can be made. An input port can be linked to only one output port, but an output port may be linked to any number of input ports.

## Connecting Ports
Connecting ports is referred to as 'linking'

# Writing Custom Modules
Writing a new modprop module class is straightforward. First, all modules must inherit from ModuleBase, which provides a set of functions to greatly simplify writing derived module classes.

## Input and Output Ports
Input and output ports should be declared as member fields/variables in your module. Ports will require a reference to the owning module upon construction and will additionally need to be registered with the module itself in the constructor.

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
