#pragma once

#include "modprop/compo/ModulesCore.h"
#include <memory>

namespace argus
{

class MeanModule
: public ModuleBase
{
public:

	MeanModule();

	void Foreprop();
	void Backprop(); // NOTE Assumes no new inputs are added in between call to fore/backprop

	void RegisterSource( OutputPort& out );
	void UnregisterSource( OutputPort& out );

	OutputPort& GetOutput();

private:

	std::vector<OutputPort*> _outputRecords;
	std::vector<std::shared_ptr<InputPort>> _inputs;
	OutputPort _output;
};
	
}