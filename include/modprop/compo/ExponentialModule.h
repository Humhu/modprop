#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{

class ExponentialModule
: public ModuleBase
{
public:

	ExponentialModule();

	void Foreprop();
	void Backprop();

	InputPort& GetInput();
	OutputPort& GetOutput();

private:

	InputPort _input;
	OutputPort _output;

};

}