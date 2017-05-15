#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{

class ScalingModule
: public ModuleBase
{
public:

	ScalingModule();
	void SetForwardScale( double s );
	void SetBackwardScale( double s );

	void Foreprop();
	void Backprop();

	InputPort& GetInput();
	OutputPort& GetOutput();

private:

	double _fS;
	double _bS;
	InputPort _input;
	OutputPort _output;
};

}