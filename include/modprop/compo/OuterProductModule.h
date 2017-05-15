#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{

class OuterProductModule
: public ModuleBase
{
public:

	OuterProductModule();

	void Foreprop();
	void Backprop();

	InputPort& GetLeftIn();
	InputPort& GetRightIn();
	OutputPort& GetOutput();

private:

	InputPort _left;
	InputPort _right;
	OutputPort _output;

};

class RepOuterProductModule
: public ModuleBase
{
public:

	RepOuterProductModule();

	void Foreprop();
	void Backprop();

	InputPort& GetInput();
	OutputPort& GetOutput();

private:

	InputPort _input;
	OutputPort _output;

};

}