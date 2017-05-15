#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{

class AdditionModule
: public ModuleBase
{
public:

	AdditionModule();

	void Foreprop();
	void Backprop();

	InputPort& GetLeftIn();
	InputPort& GetRightIn();
	OutputPort& GetOutput();

private:

	InputPort _leftIn;
	InputPort _rightIn;
	OutputPort _output;
};

class SubtractionModule
: public ModuleBase
{
public:

	SubtractionModule();

	void Foreprop();
	void Backprop();

	InputPort& GetLeftIn();
	InputPort& GetRightIn();
	OutputPort& GetOutput();

private:

	InputPort _leftIn;
	InputPort _rightIn;
	OutputPort _output;
};

class ProductModule
: public ModuleBase
{
public:

	ProductModule();

	void Foreprop();
	void Backprop();

	InputPort& GetLeftIn();
	InputPort& GetRightIn();
	OutputPort& GetOutput();

private:

	InputPort _leftIn;
	InputPort _rightIn;
	OutputPort _output;
};

class ScaleModule
: public ModuleBase
{
public:

	ScaleModule();

	void SetScale( double s );

	void Foreprop();
	void Backprop();

	InputPort& GetInput();
	OutputPort& GetOutput();

private:

	InputPort _input;
	OutputPort _output;
	double _s;
};


}