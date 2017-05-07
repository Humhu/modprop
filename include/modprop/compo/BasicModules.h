#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{
class ConstantModule
	: public ModuleBase
{
public:

	ConstantModule( const MatrixType& val = MatrixType() );
	void Foreprop();

	void Backprop();
	OutputPort& GetOutput();
	const MatrixType& GetBackpropValue() const;

private:

	OutputPort _output;
	MatrixType _value;
};

class SinkModule
	: public ModuleBase
{
public:

	SinkModule();

	void Foreprop();

	void Backprop( const MatrixType& dodx );
	void Backprop();

	void SetBackpropValue( const MatrixType& dodx );
	const MatrixType& GetBackpropValue() const;
	const MatrixType& GetValue() const;
	InputPort& GetInput();

private:

	MatrixType _backpropValue;
	InputPort _input;
};
}