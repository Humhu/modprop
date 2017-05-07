#include "modprop/compo/BasicModules.h"

namespace argus
{
ConstantModule::ConstantModule( const MatrixType& val )
	: _output( *this ), _value( val )
{
	RegisterOutput( &_output );
}

void ConstantModule::Foreprop()
{
	_output.Foreprop( _value );
}

void ConstantModule::Backprop()
{}

OutputPort& ConstantModule::GetOutput() { return _output; }

const MatrixType& ConstantModule::GetBackpropValue() const
{
	return _output.GetBackpropValue();
}

SinkModule::SinkModule()
	: _input( *this )
{
	RegisterInput( &_input );
}

void SinkModule::Foreprop()
{}

void SinkModule::Backprop( const MatrixType& dodx )
{
	SetBackpropValue( dodx );
	Backprop();
}

void SinkModule::Backprop()
{
	_input.Backprop( _backpropValue );
}

void SinkModule::SetBackpropValue( const MatrixType& dodx )
{
	_backpropValue = dodx;
}

const MatrixType& SinkModule::GetBackpropValue() const
{
	return _backpropValue;
}

const MatrixType& SinkModule::GetValue() const
{
	return _input.GetValue();
}

InputPort& SinkModule::GetInput()
{
	return _input;
}
}