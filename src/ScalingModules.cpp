#include "modprop/compo/ScalingModules.h"

namespace argus
{

ScalingModule::ScalingModule()
: _fS( 1.0 ), _bS( 1.0 ), _input( *this ), _output( *this )
{
	RegisterInput( &_input );
	RegisterOutput( &_output );
}

void ScalingModule::SetForwardScale( double s ) { _fS = s; }
void ScalingModule::SetBackwardScale( double s ) { _bS = s; }

void ScalingModule::Foreprop()
{
	_output.Foreprop( _fS * _input.GetValue() );
}

void ScalingModule::Backprop()
{
	_input.Backprop( _bS * _output.GetBackpropValue() );
}

InputPort& ScalingModule::GetInput() { return _input; }
OutputPort& ScalingModule::GetOutput() { return _output; }
}