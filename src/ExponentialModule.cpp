#include "modprop/compo/ExponentialModule.h"

namespace argus
{
ExponentialModule::ExponentialModule()
	: _input( *this ), _output( *this )
{
	RegisterInput( &_input );
	RegisterOutput( &_output );
}

void ExponentialModule::Foreprop()
{
	const MatrixType& in = _input.GetValue();
	MatrixType out = in.array().exp().matrix();
	_output.Foreprop( out );
}

void ExponentialModule::Backprop()
{
	const MatrixType& val = _output.GetValue();
	Eigen::Map<const VectorType> vec( val.data(), val.size(), 1 );
	MatrixType dS_dx( vec.asDiagonal() );
	MatrixType do_dx = _output.ChainBackprop( dS_dx );
	_input.Backprop( do_dx );
}

InputPort& ExponentialModule::GetInput() { return _input; }
OutputPort& ExponentialModule::GetOutput() { return _output; }
}