#include "modprop/compo/MathModules.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>

namespace argus
{
AdditionModule::AdditionModule()
	: _leftIn( *this ), _rightIn( *this ), _output( *this )
{
	RegisterInput( &_leftIn );
	RegisterInput( &_rightIn );
	RegisterOutput( &_output );
}

void AdditionModule::Foreprop()
{
	const MatrixType& left = _leftIn.GetValue();
	const MatrixType& right = _rightIn.GetValue();
	_output.Foreprop( left + right );
}

void AdditionModule::Backprop()
{
	MatrixType do_dl = _output.ChainBackprop();
	_leftIn.Backprop( do_dl );
	_rightIn.Backprop( do_dl );
}

InputPort& AdditionModule::GetLeftIn() { return _leftIn; }
InputPort& AdditionModule::GetRightIn() { return _rightIn; }
OutputPort& AdditionModule::GetOutput() { return _output; }

SubtractionModule::SubtractionModule()
	: _leftIn( *this ), _rightIn( *this ), _output( *this )
{
	RegisterInput( &_leftIn );
	RegisterInput( &_rightIn );
	RegisterOutput( &_output );
}

void SubtractionModule::Foreprop()
{
	const MatrixType& left = _leftIn.GetValue();
	const MatrixType& right = _rightIn.GetValue();
	_output.Foreprop( left - right );
}

void SubtractionModule::Backprop()
{
	MatrixType do_dl = _output.ChainBackprop();
	_leftIn.Backprop( do_dl );
	_rightIn.Backprop( -do_dl );
}

InputPort& SubtractionModule::GetLeftIn() { return _leftIn; }
InputPort& SubtractionModule::GetRightIn() { return _rightIn; }
OutputPort& SubtractionModule::GetOutput() { return _output; }

ProductModule::ProductModule()
	: _leftIn( *this ), _rightIn( *this ), _output( *this )
{
	RegisterInput( &_leftIn );
	RegisterInput( &_rightIn );
	RegisterOutput( &_output );
}

void ProductModule::Foreprop()
{
	const MatrixType& left = _leftIn.GetValue();
	const MatrixType& right = _rightIn.GetValue();
	_output.Foreprop( left * right );
}

void ProductModule::Backprop()
{
	const MatrixType& left = _leftIn.GetValue();
	const MatrixType& right = _rightIn.GetValue();

	unsigned int n = right.cols();
	unsigned int m = left.rows();

	MatrixType dy_dl = Eigen::kroneckerProduct( right.transpose(), MatrixType::Identity( m, m ) );
	MatrixType dy_dr = Eigen::kroneckerProduct( MatrixType::Identity( n, n ), left );

	_leftIn.Backprop( _output.ChainBackprop( dy_dl ) );
	_rightIn.Backprop( _output.ChainBackprop( dy_dr ) );
}

InputPort& ProductModule::GetLeftIn() { return _leftIn; }
InputPort& ProductModule::GetRightIn() { return _rightIn; }
OutputPort& ProductModule::GetOutput() { return _output; }

ScaleModule::ScaleModule()
: _input( *this ), _output( *this ), _s( 1.0 )
{
	RegisterInput( &_input );
	RegisterOutput( &_output );
}

void ScaleModule::SetScale( double s )
{
	_s = s;
	Invalidate();
}

void ScaleModule::Foreprop()
{
	const MatrixType& in = _input.GetValue();
	_output.Foreprop( _s * in );
}

void ScaleModule::Backprop()
{
	const MatrixType& do_dy = _output.ChainBackprop();
	_input.Backprop( _s * do_dy );
}

InputPort& ScaleModule::GetInput() { return _input; }
OutputPort& ScaleModule::GetOutput() { return _output; }
}