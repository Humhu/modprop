#include "modprop/compo/OuterProductModule.h"
#include "modprop/kalman/UpdateModule.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace argus
{

OuterProductModule::OuterProductModule()
: _left( *this ), _right( *this ), _output( *this )
{
	RegisterInput( &_left );
	RegisterInput( &_right );
	RegisterOutput( &_output );
}

void OuterProductModule::Foreprop()
{
	const MatrixType& left = _left.GetValue();
	const MatrixType& right = _right.GetValue();
	Eigen::Map<const VectorType> leftVec( left.data(), left.size(), 1 );
	Eigen::Map<const VectorType> rightVec( right.data(), right.size(), 1 );

	_output.Foreprop( left * right.transpose() );
}

void OuterProductModule::Backprop()
{
	const MatrixType& left = _left.GetValue();
	const MatrixType& right = _right.GetValue();
	Eigen::Map<const VectorType> leftVec( left.data(), left.size(), 1 );
	Eigen::Map<const VectorType> rightVec( right.data(), right.size(), 1 );

	unsigned int n = leftVec.size();
	MatrixType dy_dl = Eigen::kroneckerProduct( right, MatrixType::Identity( n, n ) );
	MatrixType dy_dr = Eigen::kroneckerProduct( MatrixType::Identity( n, n ), left );
	_left.Backprop( _output.ChainBackprop( dy_dl ) );
	_right.Backprop( _output.ChainBackprop( dy_dr ) );
}

InputPort& OuterProductModule::GetLeftIn() { return _left; }
InputPort& OuterProductModule::GetRightIn() { return _right; }
OutputPort& OuterProductModule::GetOutput() { return _output; }

RepOuterProductModule::RepOuterProductModule()
: _input( *this ), _output( *this )
{
	RegisterInput( &_input );
	RegisterOutput( &_output );
}

void RepOuterProductModule::Foreprop()
{
	const MatrixType& in = _input.GetValue();
	Eigen::Map<const VectorType> inVec( in.data(), in.size(), 1 );

	_output.Foreprop( in * in.transpose() );
}

void RepOuterProductModule::Backprop()
{
	const MatrixType& in = _input.GetValue();
	Eigen::Map<const VectorType> inVec( in.data(), in.size(), 1 );

	unsigned int n = inVec.size();
	MatrixType dy_dl = Eigen::kroneckerProduct( inVec, MatrixType::Identity( n, n ) );
	MatrixType dy_dr = Eigen::kroneckerProduct( MatrixType::Identity( n, n ), inVec );
	_input.Backprop( _output.ChainBackprop( dy_dl + dy_dr ) );
}

InputPort& RepOuterProductModule::GetInput() { return _input; }
OutputPort& RepOuterProductModule::GetOutput() { return _output; }

}