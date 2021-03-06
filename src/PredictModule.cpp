#include "modprop/kalman/PredictModule.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>
namespace argus
{
PredictModule::PredictModule()
	: _QIn( *this )
{
	RegisterInput( &_QIn );
}

void PredictModule::SetLinearParams( const MatrixType& A )
{
	_A = A;
	_x0 = VectorType::Zero( _A.rows() );
	_y0 = VectorType::Zero( _A.rows() );
	Invalidate();
}

void PredictModule::SetNonlinearParams( const MatrixType& F,
                                              const VectorType& x0,
                                              const VectorType& y0 )
{
	_A = F;
	_x0 = x0;
	_y0 = y0;
	Invalidate();
}

VectorType PredictModule::LinpointDelta() const
{
	const MatrixType& xIn = _xIn.GetValue();
	Eigen::Map<const VectorType> xVec( xIn.data(), xIn.size(), 1 );

	return xVec - _x0;
}

const MatrixType& PredictModule::GetTransMatrix() const { return _A; }

void PredictModule::Foreprop()
{
	CheckParams();

	const MatrixType& xIn = _xIn.GetValue();
	const MatrixType& PIn = _PIn.GetValue();
	const MatrixType& QIn = _QIn.GetValue();
	Eigen::Map<const VectorType> xVec( xIn.data(), xIn.size(), 1 );

	MatrixType nextX = _A * (xVec - _x0) + _y0;
	MatrixType nextP = _A * PIn * _A.transpose() + QIn;

	_xOut.Foreprop( nextX );
	_POut.Foreprop( nextP );
}

void PredictModule::Backprop()
{
	MatrixType do_dxin, do_dPin, do_dQ;
	BackpropXOut( do_dxin );
	BackpropPOut( do_dPin, do_dQ );

	_xIn.Backprop( do_dxin );
	_PIn.Backprop( do_dPin );
	_QIn.Backprop( do_dQ );
}

InputPort& PredictModule::GetQIn() { return _QIn; }

void PredictModule::CheckParams()
{
	if( _A.size() == 0 )
	{
		throw std::runtime_error( "Did not set all predict parameters!" );
	}
}

void PredictModule::BackpropXOut( MatrixType& do_dxin )
{
	MatrixType dxout_dxin = _A;
	do_dxin = _xOut.ChainBackprop( dxout_dxin );
}

void PredictModule::BackpropPOut( MatrixType& do_dPin,
                                        MatrixType& do_dQ )
{
	MatrixType dPout_dPin = Eigen::kroneckerProduct( _A, _A );
	do_dPin = _POut.ChainBackprop( dPout_dPin );
	do_dQ = _POut.GetBackpropValue();
}
}