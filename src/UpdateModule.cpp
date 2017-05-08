#include "modprop/kalman/UpdateModule.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace argus
{
MatrixType gen_transpose_matrix( size_t m, size_t n )
{
	size_t d = m * n;

	MatrixType inds( m, n );
	for( unsigned int i = 0; i < d; ++i )
	{
		inds( i ) = (i);
	}
	inds.transposeInPlace();

	MatrixType T = MatrixType::Zero( d, d );
	for( unsigned int i = 0; i < d; ++i )
	{
		T( i, inds( i ) ) = 1;
	}
	return T;
}

MatrixType llt_solve_right( const Eigen::LLT<MatrixType>& llt,
                            const MatrixType& b )
{
	return llt.solve( b.transpose() ).transpose();
}

UpdateModule::UpdateModule()
	: _RIn( *this ), _vOut( *this ), _SOut( *this )
{
	RegisterInput( &_RIn );
	RegisterOutput( &_vOut );
	RegisterOutput( &_SOut );
}

void UpdateModule::SetLinearParams( const MatrixType& C,
                                          const VectorType& y )
{
	_C = C;
	_y = y;
	_x0 = VectorType::Zero( C.cols() );
	_y0 = VectorType::Zero( C.rows() );
	Invalidate();
}

void UpdateModule::SetNonlinearParams( const MatrixType& G,
                                             const VectorType& y,
											 const VectorType& x0,
											 const VectorType& y0 )
{
	_C = G;
	_y = y;
	_x0 = x0;
	_y0 = y0;
	Invalidate();
}

VectorType UpdateModule::LinpointDelta() const
{
	const MatrixType& xIn = _xIn.GetValue();
	Eigen::Map<const VectorType> xVec( xIn.data(), xIn.size(), 1 );

	return xVec - _x0;
}

void UpdateModule::Foreprop()
{
	CheckParams();

	const MatrixType& xIn = _xIn.GetValue();
	const MatrixType& PIn = _PIn.GetValue();
	const MatrixType& RIn = _RIn.GetValue();
	Eigen::Map<const VectorType> xVec( xIn.data(), xIn.size(), 1 );

	VectorType yhat = _C * (xVec - _x0) + _y0;
	VectorType v = _y - yhat;
	MatrixType S = _C * PIn * _C.transpose() + RIn;
	_SChol.compute( S );
	_K = llt_solve_right( _SChol, PIn * _C.transpose() );

	MatrixType nextX = xIn + _K * v;
	MatrixType nextP = PIn - _K * _C * PIn;

	_xOut.Foreprop( nextX );
	_POut.Foreprop( nextP );
	_vOut.Foreprop( v );
	_SOut.Foreprop( S );
}

void UpdateModule::Backprop()
{
	MatrixType do_dxin_x, do_dPin_x, do_dR_x;
	MatrixType do_dPin_P, do_dRin_P;
	MatrixType do_dxin_v;
	MatrixType do_dPin_S, do_dRin_S;

	BackpropXOut( do_dxin_x, do_dPin_x, do_dR_x );
	BackpropPOut( do_dPin_P, do_dRin_P );
	BackpropVOut( do_dxin_v );
	BackpropSOut( do_dPin_S, do_dRin_S );

	_xIn.Backprop( sum_matrices( {do_dxin_x, do_dxin_v} ) );
	_PIn.Backprop( sum_matrices( {do_dPin_x, do_dPin_P, do_dPin_S} ) );
	_RIn.Backprop( sum_matrices( {do_dR_x, do_dRin_P, do_dRin_S} ) );
}

InputPort& UpdateModule::GetRIn() { return _RIn; }
OutputPort& UpdateModule::GetVOut() { return _vOut; }
OutputPort& UpdateModule::GetSOut() { return _SOut; }

void UpdateModule::CheckParams()
{
	if( _C.size() == 0 || _y.size() == 0 )
	{
		throw std::runtime_error( "Did not set all update parameters!" );
	}
}

void UpdateModule::BackpropXOut( MatrixType& do_dxin,
                                 MatrixType& do_dPin,
                                 MatrixType& do_dR )
{
	const MatrixType& xIn = _xIn.GetValue();
	const MatrixType& vOut = _vOut.GetValue();
	size_t N = xIn.size();

	MatrixType dxout_dxin = MatrixType::Identity( N, N ) - _K * _C;
	do_dxin = _xOut.ChainBackprop( dxout_dxin );

	MatrixType Sv = _SChol.solve( vOut );
	MatrixType CTSv = _C.transpose() * Sv;
	MatrixType KC = _K * _C;
	MatrixType dxout_dPin = Eigen::kroneckerProduct( CTSv.transpose(),
	                                                 MatrixType::Identity( N, N ) )
	                        - Eigen::kroneckerProduct( CTSv.transpose(), KC );
	do_dPin = _xOut.ChainBackprop( dxout_dPin );

	MatrixType dxout_dR = -Eigen::kroneckerProduct( Sv.transpose(), _K );
	do_dR = _xOut.ChainBackprop( dxout_dR );
}

void UpdateModule::BackpropPOut( MatrixType& do_dPin,
                                       MatrixType& do_dRin )
{
	size_t N = _xIn.GetValue().size();

	MatrixType KC = _K * _C;
	MatrixType I = MatrixType::Identity( N, N );
	MatrixType II = MatrixType::Identity( N * N, N * N );
	MatrixType T = gen_transpose_matrix( N, N );

	MatrixType dPout_dPin = II
	                        - (II + T) * Eigen::kroneckerProduct( I, KC )
	                        + Eigen::kroneckerProduct( KC, KC );
	do_dPin = _POut.ChainBackprop( dPout_dPin );

	MatrixType dPout_dRin = Eigen::kroneckerProduct( _K, _K );
	do_dRin = _POut.ChainBackprop( dPout_dRin );
}

void UpdateModule::BackpropVOut( MatrixType& do_dxin )
{
	do_dxin = _vOut.ChainBackprop( -_C );
}

void UpdateModule::BackpropSOut( MatrixType& do_dPin,
                                       MatrixType& do_dRin )
{
	MatrixType dSout_dPin = Eigen::kroneckerProduct( _C, _C );
	do_dPin = _SOut.ChainBackprop( dSout_dPin );

	do_dRin = _SOut.GetBackpropValue();
}
}