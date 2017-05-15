#include "modprop/compo/XTCXModule.h"
#include "modprop/kalman/UpdateModule.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace argus
{
XTCXModule::XTCXModule()
	: _XIn( *this ), _CIn( *this ), _SOut( *this )
{
	RegisterInput( &_XIn );
	RegisterInput( &_CIn );
	RegisterOutput( &_SOut );
}

void XTCXModule::Foreprop()
{
	const MatrixType& X = _XIn.GetValue();
	const MatrixType& C = _CIn.GetValue();

	MatrixType S = X.transpose() * C * X;
	_SOut.Foreprop( S );
}

void XTCXModule::Backprop()
{
	const MatrixType& X = _XIn.GetValue();
	const MatrixType& C = _CIn.GetValue();

	unsigned int n = X.rows();
	MatrixType Inn = MatrixType::Identity( n * n, n * n );
	MatrixType In = MatrixType::Identity( n, n );
	MatrixType Tnn = gen_transpose_matrix( n, n );
	// NOTE Form assuming C is symmetric
	// MatrixType dS_dL = Inn + gen_transpose_matrix( n, n ) *
	//                    Eigen::kroneckerProduct( In, X.transpose() * C );
	MatrixType dS_dL = Eigen::kroneckerProduct( In, X.transpose() * C )
	+ Tnn * Eigen::kroneckerProduct( In, X.transpose() * C.transpose() );
	MatrixType do_dL = _SOut.ChainBackprop( dS_dL );
	_XIn.Backprop( do_dL );


	MatrixType dS_dC = Eigen::kroneckerProduct( X.transpose(), X.transpose() );
	MatrixType do_dC = _SOut.ChainBackprop( dS_dC );
	_CIn.Backprop( do_dC );
}

InputPort& XTCXModule::GetXIn() { return _XIn; }
InputPort& XTCXModule::GetCIn() { return _CIn; }
OutputPort& XTCXModule::GetSOut() { return _SOut; }

InnerXTCXModule::InnerXTCXModule()
	: _CIn( *this ), _SOut( *this )
{
	RegisterInput( &_CIn );
	RegisterOutput( &_SOut );
}

void InnerXTCXModule::SetX( const MatrixType& X )
{
	_X = X;
}

void InnerXTCXModule::Foreprop()
{
	const MatrixType& C = _CIn.GetValue();

	MatrixType S = _X.transpose() * C * _X;
	_SOut.Foreprop( S );
}

void InnerXTCXModule::Backprop()
{
	MatrixType dS_dC = Eigen::kroneckerProduct( _X.transpose(), _X.transpose() );
	MatrixType do_dC = _SOut.ChainBackprop( dS_dC );
	_CIn.Backprop( do_dC );
}

InputPort& InnerXTCXModule::GetCIn() { return _CIn; }
OutputPort& InnerXTCXModule::GetSOut() { return _SOut; }
}