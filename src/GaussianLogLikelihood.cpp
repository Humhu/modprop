#include "modprop/optim/GaussianLogLikelihood.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace argus
{

GaussianLogLikelihood::GaussianLogLikelihood()
	: _xIn( *this ), _SIn( *this ), _llOut( *this ) 
{
	RegisterInput( &_xIn );
	RegisterInput( &_SIn );
	RegisterOutput( &_llOut );
}

/*! \brief Computes the log-likelihood of the input sample using the
	* covariance generated from the input features. */
void GaussianLogLikelihood::Foreprop()
{
	const MatrixType& xIn = _xIn.GetValue();
	const MatrixType& SIn = _SIn.GetValue();
	Eigen::Map<const VectorType> xVec( xIn.data(), xIn.size(), 1 );
	_cholS.compute( SIn );

	size_t N = xIn.size();
	_SInv = _cholS.solve( MatrixType::Identity( N, N ) );
	_xInv = _cholS.solve( xVec );

	double exponent = xVec.dot( _xInv );
	double logdet = _cholS.vectorD().array().sum();
	double logz = N * std::log( 2 * M_PI );
	double logpdf = -0.5 * (logz + logdet + exponent);

	MatrixType out( 1, 1 );
	out << logpdf;
	_llOut.Foreprop( out );
}

void GaussianLogLikelihood::Backprop()
{
	MatrixType do_dxin = _llOut.ChainBackprop( -_xInv );

	Eigen::Map<VectorType> SinvFlat( _SInv.data(), _SInv.size(), 1 );
	MatrixType dll_dSin = -0.5 * SinvFlat +
							0.5 * Eigen::kroneckerProduct( _xInv, _xInv );
	MatrixType do_dSin = _llOut.ChainBackprop( dll_dSin );

	_xIn.Backprop( do_dxin );
	_SIn.Backprop( do_dSin );
}

InputPort& GaussianLogLikelihood::GetXIn() { return _xIn; }
InputPort& GaussianLogLikelihood::GetSIn() { return _SIn; }
OutputPort& GaussianLogLikelihood::GetLLOut() { return _llOut; }

}