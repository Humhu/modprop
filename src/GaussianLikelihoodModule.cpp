#include "modprop/optim/GaussianLikelihoodModule.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace argus
{

GaussianLikelihoodModule::GaussianLikelihoodModule()
	: _xIn( *this ), _SIn( *this ), _llOut( *this ) 
{
	RegisterInput( &_xIn );
	RegisterInput( &_SIn );
	RegisterOutput( &_llOut );
}

/*! \brief Computes the log-likelihood of the input sample using the
	* covariance generated from the input features. */
void GaussianLikelihoodModule::Foreprop()
{
	const MatrixType& xIn = _xIn.GetValue();
	const MatrixType& SIn = _SIn.GetValue();
	Eigen::Map<const VectorType> xVec( xIn.data(), xIn.size(), 1 );
	_cholS.compute( SIn );

	size_t N = xIn.size();
	_SInv = _cholS.solve( MatrixType::Identity( N, N ) );
	_xInv = _cholS.solve( xVec );

	double exponent = xVec.dot( _xInv );
	double logdet = _cholS.vectorD().array().log().sum();
	double logz = N * std::log( 2 * M_PI );
	double logpdf = -0.5 * (logz + logdet + exponent);

	MatrixType out( 1, 1 );
	out << logpdf;
	_llOut.Foreprop( out );
}

void GaussianLikelihoodModule::Backprop()
{
	const MatrixType& xIn = _xIn.GetValue();	
	MatrixType do_dxin = _llOut.ChainBackprop( -_xInv.transpose() );

	Eigen::Map<VectorType> SinvFlat( _SInv.data(), _SInv.size(), 1 );
	
	MatrixType xxT = xIn * xIn.transpose();
	Eigen::Map<VectorType> xxTvec( xxT.data(), xxT.size(), 1 );

	MatrixType tempA = -0.5 * SinvFlat.transpose();
	MatrixType tempB = 0.5 * xxTvec.transpose() * Eigen::kroneckerProduct( _SInv.transpose(), _SInv );

	MatrixType dll_dSin = tempA + tempB;
	MatrixType do_dSin = _llOut.ChainBackprop( dll_dSin );

	_xIn.Backprop( do_dxin );
	_SIn.Backprop( do_dSin );
}

InputPort& GaussianLikelihoodModule::GetXIn() { return _xIn; }
InputPort& GaussianLikelihoodModule::GetSIn() { return _SIn; }
OutputPort& GaussianLikelihoodModule::GetLLOut() { return _llOut; }

}