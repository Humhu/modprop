#include "modprop/kalman/KalmanModule.h"

namespace argus
{
KalmanIn::KalmanIn()
	: _xIn( *this ), _PIn( *this )
{
	RegisterInput( &_xIn );
	RegisterInput( &_PIn );
}

KalmanIn::~KalmanIn() {}

InputPort& KalmanIn::GetXIn() { return _xIn; }
InputPort& KalmanIn::GetPIn() { return _PIn; }

KalmanOut::KalmanOut()
	: _xOut( *this ), _POut( *this )
{
	RegisterOutput( &_xOut );
	RegisterOutput( &_POut );
}

KalmanOut::~KalmanOut() {}

OutputPort& KalmanOut::GetXOut() { return _xOut; }
OutputPort& KalmanOut::GetPOut() { return _POut; }

VectorType KalmanOut::GetX() const
{
	const MatrixType& xVal = _xOut.GetValue();
	Eigen::Map<const VectorType> x( xVal.data(), xVal.size(), 1 );
	return x;
}

const MatrixType& KalmanOut::GetP() const
{
	return _POut.GetValue();
}

void link_kalman_ports( KalmanOut& pre, KalmanIn& post )
{
	link_ports( post.GetXIn(), pre.GetXOut() );
	link_ports( post.GetPIn(), pre.GetPOut() );
}

void unlink_kalman_ports( KalmanOut& pre, KalmanIn& post )
{
	link_ports( post.GetXIn(), pre.GetXOut() );
	link_ports( post.GetPIn(), pre.GetPOut() );
}

KalmanPrior::KalmanPrior() {}

KalmanPrior::KalmanPrior( const MatrixType& x, const MatrixType& P )
	: _x( x ), _P( P ) {}

void KalmanPrior::Foreprop()
{
	_xOut.Foreprop( _x );
	_POut.Foreprop( _P );
}

void KalmanPrior::Backprop()
{}

void KalmanPrior::SetX( const MatrixType& x )
{
	_x = x;
	Invalidate();
}

void KalmanPrior::SetP( const MatrixType& P )
{
	_P = P;
	Invalidate();
}

const MatrixType& KalmanPrior::GetBackpropX() const
{
	return _xOut.GetBackpropValue();
}

const MatrixType& KalmanPrior::GetBackpropP() const
{
	return _POut.GetBackpropValue();
}

KalmanPosterior::KalmanPosterior() {}

void KalmanPosterior::Foreprop()
{}

const MatrixType& KalmanPosterior::GetX() const
{
	return _xIn.GetValue();
}

const MatrixType& KalmanPosterior::GetP() const
{
	return _PIn.GetValue();
}

void KalmanPosterior::SetBackpropX( const MatrixType& dodx )
{
	_backX = dodx;
}

void KalmanPosterior::SetBackpropP( const MatrixType& dodP )
{
	_backP = dodP;
}

void KalmanPosterior::Backprop( const MatrixType& dodx,
                                const MatrixType& dodP )
{
	SetBackpropX( dodx );
	SetBackpropP( dodP );
	Backprop();
}

void KalmanPosterior::BackpropX( const MatrixType& dodx )
{
	unsigned int dim = dodx.cols();
	unsigned int nOut = dodx.rows();
	Backprop( dodx, MatrixType::Zero( nOut, dim * dim ) );
}

void KalmanPosterior::BackpropP( const MatrixType& dodP )
{
	unsigned int dim = dodP.cols();
	unsigned int nOut = dodP.rows();
	Backprop( MatrixType::Zero( nOut, dim ), dodP );
}

void KalmanPosterior::Backprop()
{
	_xIn.Backprop( _backX );
	_PIn.Backprop( _backP );
}
}