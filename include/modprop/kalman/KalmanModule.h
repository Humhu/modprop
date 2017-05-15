#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{

class KalmanIn
: virtual public ModuleBase
{
public:

	KalmanIn();
	virtual ~KalmanIn();

	InputPort& GetXIn();
	InputPort& GetPIn();

protected:

	InputPort _xIn;
	InputPort _PIn;
};

class KalmanOut
: virtual public ModuleBase
{
public:
	KalmanOut();
	virtual ~KalmanOut();

	OutputPort& GetXOut();
	OutputPort& GetPOut();

	VectorType GetX() const;
	const MatrixType& GetP() const;

protected:

	OutputPort _xOut;
	OutputPort _POut;
};

void link_kalman_ports( KalmanOut& pre, KalmanIn& post );
void unlink_kalman_ports( KalmanOut& pre, KalmanIn& post );

class KalmanPrior
: public KalmanOut
{
public:

	KalmanPrior();
	KalmanPrior( const MatrixType& x, const MatrixType& P );

	void Foreprop();
	void Backprop();

	void SetX( const MatrixType& x );
	void SetP( const MatrixType& P );

	const MatrixType& GetBackpropX() const;
	const MatrixType& GetBackpropP() const;	

private:

	MatrixType _x;
	MatrixType _P;
};

class KalmanPosterior
: public KalmanIn
{
public:

	KalmanPosterior();

	void Foreprop();
	void Backprop();

	const MatrixType& GetX() const;
	const MatrixType& GetP() const;

	void SetBackpropX( const MatrixType& dodx );
	void SetBackpropP( const MatrixType& dodP );

	void Backprop( const MatrixType& dodx, const MatrixType& dodP );
	void BackpropX( const MatrixType& dodx );
	void BackpropP( const MatrixType& dodP );

private:

	MatrixType _backX;
	MatrixType _backP;
};

class KalmanScalingModule
: public KalmanIn, public KalmanOut
{
public:

	KalmanScalingModule();

	void Foreprop();
	void Backprop();

	void SetXBackwardScale( double s );
	void SetPBackwardScale( double s );

private:

	double _xS;
	double _PS;	
};

}