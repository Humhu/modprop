#pragma once

#include "modprop/kalman/KalmanModule.h"

namespace argus
{
class PredictModule
	: public KalmanIn, public KalmanOut
{
public:

	PredictModule();

	// Linear mode
	void SetLinearParams( const MatrixType& A );

	// Nonlinear mode
	void SetNonlinearParams( const MatrixType& F,
	                         const VectorType& x0,
	                         const VectorType& y0 );

	const MatrixType& GetTransMatrix() const;

	VectorType LinpointDelta() const;

	void Foreprop();
	void Backprop();

	InputPort& GetQIn();

private:

	MatrixType _A; // or A in linear mode
	VectorType _x0;
	VectorType _y0;

	InputPort _QIn;

	void CheckParams();
	void BackpropXOut( MatrixType& do_dxin );
	void BackpropPOut( MatrixType& do_dPin, MatrixType& do_dQ );
};
}