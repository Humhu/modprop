#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{
class KalmanPredictModule
	: public ModuleBase
{
public:

	KalmanPredictModule();

	// Linear mode
	void SetLinearParams( const MatrixType& A );

	// Nonlinear mode
	void SetNonlinearParams( const MatrixType& F,
	                         const VectorType& x0,
	                         const VectorType& y0 );

	VectorType LinpointDelta() const;

	void Foreprop();
	void Backprop();

	InputPort& GetXIn();
	InputPort& GetPIn();
	InputPort& GetQIn();
	OutputPort& GetXOut();
	OutputPort& GetPOut();

private:

	MatrixType _A; // or A in linear mode
	VectorType _x0;
	VectorType _y0;

	InputPort _xIn;
	InputPort _PIn;
	InputPort _QIn;

	OutputPort _xOut;
	OutputPort _POut;

	void CheckParams();
	void BackpropXOut( MatrixType& do_dxin );
	void BackpropPOut( MatrixType& do_dPin, MatrixType& do_dQ );
};
}