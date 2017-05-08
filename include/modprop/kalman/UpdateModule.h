#pragma once

#include <iostream>
#include "modprop/kalman/KalmanModule.h"

namespace argus
{
/*! \brief Generates the vectorized transpose matrix for column-major
 * flattening. */
MatrixType gen_transpose_matrix( size_t m, size_t n );

MatrixType llt_solve_right( const Eigen::LLT<MatrixType>& llt,
                            const MatrixType& b );

/*! \brief Adds two sources together. */
class KalmanUpdateModule
	: public KalmanIn, public KalmanOut
{
public:

	KalmanUpdateModule();

	// Linear mode
	void SetLinearParams( const MatrixType& C, const VectorType& y );

	// Nonlinear mode
	void SetNonlinearParams( const MatrixType& G, const VectorType& y,
	                         const VectorType& x0, const VectorType& y0 );

	VectorType LinpointDelta() const;

	void Foreprop();
	void Backprop();

	InputPort& GetRIn();
	OutputPort& GetVOut();
	OutputPort& GetSOut();

private:

	MatrixType _C;
	VectorType _y;
	VectorType _x0;
	VectorType _y0;

	Eigen::LLT<MatrixType> _SChol;
	MatrixType _K;

	InputPort _RIn;
	OutputPort _vOut;
	OutputPort _SOut;

	void CheckParams();

	void BackpropXOut( MatrixType& do_dxin,
	                   MatrixType& do_dPin,
	                   MatrixType& do_dR );
	void BackpropPOut( MatrixType& do_dPin, MatrixType& do_dRin );
	void BackpropVOut( MatrixType& do_dxin );
	void BackpropSOut( MatrixType& do_dPin, MatrixType& do_dRin );
};
}