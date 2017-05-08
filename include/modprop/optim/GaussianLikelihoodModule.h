#pragma once

#include "modprop/compo/ModulesCore.h"
#include <boost/foreach.hpp>
#include <Eigen/Cholesky>
#include <cmath>
#include <iostream>

namespace argus
{

/*! \brief Represents a cost function calculated as the log likelihood of a
 * set of samples drawn from a zero mean Gaussian with the inverse covariance
 * specified per sample by a MatrixBase. Returns negative log-likelihood since
 * it is supposed to be a cost. */
class GaussianLikelihoodModule
	: public ModuleBase
{
public:

	/*! \brief Create a cost representing the log likelihood under the matrix
	 * outputted by the regressor. */
	GaussianLikelihoodModule();

	/*! \brief Computes the log-likelihood of the input sample using the
	 * covariance generated from the input features. */
	void Foreprop();
	void Backprop();

	InputPort& GetXIn();
	InputPort& GetSIn();
	OutputPort& GetLLOut();

private:

	Eigen::LDLT<MatrixType> _cholS;
	VectorType _xInv;
	MatrixType _SInv;

	InputPort _xIn;
	InputPort _SIn;
	OutputPort _llOut;
};
}
