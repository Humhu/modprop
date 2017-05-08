#pragma once

#include "modprop/compo/ModulesCore.h"
#include "modprop/compo/BasicModules.h"

#include <deque>

namespace argus
{

class Pipeline
{
public:

	Pipeline();
	virtual ~Pipeline();

	// Produce output
	VectorType GetOutput() const;
	MatrixType GetDerivative() const;
	
	void Foreprop();
	void Backprop();
	void Invalidate();
	
	VectorType GetParams() const;
	void SetParams( const VectorType& p );
	unsigned int ParamDim() const;

protected:

	void RegisterInput( InputPort& in, const MatrixType& init );
	void RegisterOutput( OutputPort& out );

private:

	std::deque<ConstantModule> _params;
	std::deque<SinkModule> _outputs;
};

/*! \brief Tests a Pipeline object. */
void test_derivatives( Pipeline& pipe, double stepSize, double eps );

}