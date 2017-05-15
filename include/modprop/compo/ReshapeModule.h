#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{
typedef std::pair<unsigned int, unsigned int> IndPair;

// Functions to generate single column-major indices
std::vector<IndPair>
gen_vec_to_diag_inds( unsigned int N );

std::vector<IndPair>
gen_dense_to_diag_inds( unsigned int N );

// TODO Allow for above-diagonal d?
std::vector<IndPair>
gen_trilc_inds( unsigned int N, unsigned int d );

class ReshapeModule
	: public ModuleBase
{
public:

	ReshapeModule();

	void SetShapeParams( const MatrixType& baseOut,
	                     const std::vector<IndPair>& inds );
	void GetShapeParams( MatrixType& baseOut,
	                     std::vector<IndPair>& inds ) const;

	void Foreprop();
	void Backprop();

	InputPort& GetInput();
	OutputPort& GetOutput();

private:

	InputPort _input;
	OutputPort _output;

	MatrixType _baseOut;
	std::vector<IndPair> _inds;
};
}