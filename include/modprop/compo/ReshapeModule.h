#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{
// Functions to generate single column-major indices
std::vector<unsigned int> gen_diag_inds( unsigned int N );
std::vector<unsigned int> gen_trilc_inds( unsigned int N, unsigned int d );

class ReshapeModule
	: public ModuleBase
{
public:

	ReshapeModule();

	void SetShapeParams( unsigned int outputRows, unsigned int outputCols,
	                     const std::vector<unsigned int>& inds );

	void Foreprop();
	void Backprop();

	InputPort& GetInput();
	OutputPort& GetOutput();

private:

	InputPort _input;
	OutputPort _output;

	unsigned int _outRows;
	unsigned int _outCols;
	std::vector<unsigned int> _inds;
};
}