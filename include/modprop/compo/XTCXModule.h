#pragma once

#include "modprop/compo/ModulesCore.h"

namespace argus
{

class XTCXModule
	: public ModuleBase
{
public:

	XTCXModule();

	void Foreprop();
	void Backprop();

	InputPort& GetXIn();
	InputPort& GetCIn();
	OutputPort& GetSOut();

private:

	InputPort _XIn;
	InputPort _CIn;
	OutputPort _SOut;
};

class InnerXTCXModule
: public ModuleBase
{
public:

	InnerXTCXModule();

	void SetX( const MatrixType& X );

	void Foreprop();
	void Backprop();

	InputPort& GetCIn();
	OutputPort& GetSOut();

private:

	MatrixType _X;
	InputPort _CIn;
	OutputPort _SOut;
};

}