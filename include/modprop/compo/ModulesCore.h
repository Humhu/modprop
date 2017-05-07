#pragma once

#include "modprop/ModpropTypes.h"
#include <boost/foreach.hpp>

namespace argus
{
class InputPort;
class OutputPort;

class ModuleBase
{
public:

	ModuleBase();

	void RegisterInput( InputPort* in );
	void RegisterOutput( OutputPort* out );

	virtual void Foreprop() = 0;
	virtual void Backprop() = 0;

	bool FullyValid() const;
	bool FullyInvalid() const;
	bool BackpropReady();

	void Invalidate();

private:

	std::vector<InputPort*> _inputs;
	std::vector<OutputPort*> _outputs;
};

// NOTE All ports use dynamically-sized matrices to represent both scalars
// and matrices
class InputPort
{
public:

	InputPort( ModuleBase& base );

	bool Valid() const;
	void Invalidate();
	void RegisterSource( OutputPort* src );

	void Foreprop( const MatrixType& val );
	void Backprop( const MatrixType& dodx );

	const MatrixType& GetValue() const;

private:

	ModuleBase& _module;
	OutputPort* _source;

	bool _valid;
	MatrixType _value;
};

class OutputPort
{
public:

	OutputPort( ModuleBase& base );

	bool Valid() const;
	void Invalidate();
	size_t NumConsumers() const;
	void RegisterConsumer( InputPort* in );
	void UnregisterConsumer( InputPort* in );

	void Foreprop( const MatrixType& val );
	void Backprop( const MatrixType& dodx );
	bool BackpropReady() const;
	MatrixType ChainBackprop( const MatrixType& dydx );
	const MatrixType& GetBackpropValue() const;
	const MatrixType& GetValue() const;

private:

	ModuleBase& _module;
	std::vector<InputPort*> _consumers;

	bool _valid;
	MatrixType _value;
	MatrixType _backpropAcc;
	size_t _numBacks;
};

void link_ports( InputPort& in, OutputPort& out );
void unlink_ports( InputPort& in, OutputPort& out );

MatrixType sum_matrices( const std::vector<MatrixType>& mats );
}