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

	// TODO Have the module destructor clean up by unregistering all the ports
	virtual ~ModuleBase();

	void RegisterInput( InputPort* in );
	void RegisterOutput( OutputPort* out );
	void UnregisterInput( InputPort* in );
	void UnregisterOutput( OutputPort* out );
	
	virtual void UnregisterAllSources( bool recurse = true );
	virtual void UnregisterAllConsumers( bool recurse = true );

	virtual void Foreprop() = 0;
	virtual void Backprop() = 0;

	bool FullyValid() const;
	bool FullyInvalid() const;
	bool BackpropReady();

	void Invalidate();

private:

	// Moving would invalidate all port pointers, so we make it illegal
	ModuleBase( const ModuleBase& other );
	ModuleBase& operator=( const ModuleBase& other );

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
	void UnregisterSource( bool recurse = true );

	void Foreprop( const MatrixType& val );
	void Backprop( const MatrixType& dodx );

	const MatrixType& GetValue() const;

private:

	InputPort( const InputPort& other );
	InputPort& operator=( const InputPort& other );

	ModuleBase& _module;
	bool _valid;
	OutputPort* _source;

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
	void UnregisterConsumer( InputPort* in, bool recurse = true );
	void UnregisterAllConsumers( bool recurse = true );

	void Foreprop( const MatrixType& val );
	void Backprop( const MatrixType& dodx );
	bool BackpropReady() const;
	MatrixType ChainBackprop( const MatrixType& dydx = MatrixType() );
	const MatrixType& GetBackpropValue() const;
	const MatrixType& GetValue() const;

private:

	OutputPort( const OutputPort& other );
	OutputPort& operator=( const OutputPort& other );

	ModuleBase& _module;
	std::vector<InputPort*> _consumers;

	bool _valid;
	MatrixType _value;
	MatrixType _backpropAcc;
	size_t _numBacks;
};

void link_ports( OutputPort& out, InputPort& in );
void unlink_ports( OutputPort& out, InputPort& in );

MatrixType sum_matrices( const std::vector<MatrixType>& mats );
}