#include "modprop/compo/ModulesCore.h"
#include <iostream>
#include <algorithm>

namespace argus
{
ModuleBase::ModuleBase() {}

ModuleBase::~ModuleBase()
{}

void ModuleBase::RegisterInput( InputPort * in )
{
	_inputs.push_back( in );
}

void ModuleBase::RegisterOutput( OutputPort * out )
{
	_outputs.push_back( out );
}

void ModuleBase::UnregisterInput( InputPort * in )
{
	std::vector<InputPort *>::iterator iter;
	iter = std::find( _inputs.begin(), _inputs.end(), in );
	if( iter == _inputs.end() )
	{
		throw std::runtime_error( "Cannot unregister non-registered input port" );
	}
	_inputs.erase( iter );
}

void ModuleBase::UnregisterOutput( OutputPort * out )
{
	std::vector<OutputPort *>::iterator iter;
	iter = std::find( _outputs.begin(), _outputs.end(), out );
	if( iter == _outputs.end() )
	{
		throw std::runtime_error( "Cannot unregister non-registered output port" );
	}
	_outputs.erase( iter );
}

void ModuleBase::UnregisterAllSources( bool recurse )
{
	BOOST_FOREACH ( InputPort * in, _inputs )
	{
		in->UnregisterSource( recurse );
	}
}

void ModuleBase::UnregisterAllConsumers( bool recurse )
{
	BOOST_FOREACH ( OutputPort * out, _outputs )
	{
		out->UnregisterAllConsumers( recurse );
	}
}

bool ModuleBase::FullyValid() const
{
	BOOST_FOREACH ( InputPort * in, _inputs )
	{
		if( !in->Valid() )
		{
			return false;
		}
	}
	return true;
}

bool ModuleBase::FullyInvalid() const
{
	BOOST_FOREACH ( InputPort * in, _inputs )
	{
		if( in->Valid() )
		{
			return false;
		}
	}
	BOOST_FOREACH ( OutputPort * out, _outputs )
	{
		if( out->Valid() )
		{
			return false;
		}
	}
	return true;
}

bool ModuleBase::BackpropReady()
{
	BOOST_FOREACH ( OutputPort * out, _outputs )
	{
		if( !out->BackpropReady() )
		{
			return false;
		}
	}
	return true;
}

void ModuleBase::Invalidate()
{
	BOOST_FOREACH ( InputPort * in, _inputs )
	{
		in->Invalidate();
	}
	BOOST_FOREACH ( OutputPort * out, _outputs )
	{
		out->Invalidate();
	}
}

std::string ModuleBase::ToString() const
{
	std::stringstream ss;
	ss << "Module:" << std::endl;
	BOOST_FOREACH( InputPort * in, _inputs )
	{
		ss << *in << std::endl;
	}
	BOOST_FOREACH( OutputPort * out, _outputs )
	{
		ss << *out << std::endl;
	}
	return ss.str();
}

std::ostream& operator<<( std::ostream& os, const ModuleBase& mod )
{
	os << mod.ToString();
	return os;
}

InputPort::InputPort( ModuleBase& base )
	: _module( base ), _valid( false ), _source( nullptr )
{}

InputPort::~InputPort()
{
	// UnregisterSource( true );
}

void InputPort::RegisterSource( OutputPort * src )
{
	_source = src;
}

void InputPort::UnregisterSource( bool recurse )
{
	if( _source && recurse )
	{
		_source->UnregisterConsumer( this, false );
	}
	_source = nullptr;
}

bool InputPort::Valid() const
{
	return _valid;
}

void InputPort::Invalidate()
{
	if( !Valid() )
	{
		return;
	}

	_value = MatrixType();
	_valid = false;

	// if( !_module.FullyInvalid() )
	// {
	_module.Invalidate();
	// }
	if( _source ) //&& _source->Valid() )
	{
		_source->Invalidate();
	}
}

void InputPort::Foreprop( const MatrixType& val )
{
	if( _valid )
	{
		throw std::runtime_error( "Already valid input received foreprop" );
	}

	_valid = true;
	_value = val;

	if( _module.FullyValid() )
	{
		_module.Foreprop();
	}
}

void InputPort::Backprop( const MatrixType& dodx )
{
	if( _source )
	{
		_source->Backprop( dodx );
	}
}

const MatrixType& InputPort::GetValue() const
{
	if( !_valid )
	{
		throw std::runtime_error( "Cannot get value from invalid input port" );
	}
	return _value;
}

std::string InputPort::ToString() const
{
	std::stringstream ss;
	ss << "Input port:" << std::endl
	<< "\tSource: " << _source << std::endl
	<< "\tValid: " << _valid << std::endl;
	return ss.str();
}

std::ostream& operator<<( std::ostream& os, const InputPort& in )
{
	os << in.ToString();
	return os;
}

OutputPort::OutputPort( ModuleBase& base )
	: _module( base ), _valid( false ), _numBacks( 0 )
{}

OutputPort::~OutputPort()
{
	// UnregisterAllConsumers( true );
}

bool OutputPort::Valid() const
{
	return _valid;
}

size_t OutputPort::NumConsumers() const
{
	return _consumers.size();
}

void OutputPort::RegisterConsumer( InputPort * in )
{
	if( !in )
	{
		throw std::runtime_error( "Cannot register null consumer!" );
	}
	_consumers.push_back( in );
}

void OutputPort::UnregisterConsumer( InputPort * in, bool recurse )
{
	std::vector<InputPort *>::iterator iter;
	iter = std::find( _consumers.begin(), _consumers.end(), in );
	if( iter == _consumers.end() )
	{
		throw std::invalid_argument( "Cannot unregister non-registered consumer" );
	}
	if( recurse )
	{
		(*iter)->UnregisterSource( false );
	}
	_consumers.erase( iter );
}

void OutputPort::UnregisterAllConsumers( bool recurse )
{
	std::vector<InputPort *> conCopy = _consumers;
	BOOST_FOREACH ( InputPort * con, conCopy )
	{
		UnregisterConsumer( con, recurse );
	}
}

void OutputPort::Invalidate()
{
	if( !Valid() )
	{
		return;
	}

	_backpropAcc = MatrixType();
	_numBacks = 0;
	_value = MatrixType();
	_valid = false;

	// if( !_module.FullyInvalid() )
	// {
	_module.Invalidate();
	// }
	BOOST_FOREACH ( InputPort * con, _consumers )
	{
		// if( con->Valid() )
		// {
		con->Invalidate();
		// }
	}
}

void OutputPort::Foreprop( const MatrixType& val )
{
	_value = val;
	_valid = true;
	BOOST_FOREACH ( InputPort * con, _consumers )
	{
		con->Foreprop( _value );
	}
}

void OutputPort::Backprop( const MatrixType& dodx )
{
	if( !_valid )
	{
		throw std::runtime_error( "Cannot backprop invalid output port!" );
	}

	if( dodx.cols() != _value.size() )
	{
		std::cout << "dodx: " << dodx << std::endl;
		std::cout << "value: " << _value << std::endl;
		throw std::runtime_error( "Output backprop dimension mismatch" );
	}

	if( dodx.size() == 0 )
	{
		throw std::runtime_error( "Received empty backprop value" );
	}

	if( _backpropAcc.size() == 0 )
	{
		_backpropAcc = dodx;
	}
	else
	{
		if( _backpropAcc.rows() != dodx.rows() ||
		    _backpropAcc.cols() != dodx.cols() )
		{
			throw std::runtime_error( "Backprop value size mismatch" );
		}
		_backpropAcc += dodx;
	}

	_numBacks++;

	if( _numBacks > _consumers.size() )
	{
		throw std::runtime_error( "Received more backprops than consumers!" );
	}

	if( BackpropReady() && _module.BackpropReady() )
	{
		_module.Backprop();
	}
}

bool OutputPort::BackpropReady() const
{
	return _numBacks == _consumers.size();
}

MatrixType OutputPort::ChainBackprop( const MatrixType& dydx )
{
	if( _backpropAcc.size() == 0 )
	{
		return MatrixType();
	}

	if( dydx.size() != 0 )
	{
		return _backpropAcc * dydx;
	}
	else
	{
		return _backpropAcc;
	}
}

const MatrixType& OutputPort::GetBackpropValue() const
{
	return _backpropAcc;
}

const MatrixType& OutputPort::GetValue() const
{
	if( !_valid )
	{
		throw std::runtime_error( "Cannot get value from invalid output port" );
	}
	return _value;
}

std::string OutputPort::ToString() const
{
	std::stringstream ss;
	ss << "Output port:" << std::endl
	<< "\tNum consumers: " << NumConsumers() << std::endl
	<< "\tValid: " << _valid << std::endl
	<< "\tNum backprops: " << _numBacks;
	return ss.str();
}

std::ostream& operator<<( std::ostream& os, const OutputPort& out )
{
	os << out.ToString();
	return os;
}

void link_ports( OutputPort& out, InputPort& in )
{
	in.RegisterSource( &out );
	out.RegisterConsumer( &in );
}

void unlink_ports( OutputPort& out, InputPort& in )
{
	in.UnregisterSource( false );
	out.UnregisterConsumer( &in, false );
}

MatrixType sum_matrices( const std::vector<MatrixType>& mats )
{
	if( mats.size() == 0 )
	{
		throw std::runtime_error( "Cannot sum empty vector of matrices" );
	}

	bool init = false;
	MatrixType out;
	for( unsigned int i = 0; i < mats.size(); ++i )
	{
		if( mats[i].size() == 0 )
		{
			continue;
		}
		if( !init )
		{
			out = mats[i];
			init = true;
			continue;
		}
		out += mats[i];
	}

	if( !init )
	{
		throw std::runtime_error( "Received no non-empty matrices" );
	}
	return out;
}
}