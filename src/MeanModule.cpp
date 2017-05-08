#include "modprop/optim/MeanModule.h"

namespace argus
{
MeanModule::MeanModule()
	: _output( *this )
{
	RegisterOutput( &_output );
}

void MeanModule::Foreprop()
{
	if( _inputs.empty() )
	{
		throw std::runtime_error( "Cannot compute mean over no inputs" );
	}

	MatrixType out = _inputs[0]->GetValue();
	for( unsigned int i = 1; i < _inputs.size(); ++i )
	{
		out += _inputs[i]->GetValue();
	}
	out = out / _inputs.size();
	_output.Foreprop( out );
}

void MeanModule::Backprop()
{
	const MatrixType& out = _output.GetValue();
	unsigned int dim = out.size();
	MatrixType dy_dx = MatrixType::Identity( dim, dim ) / _inputs.size();
	MatrixType do_dx = _output.ChainBackprop( dy_dx );
	for( unsigned int i = 0; i < _inputs.size(); ++i )
	{
		_inputs[i]->Backprop( do_dx );
	}
}

void MeanModule::RegisterSource( OutputPort& out )
{
	std::shared_ptr<InputPort> in = std::make_shared<InputPort>( *this );
	_inputs.emplace_back( in );
	_outputRecords.push_back( &out );

	RegisterInput( in.get() );
	link_ports( out, *in );
}

void MeanModule::UnregisterSource( OutputPort& out )
{
	bool succ = false;
	unsigned int ind;
	for( unsigned int i = 0; i < _outputRecords.size(); ++i )
	{
		if( _outputRecords[i] == &out )
		{
			ind = i;
			succ = true;
			break;
		}
	}
	if( !succ )
	{
		throw std::invalid_argument( "Cannot unregister non-registered source" );
	}

	unlink_ports( out, *_inputs[ind] );
	UnregisterInput( _inputs[ind].get() );
	_outputRecords.erase( _outputRecords.begin() + ind );
	_inputs.erase( _inputs.begin() + ind );
}

OutputPort& MeanModule::GetOutput() { return _output; }
}