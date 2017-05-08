#include "modprop/compo/ReshapeModule.h"

#include <iostream>

namespace argus
{
inline unsigned int ravel_inds( unsigned int i, unsigned int j,
                                unsigned int rows )
{
	return i + j * rows;
}

std::vector<unsigned int> gen_diag_inds( unsigned int N )
{
	std::vector<unsigned int> inds;
	inds.reserve( N );
	for( unsigned int i = 0; i < N; ++i )
	{
		inds.push_back( ravel_inds( i, i, N ) );
	}
	return inds;
}

std::vector<unsigned int> gen_trilc_inds( unsigned int N, unsigned int d )
{
	std::vector<unsigned int> inds;
	for( unsigned int j = 0; j < N - d; ++j )
	{
		for( unsigned int i = j + d; i < N; ++i )
		{
			inds.push_back( ravel_inds( i, j, N ) );
		}
	}
	return inds;
}

ReshapeModule::ReshapeModule()
	: _input( *this ), _output( *this )
{
	RegisterInput( &_input );
	RegisterOutput( &_output );
}

void ReshapeModule::SetShapeParams( const MatrixType& baseOut,
                                    const std::vector<unsigned int>& inds )
{
	_baseOut = baseOut;
	_inds = inds;
}

void ReshapeModule::Foreprop()
{
	const MatrixType& l = _input.GetValue();
	if( l.size() != _inds.size() )
	{
		throw std::runtime_error( "Incorrect reshape input size" );
	}

	MatrixType L = _baseOut;
	for( unsigned int i = 0; i < _inds.size(); ++i )
	{
		L( _inds[i] ) = l( i );
	}

	_output.Foreprop( L );
}

void ReshapeModule::Backprop()
{
	MatrixType dL_dl = MatrixType::Zero( _baseOut.size(), _inds.size() );
	for( unsigned int i = 0; i < _inds.size(); ++i )
	{
		dL_dl( _inds[i], i ) = 1;
	}
	MatrixType do_dl = _output.ChainBackprop( dL_dl );
	_input.Backprop( do_dl );
}

InputPort& ReshapeModule::GetInput() { return _input; }
OutputPort& ReshapeModule::GetOutput() { return _output; }
}