#include "modprop/compo/ReshapeModule.h"

#include <iostream>

namespace argus
{
inline unsigned int ravel_inds( unsigned int i, unsigned int j,
                                unsigned int rows )
{
	return i + j * rows;
}

std::vector<IndPair> gen_vec_to_diag_inds( unsigned int N )
{
	std::vector<IndPair> inds;
	inds.reserve( N );
	for( unsigned int i = 0; i < N; ++i )
	{
		inds.emplace_back( i, ravel_inds( i, i, N ) );
	}
	return inds;
}

std::vector<IndPair> gen_dense_to_diag_inds( unsigned int N )
{
	std::vector<IndPair> inds;
	inds.reserve( N );
	for( unsigned int i = 0; i < N; ++i )
	{
		inds.emplace_back( ravel_inds( i, i, N ), ravel_inds( i, i, N ) );
	}
	return inds;
}

std::vector<IndPair> gen_trilc_inds( unsigned int N, unsigned int d )
{
	std::vector<IndPair> inds;
	for( unsigned int j = 0; j < N - d; ++j )
	{
		for( unsigned int i = j + d; i < N; ++i )
		{
			inds.emplace_back( inds.size(), ravel_inds( i, j, N ) );
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
                                    const std::vector<IndPair>& inds )
{
	_baseOut = baseOut;
	_inds = inds;
}

void ReshapeModule::GetShapeParams( MatrixType& baseOut,
                                    std::vector<IndPair>& inds ) const
{
	baseOut = _baseOut;
	inds = _inds;
}

void ReshapeModule::Foreprop()
{
	const MatrixType& l = _input.GetValue();

	MatrixType L = _baseOut;
	for( unsigned int i = 0; i < _inds.size(); ++i )
	{
		// if( _inds[i].first() >= l.size() )
		// {
		//  throw std::runtime_error( "Incorrect reshape input size" );
		// }
		L( _inds[i].second ) = l( _inds[i].first );
	}

	_output.Foreprop( L );
}

void ReshapeModule::Backprop()
{
	unsigned int inSize = _input.GetValue().size();
	MatrixType dL_dl = MatrixType::Zero( _baseOut.size(), inSize );
	for( unsigned int i = 0; i < _inds.size(); ++i )
	{
		dL_dl( _inds[i].second, _inds[i].first ) = 1;
	}
	MatrixType do_dl = _output.ChainBackprop( dL_dl );
	_input.Backprop( do_dl );
}

InputPort& ReshapeModule::GetInput() { return _input; }
OutputPort& ReshapeModule::GetOutput() { return _output; }
}