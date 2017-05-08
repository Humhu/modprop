#include "modprop/utils/DerivativeTester.h"
#include "modprop/utils/MatrixUtils.hpp"

#include <boost/foreach.hpp>
#include <iostream>

namespace argus
{
Pipeline::Pipeline() {}

Pipeline::~Pipeline() {}

VectorType Pipeline::GetOutput() const
{
	std::vector<MatrixType> outs;
	BOOST_FOREACH( const SinkModule& out, _outputs )
	{
		outs.push_back( out.GetValue() );
	}
	return flatten_matrices( outs );
}

MatrixType Pipeline::GetDerivative() const
{
	std::vector<MatrixType> accs;
	BOOST_FOREACH( const ConstantModule& mod, _params )
	{
		accs.push_back( mod.GetBackpropValue() );
	}
	return hstack_matrices( accs );
}

void Pipeline::Foreprop()
{
	BOOST_FOREACH( ConstantModule & mod, _params )
	{
		mod.Foreprop();
	}
}

void Pipeline::Backprop()
{
	unsigned int tot = 0;
	BOOST_FOREACH( SinkModule & mod, _outputs )
	{
		tot += mod.GetValue().size();
	}

	unsigned int ind = 0;
	BOOST_FOREACH( SinkModule & mod, _outputs )
	{
		unsigned int n = mod.GetValue().size();
		MatrixType do_dout = MatrixType::Zero( tot, n );
		do_dout.block( ind, 0, n, n ) = MatrixType::Identity( n, n );
		ind += n;
		mod.Backprop( do_dout );
	}
}

void Pipeline::Invalidate()
{
	BOOST_FOREACH( ConstantModule & mod, _params )
	{
		mod.Invalidate();
	}
	BOOST_FOREACH( SinkModule & mod, _outputs )
	{
		mod.Invalidate();
	}
}

VectorType Pipeline::GetParams() const
{
	std::vector<MatrixType> params;
	BOOST_FOREACH( const ConstantModule &m, _params )
	{
		params.push_back( m.GetValue() );
	}
	return flatten_matrices( params );
}

void Pipeline::SetParams( const VectorType& p )
{
	if( p.size() != ParamDim() )
	{
		throw std::invalid_argument( "Incorrect parameter dimension" );
	}

	unsigned int i = 0;
	BOOST_FOREACH( ConstantModule & m, _params )
	{
		Eigen::Map<const MatrixType> param( p.data() + i,
		                                    m.GetValue().rows(),
		                                    m.GetValue().cols() );
		m.SetValue( param );
		i += param.size();
	}
}

unsigned int Pipeline::ParamDim() const
{
	unsigned int n = 0;
	BOOST_FOREACH( const ConstantModule &m, _params )
	{
		n += m.GetValue().size();
	}
	return n;
}

void Pipeline::RegisterInput( InputPort& in, const MatrixType& init )
{
	_params.emplace_back();
	link_ports( in, _params.back().GetOutput() );
	_params.back().SetValue( init );
}

void Pipeline::RegisterOutput( OutputPort& out )
{
	_outputs.emplace_back();
	link_ports( _outputs.back().GetInput(), out );
}

void test_derivatives( Pipeline& pipe, double stepSize, double eps )
{
	VectorType theta0 = pipe.GetParams();
	
	pipe.Invalidate();
	pipe.Foreprop();
	pipe.Backprop();
	VectorType y0 = pipe.GetOutput();
	MatrixType jacobian = pipe.GetDerivative();

	for( unsigned int i = 0; i < theta0.size(); ++i )
	{
		VectorType delta = VectorType::Zero( theta0.size() );
		delta(i) = stepSize;
		VectorType predDelta = jacobian * delta;

		pipe.SetParams( theta0 + delta );
		pipe.Invalidate();
		pipe.Foreprop();
		VectorType trueDelta = pipe.GetOutput() - y0;


		VectorType err = predDelta - trueDelta;
		if( (err.array().abs() > eps ).any() )
		{
			std::cout << "Derivative " << i << " failed test!" << std::endl;
			std::cout << "Pred delta: " << predDelta.transpose() << std::endl;
			std::cout << "True delta: " << trueDelta.transpose() << std::endl;
		}
		else
		{
			std::cout << "Derivative " << i << " passed!" << std::endl;
		}
	}
}

}