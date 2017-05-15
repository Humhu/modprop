#include "modprop/compo/core.hpp"
#include "modprop/kalman/kalman.hpp"

#include "modprop/optim/GaussianLikelihoodModule.h"

#include "modprop/utils/DerivativeTester.h"
#include "modprop/utils/MatrixUtils.hpp"

using namespace argus;

MatrixType random_PD( unsigned int n )
{
	MatrixType A = MatrixType::Random( n, n );
	return A * A.transpose();
}

struct PredictPipeline
	: public Pipeline
{
	PredictPipeline( unsigned int state_dim )
	{
		MatrixType x0 = VectorType::Random( state_dim );
		MatrixType P0 = random_PD( state_dim );
		MatrixType Q = random_PD( state_dim );
		MatrixType A = MatrixType::Random( state_dim, state_dim );

		predMod.SetLinearParams( A );

		RegisterInput( predMod.GetXIn(), x0 );
		RegisterInput( predMod.GetPIn(), P0 );
		RegisterInput( predMod.GetQIn(), Q );
		RegisterOutput( predMod.GetXOut() );
		RegisterOutput( predMod.GetPOut() );
	}

	PredictModule predMod;
};

struct UpdatePipeline
	: public Pipeline
{
	UpdatePipeline( unsigned int state_dim, unsigned int obs_dim )
	{
		MatrixType x0 = VectorType::Random( state_dim );
		MatrixType P0 = MatrixType::Identity( state_dim, state_dim );     //random_PD( state_dim );
		MatrixType R = MatrixType::Identity( obs_dim, obs_dim );     //random_PD( obs_dim );
		MatrixType C = MatrixType::Random( obs_dim, state_dim );
		VectorType y = VectorType::Random( obs_dim );

		upMod.SetLinearParams( C, y );

		// Initialize R
		Eigen::LDLT<MatrixType> ldlt( R );
		std::vector<IndPair > trilInds = gen_trilc_inds( obs_dim, 1 );
		VectorType lInit( trilInds.size() );
		MatrixType matL = ldlt.matrixL();
		for( unsigned int i = 0; i < trilInds.size(); ++i )
		{
			lInit( trilInds[i].first ) = matL( trilInds[i].second );
		}
		_RlReshape.SetShapeParams( MatrixType::Identity( obs_dim, obs_dim ), trilInds );

		VectorType dInit = ldlt.vectorD().array().log().matrix();
		_RdReshape.SetShapeParams( MatrixType::Zero( obs_dim, obs_dim ),
		                           gen_vec_to_diag_inds( obs_dim ) );

		link_ports( _RexpD.GetOutput(), _RdReshape.GetInput() );
		link_ports( _RdReshape.GetOutput(), _Rldlt.GetCIn() );
		link_ports( _RlReshape.GetOutput(), _Rldlt.GetXIn() );
		link_ports( _Rldlt.GetSOut(), upMod.GetRIn() );
		RegisterInput( _RexpD.GetInput(), dInit );
		RegisterInput( _RlReshape.GetInput(), lInit );

		// Initialize Pin
		ldlt.compute( P0 );
		trilInds = gen_trilc_inds( obs_dim, 1 );
		lInit = VectorType( trilInds.size() );
		matL = ldlt.matrixL();
		for( unsigned int i = 0; i < trilInds.size(); ++i )
		{
			lInit( trilInds[i].first ) = matL( trilInds[i].second );
		}
		_PlReshape.SetShapeParams( MatrixType::Identity( state_dim, state_dim ), trilInds );

		dInit = ldlt.vectorD().array().log().matrix();
		_PdReshape.SetShapeParams( MatrixType::Zero( state_dim, state_dim ),
		                           gen_vec_to_diag_inds( state_dim ) );

		link_ports( _PexpD.GetOutput(), _PdReshape.GetInput() );
		link_ports( _PdReshape.GetOutput(), _Pldlt.GetCIn() );
		link_ports( _PlReshape.GetOutput(), _Pldlt.GetXIn() );
		link_ports( _Pldlt.GetSOut(), upMod.GetPIn() );
		RegisterInput( _PexpD.GetInput(), dInit );
		RegisterInput( _PlReshape.GetInput(), lInit );

		RegisterInput( upMod.GetXIn(), x0 );
		RegisterOutput( upMod.GetXOut() );
		RegisterOutput( upMod.GetPOut() );
		RegisterOutput( upMod.GetVOut() );
		RegisterOutput( upMod.GetSOut() );
		RegisterOutput( upMod.GetUOut() );
	}

	ExponentialModule _RexpD;
	ReshapeModule _RlReshape;
	ReshapeModule _RdReshape;
	XTCXModule _Rldlt;

	ExponentialModule _PexpD;
	ReshapeModule _PlReshape;
	ReshapeModule _PdReshape;
	XTCXModule _Pldlt;

	UpdateModule upMod;
};

struct LikelihoodPipeline
	: public Pipeline
{
	LikelihoodPipeline( unsigned int dim )
	{
		VectorType sample = VectorType::Random( dim );
		MatrixType cov = random_PD( dim );

		Eigen::LDLT<MatrixType> ldlt( cov );
		std::vector<IndPair > trilInds = gen_trilc_inds( dim, 1 );
		VectorType lInit( trilInds.size() );
		MatrixType matL = ldlt.matrixL();
		for( unsigned int i = 0; i < trilInds.size(); ++i )
		{
			lInit( trilInds[i].first ) = matL( trilInds[i].second );
		}
		_lReshape.SetShapeParams( MatrixType::Identity( dim, dim ), trilInds );

		VectorType dInit = ldlt.vectorD().array().log().matrix();
		_dReshape.SetShapeParams( MatrixType::Zero( dim, dim ),
		                          gen_vec_to_diag_inds( dim ) );

		link_ports( _expD.GetOutput(), _dReshape.GetInput() );
		link_ports( _dReshape.GetOutput(), _ldlt.GetCIn() );
		link_ports( _lReshape.GetOutput(), _ldlt.GetXIn() );
		link_ports( _ldlt.GetSOut(), gll.GetSIn() );

		RegisterInput( _expD.GetInput(), dInit );
		RegisterInput( _lReshape.GetInput(), lInit );
		RegisterInput( gll.GetXIn(), sample );
		RegisterOutput( gll.GetLLOut() );
	}

	ExponentialModule _expD;
	ReshapeModule _lReshape;
	ReshapeModule _dReshape;
	XTCXModule _ldlt;

	GaussianLikelihoodModule gll;
};

struct ReshapePipeline
	: public Pipeline
{
	ReshapePipeline( unsigned int dim, unsigned int d )
	{
		std::vector<IndPair> inds = gen_trilc_inds( dim, d );
		lt.SetShapeParams( MatrixType::Identity( dim, dim ), inds );
		VectorType l = VectorType::Random( inds.size() );
		RegisterInput( lt.GetInput(), l );
		RegisterOutput( lt.GetOutput() );
	}

	ReshapeModule lt;
};

struct QuadraticPipeline
	: public Pipeline
{
	QuadraticPipeline( unsigned int dim )
	{
		MatrixType X = MatrixType::Random( dim, dim );
		MatrixType C = random_PD( dim );
		RegisterInput( xmod.GetXIn(), X );
		RegisterInput( xmod.GetCIn(), C );
		RegisterOutput( xmod.GetSOut() );
	}

	XTCXModule xmod;
};

struct ProductPipeline
	: public Pipeline
{
	ProductPipeline( unsigned int m, unsigned int n )
	{
		MatrixType L = MatrixType::Random( m, n );
		MatrixType R = MatrixType::Random( n, m );
		RegisterInput( prod.GetLeftIn(), L );
		RegisterInput( prod.GetRightIn(), R );
		RegisterOutput( prod.GetOutput() );
	}

	ProductModule prod;
};

struct ExponentialPipeline
	: public Pipeline
{
	ExponentialPipeline( unsigned int dim )
	{
		MatrixType A = MatrixType::Random( dim, dim );
		RegisterInput( emod.GetInput(), A );
		RegisterOutput( emod.GetOutput() );
	}

	ExponentialModule emod;
};

struct OuterProductPipeline
	: public Pipeline
{
	OuterProductPipeline( unsigned int dim )
	{
		VectorType v = VectorType::Random( dim );
		VectorType u = VectorType::Random( dim );
		RegisterInput( op.GetLeftIn(), v );
		RegisterInput( op.GetRightIn(), u );
		RegisterOutput( op.GetOutput() );
	}

	OuterProductModule op;
};

int main( int argc, char** argv )
{
	PredictPipeline pp( 3 );

	std::cout << "Testing predict derivatives..." << std::endl;
	test_derivatives( pp, 1E-6, 1E-7 );

	UpdatePipeline up( 3, 2 );
	std::cout << "Testing update derivatives..." << std::endl;
	test_derivatives( up, 1E-6, 1E-7 );

	LikelihoodPipeline lp( 2 );
	std::cout << "Testing likelihood derivatives..." << std::endl;
	test_derivatives( lp, 1E-6, 1E-7 );

	ReshapePipeline rp( 3, 0 );
	std::cout << "Testing reshape derivatives..." << std::endl;
	test_derivatives( rp, 1E-6, 1E-7 );

	QuadraticPipeline xp( 2 );
	std::cout << "Testing quadratic derivatives..." << std::endl;
	test_derivatives( xp, 1E-6, 1E-7 );

	ExponentialPipeline ep( 3 );
	std::cout << "Testing exponential derivatives..." << std::endl;
	test_derivatives( ep, 1E-6, 1E-7 );

	ProductPipeline prodp( 3, 4 );
	std::cout << "Testing product derivatives..." << std::endl;
	test_derivatives( prodp, 1E-6, 1E-7 );

	OuterProductPipeline opd( 3 );
	std::cout << "Testing outer product pipeline..." << std::endl;
	test_derivatives( opd, 1E-6, 1E-7 );
}