#include "modprop/compo/core.hpp"
#include "modprop/kalman/kalman.hpp"

#include "modprop/optim/GaussianLogLikelihood.h"

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

	KalmanPredictModule predMod;
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

		RegisterInput( upMod.GetXIn(), x0 );
		RegisterInput( upMod.GetPIn(), P0 );
		RegisterInput( upMod.GetRIn(), R );
		RegisterOutput( upMod.GetXOut() );
		RegisterOutput( upMod.GetPOut() );
		RegisterOutput( upMod.GetVOut() );
		RegisterOutput( upMod.GetSOut() );
	}

	KalmanUpdateModule upMod;
};

struct LikelihoodPipeline
	: public Pipeline
{
	LikelihoodPipeline( unsigned int dim )
	{
		VectorType sample = VectorType::Random( dim );
		MatrixType cov = random_PD( dim );

		RegisterInput( gll.GetXIn(), sample );
		RegisterInput( gll.GetSIn(), cov );
		RegisterOutput( gll.GetLLOut() );
	}

	GaussianLogLikelihood gll;
};

struct ReshapePipeline
	: public Pipeline
{
	ReshapePipeline( unsigned int dim, unsigned int d )
	{
		std::vector<unsigned int> inds = gen_trilc_inds( dim, d );
		lt.SetShapeParams( dim, dim, inds );
		VectorType l = VectorType::Random( inds.size() );
		RegisterInput( lt.GetInput(), l );
		RegisterOutput( lt.GetOutput() );
	}

	ReshapeModule lt;
};

struct ProductPipeline
	: public Pipeline
{
	ProductPipeline( unsigned int dim )
	{
		MatrixType X = MatrixType::Random( dim, dim );
		MatrixType C = random_PD( dim );
		RegisterInput( xmod.GetXIn(), X );
		RegisterInput( xmod.GetCIn(), C );
		RegisterOutput( xmod.GetSOut() );
	}

	XTCXModule xmod;
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

	ProductPipeline xp( 2 );
	std::cout << "Testing product derivatives..." << std::endl;
	test_derivatives( xp, 1E-6, 1E-7 );

	ExponentialPipeline ep( 3 );
	std::cout << "Testing exponential derivatives..." << std::endl;
	test_derivatives( ep, 1E-6, 1E-7 );
}