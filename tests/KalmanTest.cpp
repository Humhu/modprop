#include "modprop/kalman/PredictModule.h"
#include "modprop/kalman/UpdateModule.h"
#include "modprop/compo/BasicModules.h"
#include "modprop/optim/GaussianLogLikelihood.h"

using namespace argus;

void TestPredict()
{
	unsigned int state_dim = 3;

	ConstantModule xInit( VectorType::Zero( state_dim ) );
	ConstantModule PInit( MatrixType::Identity( state_dim, state_dim ) );
	ConstantModule QInit( MatrixType::Identity( state_dim, state_dim ) );

	SinkModule xOut, POut;

	KalmanPredictModule predMod;
	predMod.SetLinearParams( MatrixType::Identity( state_dim, state_dim ) );
	link_ports( predMod.GetXIn(), xInit.GetOutput() );
	link_ports( predMod.GetPIn(), PInit.GetOutput() );
	link_ports( predMod.GetQIn(), QInit.GetOutput() );
	link_ports( xOut.GetInput(), predMod.GetXOut() );
	link_ports( POut.GetInput(), predMod.GetPOut() );

	xInit.Foreprop();
	PInit.Foreprop();
	QInit.Foreprop();

	std::cout << "pred xout: " << xOut.GetValue() << std::endl;
	std::cout << "pred Pout: " << POut.GetValue() << std::endl;

	xOut.Backprop( MatrixType::Ones( 1, state_dim ) );
	POut.Backprop( MatrixType::Ones( 1, state_dim * state_dim ) );

	std::cout << "xinit backprop: " << xInit.GetBackpropValue() << std::endl;
	std::cout << "Pinit backprop: " << PInit.GetBackpropValue() << std::endl;
	std::cout << "Qinit backprop: " << QInit.GetBackpropValue() << std::endl;

	xInit.Invalidate();
	PInit.Invalidate();
	QInit.Invalidate();
}

void TestUpdate()
{
	unsigned int state_dim = 3;
	unsigned int obs_dim = 2;

	ConstantModule xInit( VectorType::Zero( state_dim ) );
	ConstantModule PInit( MatrixType::Identity( state_dim, state_dim ) );
	ConstantModule RInit( MatrixType::Identity( obs_dim, obs_dim ) );
	SinkModule xOut, POut;

	KalmanUpdateModule upMod;
	MatrixType C = MatrixType::Random( 2, 3 );
	MatrixType y = MatrixType::Ones( obs_dim, 1 );
	upMod.SetLinearParams( C, y );

	link_ports( upMod.GetXIn(), xInit.GetOutput() );
	link_ports( upMod.GetPIn(), PInit.GetOutput() );
	link_ports( upMod.GetRIn(), RInit.GetOutput() );
	link_ports( xOut.GetInput(), upMod.GetXOut() );
	link_ports( POut.GetInput(), upMod.GetPOut() );

	xInit.Foreprop();
	PInit.Foreprop();
	RInit.Foreprop();

	std::cout << "up xout: " << xOut.GetValue() << std::endl;
	std::cout << "up Pout: " << POut.GetValue() << std::endl;

	xOut.Backprop( MatrixType::Ones( 1, state_dim ) );
	POut.Backprop( MatrixType::Ones( 1, state_dim * state_dim ) );
	std::cout << "xinit backprop: " << xInit.GetBackpropValue() << std::endl;
	std::cout << "Pinit backprop: " << PInit.GetBackpropValue() << std::endl;
	std::cout << "RInit backprop: " << RInit.GetBackpropValue() << std::endl;

	xInit.Invalidate();
	PInit.Invalidate();
	RInit.Invalidate();
}

void TestLikelihood()
{
	unsigned int x_dim = 3;

	ConstantModule sample( VectorType::Random( x_dim ) );
	ConstantModule cov( MatrixType::Random( x_dim, x_dim ) );
	SinkModule llOut;

	GaussianLogLikelihood gll;
	link_ports( gll.GetXIn(), sample.GetOutput() );
	link_ports( gll.GetSIn(), cov.GetOutput() );
	link_ports( llOut.GetInput(), gll.GetLLOut() );

	sample.Foreprop();
	cov.Foreprop();

	std::cout << "gll: " << llOut.GetValue() << std::endl;
}

int main( int argc, char** argv )
{
	TestPredict();
	TestUpdate();
	TestLikelihood();
}