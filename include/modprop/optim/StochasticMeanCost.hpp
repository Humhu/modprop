#pragma once

#include <boost/foreach.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <iostream>

#include "modprop/optim/MeanCost.hpp"
#include "modprop/utils/SubsetSamplers.hpp"

namespace percepto
{

/*! \brief Randomly samples subsets of the population to return
 * estimates of the objective and gradient. 
 *
 * Resampling is executed on calls to Evaluate() only, since adding an
 * explicit Resample() method would break the cost function interface. 
 *
 * The container holding the costs can be altered to change the
 * population. Make sure to call Evaluate() after alteration to set
 * the active indices.
 */
template <typename DataType>
class StochasticMeanCost
: public MeanCost<DataType>
{
public:

	typedef DataType OutputType;
	typedef MeanCost<DataType> ParentCost;

	/*! \brief Creates a cost by averaging costs on a poulation of
	 * inputs. */
	StochasticMeanCost()
	{
		boost::random::random_device rng;
		_generator.seed( rng );
	}

	void SetBatchSize( unsigned int ss ) { _subsetSize = ss; }

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// clock_t start = clock();
		if( _activeInds.size() == 0 ) { return; }
		MatrixType thisDodx = nextDodx / _activeInds.size();
		for( unsigned int i = 0; i < _activeInds.size(); i++ )
		{
			// clock_t starti = clock();
			ParentCost::_sinks[ _activeInds[i] ].Backprop( thisDodx );
			// std::cout << "SMC " << i << ": " << ((double) clock() - starti) / CLOCKS_PER_SEC;
		}
		// std::cout << "SMC total: " << ((double) clock() - start ) / CLOCKS_PER_SEC;
	}

	void Resample()
	{
		RandomSample();
	}

	const std::vector<unsigned int>& GetActiveInds() const
	{
		return _activeInds;
	}

	/*! \brief Calculate the objective function by averaging the 
	 * underlying cost function over the population. Resamples the
	 * active population at the beginning of the call. When making sequential
	 * calls to Evaluate() and Derivative(), make sure to call Evaluate()
	 * first to get an equivalent Derivative(). */
	virtual void Foreprop()
	{
		if( _activeInds.size() == 0 ) { return; }
		for( unsigned int i = 0; i < _activeInds.size(); i++ )
		{
			if( !ParentCost::_sinks[ _activeInds[i] ].IsValid() ) { return; }
		}

		OutputType acc = ParentCost::_sinks[ _activeInds[0] ].GetInput();
		for( unsigned int i = 1; i < _activeInds.size(); i++ )
		{
			acc += ParentCost::_sinks[ _activeInds[i] ].GetInput();
		}

		ParentCost::SourceType::SetOutput( acc / _activeInds.size() );
		ParentCost::SourceType::Foreprop();
	}

	virtual void Invalidate()
	{
		ParentCost::SourceType::Invalidate();
	}

private:

	// Used for random selection
	mutable boost::random::mt19937 _generator;

	unsigned int _subsetSize;
	mutable std::vector<unsigned int> _activeInds;

	void RandomSample() const
	{
		if( ParentCost::_sinks.size() > _subsetSize )
		{
			BitmapSampling( ParentCost::_sinks.size(), _subsetSize, 
			                   _activeInds, _generator );
			return;
		}

		// If there isn't enough data, just use all of it
		_activeInds.clear();
		_activeInds.reserve( ParentCost::_sinks.size() );
		for( unsigned int i = 0; i < ParentCost::_sinks.size(); ++i )
		{
			_activeInds.push_back( i );
		}
	}

};

}