#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <boost/foreach.hpp>

namespace argus
{

VectorType flatten_matrices( const std::vector<MatrixType>& mats )
{
	unsigned int n = 0;
	BOOST_FOREACH( const MatrixType& mat, mats )
	{
		n += mat.size();
	}
	VectorType out(n);
	unsigned int ind = 0;
	BOOST_FOREACH( const MatrixType& mat, mats )
	{
		Eigen::Map<const VectorType> v( mat.data(), mat.size(), 1 );
		out.segment(ind, mat.size()) = v;
		ind += v.size();
	}
	return out;
}

MatrixType vstack_matrices( const std::vector<MatrixType>& mats )
{
	unsigned int rows = 0;
	unsigned int cols = mats[0].cols();

	BOOST_FOREACH( const MatrixType& mat, mats )
	{
		rows += mat.rows();
		if( mat.cols() != cols )
		{
			throw std::invalid_argument( "Not all mats same width!" );
		}
	}

	MatrixType out( rows, cols );
	unsigned int ind = 0;
	BOOST_FOREACH( const MatrixType& mat, mats )
	{
		out.block(ind, 0, mat.rows(), mat.cols() ) = mat;
		ind += mat.rows();
	}
	return out;
}

MatrixType hstack_matrices( const std::vector<MatrixType>& mats )
{
	unsigned int rows = mats[0].rows();
	unsigned int cols = 0;

	BOOST_FOREACH( const MatrixType& mat, mats )
	{
		cols += mat.cols();
		if( mat.rows() != rows )
		{
			throw std::invalid_argument( "Not all mats same height!" );
		}
	}

	MatrixType out( rows, cols );
	unsigned int ind = 0;
	BOOST_FOREACH( const MatrixType& mat, mats )
	{
		out.block(0, ind, mat.rows(), mat.cols() ) = mat;
		ind += mat.cols();
	}
	return out;
}

// template <class Derived>
// Derived ConcatenateHor( const Eigen::DenseBase<Derived>& l,
//                         const Eigen::DenseBase<Derived>& r )
// {
// 	if( l.size() == 0 )
// 	{
// 		return r;
// 	}
// 	if( r.size() == 0 )
// 	{
// 		return l;
// 	}

// 	if( l.rows() != r.rows() )
// 	{
// 		throw std::runtime_error( "ConcatenateHor: Dimension mismatch." );
// 	}
// 	Derived out( l.rows(), l.cols() + r.cols() );
// 	out.leftCols( l.cols() ) = l;
// 	out.rightCols( r.cols() ) = r;
// 	return out;
// }

// template <class Derived>
// Derived ConcatenateVer( const Eigen::DenseBase<Derived>& l,
//                         const Eigen::DenseBase<Derived>& r )
// {
// 	if( l.size() == 0 )
// 	{
// 		return r;
// 	}
// 	if( r.size() == 0 )
// 	{
// 		return l;
// 	}

// 	if( l.cols() != r.cols() )
// 	{
// 		throw std::runtime_error( "ConcatenateVer: Dimension mismatch." );
// 	}
// 	Derived out( l.rows() + r.rows(), l.cols() );
// 	out.topRows( l.rows() ) = l;
// 	out.bottomRows( r.rows() ) = r;
// 	return out;
// }

}