// 2019/04/15 - created by Tsung-Wei Huang
//   - modified from boost/predef/version_number.hpp

#pragma once

// TF_VERSION_NUMBER(major,minor,patch)
//
// Defines standard version numbers, with these properties:
//
//  Decimal base whole numbers in the range \[0,1000000000).
//  The number range is designed to allow for a (2,2,5) triplet.
//  Which fits within a 32 bit value.
//
//  The `major` number can be in the \[0,99\] range.
//  The `minor` number can be in the \[0,99\] range.
//  The `patch` number can be in the \[0,99999\] range.
//
//  Values can be specified in any base. As the defined value
//  is an constant expression.
//
//  Value can be directly used in both preprocessor and compiler
//  expressions for comparison to other similarly defined values.
//
//  The implementation enforces the individual ranges for the
//  major, minor, and patch numbers. And values over the ranges
//  are truncated (modulo).


#define TF_VERSION_NUMBER(major,minor,patch) \
( (((major)%100)*10000000) + (((minor)%100)*100000) + ((patch)%100000) )

#define TF_VERSION_NUMBER_MAX TF_VERSION_NUMBER(99,99,99999)

#define TF_VERSION_NUMBER_ZERO TF_VERSION_NUMBER(0,0,0)

#define TF_VERSION_NUMBER_MIN TF_VERSION_NUMBER(0,0,1)

#define TF_VERSION_NUMBER_AVAILABLE TF_VERSION_NUMBER_MIN

#define TF_VERSION_NUMBER_NOT_AVAILABLE TF_VERSION_NUMBER_ZERO

#define TF_VERSION_NUMBER_MAJOR(N) ( ((N)/10000000)%100 )

#define TF_VERSION_NUMBER_MINOR(N) ( ((N)/100000)%100 )

#define TF_VERSION_NUMBER_PATCH(N) ( (N)%100000 )
