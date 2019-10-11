#ifndef TEST_UTILS_HH_INCLUDED
#define TEST_UTILS_HH_INCLUDED

#include <cuda.h>

#include "uint256_t.h"

namespace test_utils
{
    template<class T>
        cudaError_t HtoD( T *dest, T *src, std::size_t bytes )
    {
        cudaError_t err = cudaMemcpy( dest, src, bytes, cudaMemcpyHostToDevice );
        return err;
    }

    template<class T>
        cudaError_t DtoH( T *dest, T *src, std::size_t bytes )
    {
        cudaError_t err = cudaMemcpy( dest, src, bytes, cudaMemcpyDeviceToHost );
        return err;
    }

        __device__ bool eq_dev( uint256_t *one, uint256_t *two )
        {
            return *one == *two;
        }

        
        /**
         * Device function to perform a binary operation.
         * @param Type the type whose binary operator we wish to test.
         * @param func Binary operator member of Type that takes a member of Type.
         * @param one The member of Type whose func will be called
         * @param two The member of Type that will be passed into the 
         *        func call 
         * @returns the boolean result of the call, true or false
         **/
        template<typename Type, bool (Type::*func)( Type )>
            __device__ bool bin_op_dev( Type *one, Type *two )
        {
            return (one->*func)( *two );
        }


        /**
         * Device function to perform a unary operation.
         * @param Type the type on which to perform the operation.
         * @param func Pointer to the  unary operator, 
         *        which is a function member of Type to test
         * @param one Pointer to the object to test.
         * @returns a member of type Type, the result of the unary 
         *          operator
         **/
        template<typename Type, Type (Type::*func)()>
            __device__ Type unary_op_dev( Type *one )
        {
            return (one->*func)();
        }

        /**
         * A kernel to test a binary operator for class Type.
         * @param Type The type whose binary operator to test
         * @param func A pointer to Type's member function that 
         *        takes a Type and returns bool
         * @param one Pointer to the first variable to test
         * @param two Pointer to the first variable to test
         * @param dest Pointer to bool where the result should be stored
         * @note (one->*func)( *two ) will be called by this function
         **/
        template<class Type, bool (Type::*func)( Type )>
            __global__ void binary_op_kernel( Type *one, Type *two,
                                              bool *dest
                                            )
            {
                bool comp = false;

                comp = bin_op_dev<Type, func>( one, two );
                *dest = comp;
            }


        /**
         * A kernel to test a unary operator for class Type.
         * @param Type The type whose binary operator to test
         * @param func Pointer to member function of Type to test.
         *        this function takes no parameters, and returns the 
         *        'unary-operated' one.
         * @param one the member of Type on which to perform func
         * @param dest The destination, where the result of func will 
         *        be stored
         **/
        template<class Type, Type (Type::*func)()>
            __global__ void unary_op_kernel( Type *one,
                                             Type *dest
                                             )
            {
                Type ret = unary_op_dev<Type, func>( one );

                *dest = ret;
            }
};

#endif
