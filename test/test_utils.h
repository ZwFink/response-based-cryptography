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

        
        template<typename Type, bool (Type::*func)( Type )>
            __device__ bool bin_op_dev( Type *one, Type *two )
        {
            return (one->*func)( *two );
        }

        template<typename Type, Type (Type::*func)()>
            __device__ Type unary_op_dev( Type *one )
        {
            return (one->*func)();
        }

        template<class Type, bool (Type::*func)( Type )>
    __global__ void binary_op_kernel( Type *one, Type *two,
                                      bool *dest
                                    )
    {
        bool comp = false;

        comp = bin_op_dev<Type, func>( one, two );
        *dest = comp;
    }

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
