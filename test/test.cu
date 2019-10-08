#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <string>
#include <iostream>
#include <cuda.h>

#include "test_utils.h"

#include "uint256_t.h"
#include "AES.h"

TEST_CASE( "uint256_t_eq_host", "[uint256_t]" )
{
    uint256_t an_int;
    uint256_t an_int2;

    bool eq = an_int == an_int2; 
    bool neq = an_int != an_int2;

    REQUIRE( eq );
    REQUIRE( !neq );

    uint256_t a1;
    uint256_t a2;

    for( std::uint8_t idx = 0; idx < UINT256_SIZE_IN_BYTES; ++idx )
        {
            a1[ idx ] = idx;
            a2[ idx ] = idx;
        }

    eq = a1 == a2;

    REQUIRE( eq );

    a1[ 0 ] = 0x01;
    a2[ 0 ] = 0x02;

    eq = a1 == a2;

    REQUIRE( !eq );
}

TEST_CASE( "uint256_t_eq_dev", "[uint256_t]" )
{
    uint256_t a1;
    uint256_t a2;

    bool *result_code_dev = nullptr;
    bool result_code = false;

    uint256_t *a1_dev = nullptr;
    uint256_t *a2_dev = nullptr;

    cudaMalloc( (void**) &a1_dev, sizeof( uint256_t ) );
    cudaMalloc( (void**) &a2_dev, sizeof( uint256_t ) );
    cudaMalloc( (void**) &result_code_dev, sizeof( bool ) );

    if( test_utils::HtoD( a1_dev, &a1, sizeof( uint256_t ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer a1 to device\n";
        }

    if( test_utils::HtoD( a2_dev, &a2, sizeof( uint256_t ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer a2 to device\n";
        }
    if( test_utils::HtoD( result_code_dev, &result_code, sizeof( bool ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer result_code to device\n";
        }

    test_utils::eq_kernel<<<1,1>>>( a1_dev,
                                    a2_dev,
                                    result_code_dev
                                  );

    if( test_utils::DtoH( &result_code, result_code_dev, sizeof( bool ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer to host \n";
        }

    REQUIRE( result_code );

    result_code = false;

    a1[ 0 ] = 0x02;

    if( test_utils::HtoD( a1_dev, &a1, sizeof( uint256_t ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer a1 to device\n";
        }

    if( test_utils::HtoD( result_code_dev, &result_code, sizeof( bool ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer result_code to device\n";
        }

    test_utils::eq_kernel<<<1,1>>>( a1_dev,
                                    a2_dev,
                                    result_code_dev
                                  );

    if( test_utils::DtoH( &result_code, result_code_dev, sizeof( bool ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer to host \n";
        }

    REQUIRE( !result_code );


    cudaFree( a1_dev );
    cudaFree( a2_dev );
    cudaFree( result_code_dev );

}