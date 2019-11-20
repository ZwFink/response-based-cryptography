#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <string>
#include <iostream>
#include <cuda.h>
#include <algorithm>
#include <cstdlib>

#include "test_utils.h"

#include "aes_per_round.h"
#include "uint256_iterator.h"
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

    test_utils::binary_op_kernel<uint256_t, &uint256_t::operator==><<<1,1>>>( a1_dev, a2_dev, result_code_dev );

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

    test_utils::binary_op_kernel<uint256_t, &uint256_t::operator==><<<1,1>>>( a1_dev, a2_dev, result_code_dev );

    if( test_utils::DtoH( &result_code, result_code_dev, sizeof( bool ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer to host \n";
        }

    REQUIRE( !result_code );

    cudaFree( a1_dev );
    cudaFree( a2_dev );
    cudaFree( result_code_dev );

}

TEST_CASE( "uint256_t_negation_cpu", "[uint256_t]" )
{

    uint256_t a1;
    uint256_t a2;

    bool cond = false;

    auto check_reqs = [&]()
        {
            cond = a1 == ~a2;
            REQUIRE( cond );

            cond = ~a1 == a2;
            REQUIRE( cond );

        };


    for( std::uint8_t idx = 0; idx < UINT256_SIZE_IN_BYTES; ++idx )
        {
            a1[ idx ] = 0xFF; // a1 == ~a2
            a2[ idx ] = 0x00;
        }

    check_reqs();

    for( std::uint8_t idx = 0; idx < UINT256_SIZE_IN_BYTES; idx += 2 )
        {
            a1[ idx ] = 0x00; // a1 == ~a2
            a2[ idx ] = 0xFF;
        }

    check_reqs();

    for( std::uint8_t idx = 0; idx < UINT256_SIZE_IN_BYTES; idx += 2 )
        {
            a1[ idx ] = ( rand() % 256 );
            a2[ idx ] = ~(a1[idx]);
        }

    check_reqs();
}

TEST_CASE( "uint256_t_negation_gpu", "[uint256_t]" )
{

    uint256_t a1;
    uint256_t a2;

    uint256_t *a1_dev = nullptr;
    uint256_t *a2_dev = nullptr;

    cudaMalloc( (void**) &a1_dev, sizeof( uint256_t ) );
    cudaMalloc( (void**) &a2_dev, sizeof( uint256_t ) );

    bool result_code = false;

    if( test_utils::HtoD( a1_dev, &a1, sizeof( uint256_t ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer a1 to device\n";
        }

    if( test_utils::HtoD( a2_dev, &a2, sizeof( uint256_t ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer a2 to device\n";
        }

    test_utils::unary_op_kernel<uint256_t, &uint256_t::operator~><<<1,1>>>
        ( a1_dev,
          a2_dev
        );

    if( test_utils::DtoH( &a2, a2_dev, sizeof( uint256_t ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer to host \n";
        }

    result_code = a2 == ~a1;

    REQUIRE( result_code );

    cudaFree( a1_dev );
    cudaFree( a2_dev );
}

TEST_CASE( "uint256_t_ctz_popc", "[uint256_t]" )
{
    uint256_t my_int;
    uint256_t my_int_2;
    int z_count = 0;

    uint256_t *my_int_dev;
    uint256_t *my_int_dev_2;
    int *z_count_dev;

    cudaMalloc( (void**) &my_int_dev, sizeof( uint256_t ) );
    cudaMalloc( (void**) &my_int_dev_2, sizeof( uint256_t ) );
    cudaMalloc( (void**) &z_count_dev, sizeof( int ) );

    bool result_code = false;

    if( test_utils::HtoD( my_int_dev, &my_int, sizeof( uint256_t ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer a1 to device\n";
        }

    if( test_utils::HtoD( z_count_dev, &z_count, sizeof( int ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer a2 to device\n";
        }

    test_utils::popc<<<1,1>>>( my_int_dev, z_count_dev );

    if( test_utils::DtoH( &z_count, z_count_dev, sizeof( int ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer a2 to device\n";
        }

    REQUIRE( z_count == 0 );

    test_utils::ctz<<<1,1>>>( my_int_dev, z_count_dev );

    if( test_utils::DtoH( &z_count, z_count_dev, sizeof( int ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer a2 to device\n";
        }

    REQUIRE( z_count == 256 );

    test_utils::unary_op_kernel<uint256_t, &uint256_t::operator~><<<1,1>>>
        ( my_int_dev,
          my_int_dev_2
        );

    test_utils::popc<<<1,1>>>( my_int_dev_2, z_count_dev );

    if( test_utils::DtoH( &z_count, z_count_dev, sizeof( int ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer a2 to device\n";
        }

    REQUIRE( z_count == 256 );

    test_utils::ctz<<<1,1>>>( my_int_dev_2, z_count_dev );

    if( test_utils::DtoH( &z_count, z_count_dev, sizeof( int ) ) != cudaSuccess)
        {
            std::cout << "Failure to transfer a2 to device\n";
        }

    REQUIRE( z_count == 0 );

    cudaFree( my_int_dev );
    cudaFree( my_int_dev_2 );
    cudaFree( z_count_dev );
}

TEST_CASE( "uint256_t_<<", "[uint256_t]" )
{

    uint256_t large = UINT256_MAX_INT;

    uint256_t shifted = large << 256;

    bool success = shifted == UINT256_ZERO;

    REQUIRE( success );

    // reset it
    large.set_all( 0xFF );

    // check first bytes = 16 = 0xFF, last 16 = 0x00
    shifted = large << 128;


    for( int x = 0; x < UINT256_SIZE_IN_BYTES / 2; ++x )
        {
            REQUIRE( shifted[ x ] == 0x00 );
        }
    for( int x = UINT256_SIZE_IN_BYTES / 2; x < UINT256_SIZE_IN_BYTES; ++x )
        {
            REQUIRE( shifted[ x ] == 0xFF );
        }

    large.set_all( 0xFF );
    shifted = large << 2;

    for( int x = 31; x > 0; --x )
        {
            REQUIRE( shifted[ x ] == 0xFF );
        }

    REQUIRE( shifted[ 0 ] == 0xFC );

    large.set_all( 0xFF );

    shifted = large << 1;

    for( int x = 31; x > 0; --x )
        {
            REQUIRE( shifted[ x ] == 0xFF );
        }

    REQUIRE( shifted[ 0 ] == 0xFE );
}

TEST_CASE( "uint256_t_>>", "[uint256_t]" )
{

    uint256_t large = UINT256_MAX_INT;

    uint256_t shifted = large >> 256;

    bool success = shifted == UINT256_ZERO;

    REQUIRE( success );

    // reset it
    large.set_all( 0xFF );

    // check first bytes = 16 = 0xFF, last 16 = 0x00
    shifted = large >> 128;


    for( int x = 0; x < UINT256_SIZE_IN_BYTES / 2; ++x )
        {
            REQUIRE( shifted[ x ] == 0xFF );
        }
    for( int x = UINT256_SIZE_IN_BYTES / 2; x < UINT256_SIZE_IN_BYTES; ++x )
        {
            REQUIRE( shifted[ x ] == 0x00 );
        }

    large.set_all( 0xFF );
    shifted = large >> 2;

    for( int x = 0; x < 31; ++x )
        {
            REQUIRE( shifted[ x ] == 0xFF );
        }

    REQUIRE( shifted[ 31 ] == 0x3F );

    large.set_all( 0xFF );

    shifted = large >> 1;

    for( int x = 0; x < 31; ++x )
        {
            REQUIRE( shifted[ x ] == 0xFF );
        }

    REQUIRE( shifted[ 31 ] == 0x7F );
}

TEST_CASE( "uint256_t_add", "[uint256_t]" )
{
    uint256_t a1;
    uint256_t a2;
    uint256_t result;

    uint256_t *a1_dev = nullptr;
    uint256_t *a2_dev = nullptr;
    uint256_t *result_dev = nullptr;

    cudaMalloc( (void**) &a1_dev, sizeof( uint256_t ) );
    cudaMalloc( (void**) &a2_dev, sizeof( uint256_t ) );
    cudaMalloc( (void**) &result_dev, sizeof( uint256_t ) );

    bool result_code = false;

    SECTION( "0+255=0" )
        {
            a1.set_all( 0xFF );
            a2.set_all( 0x00 );
            result.set_all( 0x00 );

            if( test_utils::HtoD( a1_dev, &a1, sizeof( uint256_t ) ) != cudaSuccess )
                {
                    std::cout << "Failure to transfer a1 to device\n";
                }

            if( test_utils::HtoD( a2_dev, &a2, sizeof( uint256_t ) ) != cudaSuccess)
                {
                    std::cout << "Failure to transfer a2 to device\n";
                }

            if( test_utils::HtoD( result_dev, &result, sizeof( uint256_t ) ) != cudaSuccess)
                {
                    std::cout << "Failure to transfer result to device\n";
                }

            test_utils::add_knl<<<1,1>>>( a1_dev, a2_dev, result_dev );
            cudaDeviceSynchronize();

            if( test_utils::DtoH( &result, result_dev, sizeof( uint256_t ) ) != cudaSuccess)
                {
                    std::cout << "Failure to transfer to host \n";
                }

            result_code = result == UINT256_MAX_INT;
            REQUIRE( result_code );
        }

    SECTION( "188+67=255" )
        {
            // 188 + 67 = 255

            // 188
            a1.set_all( 0xBC );

            // 67
            a2.set_all( 0x43 );

            if( test_utils::HtoD( a1_dev, &a1, sizeof( uint256_t ) ) != cudaSuccess )
                {
                    std::cout << "Failure to transfer a1 to device\n";
                }

            if( test_utils::HtoD( a2_dev, &a2, sizeof( uint256_t ) ) != cudaSuccess)
                {
                    std::cout << "Failure to transfer a2 to device\n";
                }

            if( test_utils::HtoD( result_dev, &result, sizeof( uint256_t ) ) != cudaSuccess)
                {
                    std::cout << "Failure to transfer result to device\n";
                }

            test_utils::add_knl<<<1,1>>>( a1_dev, a2_dev, result_dev );
            cudaDeviceSynchronize();

            if( test_utils::DtoH( &result, result_dev, sizeof( uint256_t ) ) != cudaSuccess)
                {
                    std::cout << "Failure to transfer to host \n";
                }

            result_code = result == UINT256_MAX_INT;
            REQUIRE( result_code );
        }

    cudaFree( a1_dev );
    cudaFree( a2_dev );

}

TEST_CASE( "uint256_t<", "[uint256_t]" )
{
    uint256_t a1( 0x00 );
    uint256_t a2( 0x00 );
    bool result = false;

    SECTION( "0<255" )
        {
            a1.set_all( 0xFF );

            result = a2 < a1;

            REQUIRE( a2 < a1 );
        }
    SECTION( "n-1<n" )
        {
            a1.set_all( 0x43 );
            a2.set_all( 0x43 );

            a2[ 0 ] = 0x42;
            result = a2 < a1;

            REQUIRE( result );
        }
    SECTION( "n < x, x way bigger than n" )
        {
            a1.set_all( 0x43 );
            a2.set_all( 0x43 );
            a1[ 31 ] = 0x44;

            result = a2 < a1;

            REQUIRE( result );


        }
    SECTION( "!(n < n)" )
        {
            a1.set_all( 0xFF );
            a2.set_all( 0xFF );

            result = a1 < a2;

            REQUIRE( !result );

            result = a2 < a1;

            REQUIRE( !result );
        }

}

TEST_CASE( "uint256_t>", "[uint256_t]" )
{
    uint256_t a1( 0x00 );
    uint256_t a2( 0x00 );
    bool result = false;

    SECTION( "255>0" )
        {
            a2.set_all( 0xFF );

            result = a2 > a1;

            REQUIRE( result );
        }
    SECTION( "n>n-n" )
        {
            a1.set_all( 0x43 );
            a2.set_all( 0x43 );

            a2[ 0 ] = 0x44;
            result = a2 > a1;

            REQUIRE( result );
        }
    SECTION( "x > n, x way bigger than n" )
        {
            a1.set_all( 0x43 );
            a2.set_all( 0x43 );
            a2[ 31 ] = 0x44;

            result = a2 > a1;

            REQUIRE( result );


        }
    SECTION( "!(n > n)" )
        {
            a1.set_all( 0xFF );
            a2.set_all( 0xFF );

            result = a1 > a2;

            REQUIRE( !result );

            result = a2 > a1;

            REQUIRE( !result );
        }

}

TEST_CASE( "uint256_t::neg", "[uint256_t]" )
{
    uint256_t a( 0xFF );
    uint256_t b( 0x01, UINT256_SIZE_IN_BYTES - 1 );
    uint256_t c( 0x00 );

    uint256_t *a_dev = nullptr;
    uint256_t *c_dev = nullptr;

    cudaMalloc( (void**) &a_dev, sizeof( uint256_t ) );
    cudaMalloc( (void**) &c_dev, sizeof( uint256_t ) );

    if( test_utils::HtoD( a_dev, &a, sizeof( uint256_t ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer a to device\n";
        }
    if( test_utils::HtoD( c_dev, &c, sizeof( uint256_t ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer c to device\n";
        }

    test_utils::neg_knl<<<1,1>>>( a_dev, c_dev );


    if( test_utils::DtoH( &c, c_dev, sizeof( uint256_t ) ) != cudaSuccess )
        {
            std::cout << "Failure to transfer c to device\n";
        }

    bool result = c == b;
    REQUIRE( result );
}


TEST_CASE( "uint256_iter_constructor", "[uint256_iterator]" )
{

    uint256_t a( 0x00 );
    uint256_t b( 0xFF );
    uint256_iter( (const unsigned char *)
                  "abcdefghijklmnopqrstuvwxyz012345",
                  a,
                  b
                );

}