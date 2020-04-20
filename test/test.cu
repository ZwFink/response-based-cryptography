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

     // test_utils::binary_op_kernel<uint256_t, &uint256_t::operator==><<<1,1>>>( a1_dev, a2_dev, result_code_dev );
     cudaDeviceSynchronize();

     if( test_utils::DtoH( &result_code, result_code_dev, sizeof( bool ) ) != cudaSuccess)
         {
             std::cout << "Failure to transfer to host \n";
         }

     // REQUIRE( result_code );

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

     // test_utils::binary_op_kernel<uint256_t, &uint256_t::operator==><<<1,1>>>( a1_dev, a2_dev, result_code_dev );
     cudaDeviceSynchronize();

     if( test_utils::DtoH( &result_code, result_code_dev, sizeof( bool ) ) != cudaSuccess)
         {
             std::cout << "Failure to transfer to host \n";
         }

     // REQUIRE( !result_code );

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
             a1[ idx ] = 0xFFFFFFFF; // a1 == ~a2
             a2[ idx ] = 0x00000000;
         }

     check_reqs();

     for( std::uint8_t idx = 0; idx < UINT256_SIZE_IN_BYTES; idx += 2 )
         {
             a1[ idx ] = 0x00000000; // a1 == ~a2
             a2[ idx ] = 0xFFFFFFFF;
         }

     check_reqs();

     for( std::uint8_t idx = 0; idx < UINT256_SIZE_IN_BYTES; idx += 2 )
         {
             a1[ idx ] = ( rand() % 256 );
             a2[ idx ] = ~(a1[idx]);
         }

     check_reqs();
 }

 TEST_CASE( "uint256_t_add", "[uint256_t]" )
 {
     uint256_t a1;
     uint256_t a2;
     uint256_t result;
     uint256_t to_comp;

     uint256_t *a1_dev = nullptr;
     uint256_t *a2_dev = nullptr;
     uint256_t *result_dev = nullptr;

     cudaMalloc( (void**) &a1_dev, sizeof( uint256_t ) );
     cudaMalloc( (void**) &a2_dev, sizeof( uint256_t ) );
     cudaMalloc( (void**) &result_dev, sizeof( uint256_t ) );

     bool result_code = false;

     SECTION( "0+214748367=0" )
         {
             a1.set_all( 0xFFFFFFFF );
             a2.set_all( 0x00000000 );
             result.set_all( 0x00000000 );

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

     SECTION( "65280 + 11141120 = 11206400" )
         {
             a1.set_all( 0 );
             a2.set_all( 0 );
             a1.set( 65280, 0 );
             a2.set( 11141120, 0 );
             result.set_all( 0 );
             to_comp.set_all( 0 );
             to_comp.set( 11206400, 0 );

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

             result_code = result == to_comp;
             REQUIRE( result_code );
         } 

     SECTION( "1073741823 + 3221225472 = 4294967295" )
         {
             // 1073741823 + 1073741824 = 2147483647

             // 1073741823
             a1.set_all( 0x3FFFFFFF );

             // 
             a2.set_all( 0xC0000000 );

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

             // a1 + a2, but storing the result back in a1
             test_utils::add_knl<<<1,1>>>( a1_dev, a2_dev, a1_dev );
             cudaDeviceSynchronize();

             if( test_utils::DtoH( &result, a1_dev, sizeof( uint256_t ) ) != cudaSuccess)
                 {
                     std::cout << "Failure to transfer to host \n";
                 }

             result_code = result == UINT256_MAX_INT;
             REQUIRE( result_code );

         }

     cudaFree( a1_dev );
     cudaFree( a2_dev );
     cudaFree( result_dev );

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
             a1[ 7 ] = 0x44;

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
             a2[ 7 ] = 0x44;

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

 TEST_CASE( "uint256_iter", "[uint256_iterator]" )
 {

     // key, first_perm, last_perm
     uint256_t a( 0xAF ), b( 0x00 ), c( 0x00 );
     uint256_t *a_dev, *b_dev, *c_dev;
     uint256_iter *iter_ptr;
     int *count_ptr;

     cudaMallocManaged( &a_dev, sizeof( uint256_t ) );
     cudaMallocManaged( &b_dev, sizeof( uint256_t ) );
     cudaMallocManaged( &c_dev, sizeof( uint256_t ) );
     cudaMallocManaged( &count_ptr, sizeof( int ) );
     cudaMallocManaged( &iter_ptr, sizeof( uint256_iter ) );

     *a_dev = a;
     *b_dev = b;
     *c_dev = c;

     SECTION( "All of the keys are generated when there is one thread" )
         {
             test_utils::get_perm_pair_knl<<<1,1>>>( b_dev, c_dev, 0, 1 );

             cudaDeviceSynchronize();
             uint256_iter iter( *a_dev, *b_dev, *c_dev ) ;

             *iter_ptr = iter;

             test_utils::uint256_iter_next_knl<<<1,1>>>( iter_ptr,
                                                         a_dev,
                                                         b_dev,
                                                         c_dev,
                                                         count_ptr
                                                         );

             cudaDeviceSynchronize();
             // REQUIRE( *count_ptr == 32640 );
         }
     SECTION( "All of the keys are generated when there are two threads" )
         {
             uint256_t start_2( 0x00 ), end_2( 0x00 );

             b_dev->set_all( 0x00 );
             c_dev->set_all( 0x00 );

             uint256_t *start_2_ptr, *end_2_ptr;
             uint256_iter *iter_2_ptr;
             int *count_2_ptr;

             cudaMallocManaged( &start_2_ptr, sizeof( uint256_t ) );
             cudaMallocManaged( &end_2_ptr, sizeof( uint256_t ) );
             cudaMallocManaged( &count_2_ptr, sizeof( int ) );
             cudaMallocManaged( &iter_2_ptr, sizeof( uint256_iter ) );


             test_utils::get_perm_pair_knl<<<1,1>>>( b_dev, c_dev, 0, 2 );
             test_utils::get_perm_pair_knl<<<1,1>>>( start_2_ptr, end_2_ptr, 1, 2 );

             cudaDeviceSynchronize();

             uint256_iter iter( *a_dev, *b_dev, *c_dev ) ;
             uint256_iter iter2( *a_dev, *start_2_ptr, *end_2_ptr ) ;

             *iter_ptr = iter;
             *iter_2_ptr = iter2;
             test_utils::uint256_iter_next_knl<<<1,1>>>( iter_ptr,
                                                         a_dev,
                                                         b_dev,
                                                         c_dev,
                                                         count_ptr
                                                         );

             test_utils::uint256_iter_next_knl<<<1,1>>>( iter_2_ptr,
                                                         a_dev,
                                                         b_dev,
                                                         c_dev,
                                                         count_2_ptr
                                                       );

             cudaDeviceSynchronize();
             REQUIRE( *count_ptr - *count_2_ptr < 2 );

         }

 }

TEST_CASE( "AES Encryption", "[aes_per_round]" )
{

    uint *pt, *ct, *key;

    cudaMallocManaged( (void**) &pt, sizeof( uint ) * 4 );
    cudaMallocManaged( (void**) &ct, sizeof( uint ) * 4 );
    cudaMallocManaged( (void**) &key, sizeof( uint ) * 8 );

    pt[ 0 ] = 0x00112233;
    pt[ 1 ] = 0x44556677;
    pt[ 2 ] = 0x8899aabb;
    pt[ 3 ] = 0xccddeeff;


    ct[ 0 ] = 0;
    ct[ 1 ] = 0;
    ct[ 2 ] = 0;
    ct[ 3 ] = 0;

    key[ 0 ] = 0x00010203;
    key[ 1 ] = 0x04050607;
    key[ 2 ] = 0x08090a0b;
    key[ 3 ] = 0x0c0d0e0f;
    key[ 4 ] = 0x10111213;
    key[ 5 ] = 0x14151617;
    key[ 6 ] = 0x18191a1b;
    key[ 7 ] = 0x1c1d1e1f;

    test_utils::aes_encryption_test<<<1,1>>>( ct, key, pt );

    cudaDeviceSynchronize();

    printf( "Key: 0x%X%X%X%X%X%X%X%X\n",
            key[ 0 ],
            key[ 1 ],
            key[ 2 ],
            key[ 3 ],
            key[ 4 ],
            key[ 5 ],
            key[ 6 ],
            key[ 7 ]



            );
    printf( "Plaintext: 0x%X%X%X%X\n",
            pt[ 0 ],
            pt[ 1 ],
            pt[ 2 ],
            pt[ 3 ]
            );



    printf( "0x%X\n", ct[ 0 ] );
    printf( "0x%X\n", ct[ 1 ] );
    printf( "0x%X\n", ct[ 2 ] );
    printf( "0x%X\n", ct[ 3 ] );

    cudaFree( pt );
    cudaFree( ct );
    cudaFree( key );
}



 TEST_CASE( "uint256_t_negation_gpu", "[uint256_t]" )
 {

     uint256_t a1( 1, 0 );
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

     //test_utils::unary_op_kernel<uint256_t, &uint256_t::operator~><<<1,1>>>
     //     ( a1_dev,
     //       a2_dev
     //     );

     cudaDeviceSynchronize();

     if( test_utils::DtoH( &a2, a2_dev, sizeof( uint256_t ) ) != cudaSuccess)
         {
             std::cout << "Failure to transfer to host \n";
         }

     result_code = a2 == ~a1;

     // REQUIRE( result_code );

     cudaFree( a1_dev );
     cudaFree( a2_dev );
 }

 TEST_CASE( "uint256_t_<<", "[uint256_t]" )
 {

     // test case 1
     uint256_t large = UINT256_MAX_INT;
     uint256_t shifted = large << 256;
     bool success = shifted == UINT256_ZERO;
     REQUIRE( success );


     // test case 2
     large.set_all( 0xFFFFFFFF );
     shifted = large << 128;
     for( int x = 0; x < UINT256_SIZE_IN_BYTES / 2; ++x )
         {
             REQUIRE( shifted[ x ] == 0x00000000 );
         }
     for( int x = UINT256_SIZE_IN_BYTES / 2; x < UINT256_SIZE_IN_BYTES; ++x )
         {
             REQUIRE( shifted[ x ] == 0xFFFFFFFF );
         }


     // test case 3
     large.set_all( 0xFFFFFFFF );
     shifted = large << 2;
     for( int x = UINT256_SIZE_IN_BYTES-1; x > 0; --x )
         {
             REQUIRE( shifted[ x ] == 0xFFFFFFFF );
         }
     REQUIRE( shifted[ 0 ] == 0xFFFFFFFC );


     // test case 4
     large.set_all( 0xFFFFFFFF );
     shifted = large << 1;
     for( int x = UINT256_SIZE_IN_BYTES-1; x > 0; --x )
         {
             REQUIRE( shifted[ x ] == 0xFFFFFFFF );
         }
     REQUIRE( shifted[ 0 ] == 0xFFFFFFFE );
 }

 TEST_CASE( "uint256_t_>>", "[uint256_t]" )
 {

     uint256_t large = UINT256_MAX_INT;

     uint256_t shifted = large >> 256;

     bool success = shifted == UINT256_ZERO;

     REQUIRE( success );

     // reset it
     large.set_all( 0xFFFFFFFF );

     // check first bytes = 16 = 0xFF, last 16 = 0x00
     shifted = large >> 128;


     for( int x = 0; x < UINT256_SIZE_IN_BYTES / 2; ++x )
         {
             REQUIRE( shifted[ x ] == 0xFFFFFFFF );
         }
     for( int x = UINT256_SIZE_IN_BYTES / 2; x < UINT256_SIZE_IN_BYTES; ++x )
         {
             REQUIRE( shifted[ x ] == 0x00000000 );
         }

     large.set_all( 0xFFFFFFFF );
     shifted = large >> 2;

     for( int x = 0; x < UINT256_SIZE_IN_BYTES-1; ++x )
         {
             REQUIRE( shifted[ x ] == 0xFFFFFFFF );
         }

     REQUIRE( shifted[ UINT256_SIZE_IN_BYTES-1 ] == 0x3FFFFFFF );

     large.set_all( 0xFFFFFFFF );

     shifted = large >> 1;

     for( int x = 0; x < UINT256_SIZE_IN_BYTES-1; ++x )
         {
             REQUIRE( shifted[ x ] == 0xFFFFFFFF );
         }

     REQUIRE( shifted[ UINT256_SIZE_IN_BYTES-1 ] == 0x7FFFFFFF );
 }


 TEST_CASE( "uint256_t_ctz_popc", "[uint256_t]" )
 {
     uint256_t my_int;
     uint256_t my_int_2;
     int z_count = 0;

     uint256_t *my_int_dev;
     uint256_t *my_int_dev_2;
     int *z_count_dev;

     cudaMallocManaged( (void**) &my_int_dev, sizeof( uint256_t ) );
     cudaMallocManaged( (void**) &my_int_dev_2, sizeof( uint256_t ) );
     cudaMallocManaged( (void**) &z_count_dev, sizeof( int ) );

     bool result_code = false;

     *my_int_dev = my_int;
     *my_int_dev_2 = my_int_2;
     *z_count_dev = z_count;

     test_utils::popc<<<1,1>>>( my_int_dev, z_count_dev );
     cudaDeviceSynchronize();

     REQUIRE( *z_count_dev == 0 );

     test_utils::ctz<<<1,1>>>( my_int_dev, z_count_dev );
     cudaDeviceSynchronize();

     REQUIRE( *z_count_dev == 256 );

     // test_utils::unary_op_kernel<uint256_t, &uint256_t::operator~><<<1,1>>>
     //     ( my_int_dev,
     //       my_int_dev_2
     //     );

     test_utils::popc<<<1,1>>>( my_int_dev_2, z_count_dev );
     cudaDeviceSynchronize();

     // REQUIRE( *z_count_dev == 256 );

     test_utils::ctz<<<1,1>>>( my_int_dev_2, z_count_dev );
     cudaDeviceSynchronize();

     // REQUIRE( *z_count_dev == 0 );

     (*my_int_dev)[ 0 ] = 0x00000001;
     test_utils::ctz<<<1,1>>>( my_int_dev, z_count_dev );
     cudaDeviceSynchronize();
     REQUIRE( *z_count_dev == 0 );

     (*my_int_dev)[ 0 ] = 0x00000002;
     test_utils::ctz<<<1,1>>>( my_int_dev, z_count_dev );
     cudaDeviceSynchronize();
     REQUIRE( ( *z_count_dev == 1 ) );

     my_int_dev->set_all( 0 );

     //(*my_int_dev)[ 8 ] = 0x00000001;
     //(*my_int_dev)[ 9 ] = 0x00000002; // to make sure we aren't reading downstream trailing zeroes
     //test_utils::ctz<<<1,1>>>( my_int_dev, z_count_dev );
     //cudaDeviceSynchronize();
     //REQUIRE( ( *z_count_dev == 64 ) );


     cudaFree( my_int_dev );
     cudaFree( my_int_dev_2 );
     cudaFree( z_count_dev );
 }


 TEST_CASE( "uint256_t::neg", "[uint256_t]" )
 {
     uint256_t a( 0xFFFFFFFF );
     uint256_t b( 0x00000001, 0 );
     uint256_t c( 0x00000000 );

     uint256_t *a_dev = nullptr;
     uint256_t *c_dev = nullptr;

     cudaMallocManaged( &a_dev, sizeof( uint256_t ) );
     cudaMallocManaged( &c_dev, sizeof( uint256_t ) );

     *a_dev = a;
     *c_dev = c;

     test_utils::neg_knl<<<1,1>>>( a_dev, c_dev );
     cudaDeviceSynchronize();

     bool result = *c_dev == b;
     REQUIRE( result );
 }



