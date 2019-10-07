#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <string>
#include <iostream>
#include <cuda.h>

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
    REQUIRE( true );
}