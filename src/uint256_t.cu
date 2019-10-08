#include <iostream>
#include <iomanip>

#include "uint256_t.h"
uint256_t::uint256_t()
{
    memset( data, 0, UINT256_SIZE_IN_BYTES );
}

CUDA_CALLABLE_MEMBER std::uint8_t& uint256_t::operator[]( std::uint8_t idx )
{
    return data[ idx ];
}

CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator~()
{
    uint256_t ret;

    for( std::uint8_t index = 0;
         index < UINT256_SIZE_IN_BYTES;
         ++index
       )
        {
            ret[ index ] = ~data[ index ];
        }
    return ret;
}

CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator&( uint256_t comp )
{
    uint256_t ret;

    for( std::uint8_t index = 0;
         index < UINT256_SIZE_IN_BYTES;
         ++index
       )
        {
            ret[ index ] = comp[ index ] & data[ index ];
        }

    return ret;
}

CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator^( uint256_t comp )
{
    uint256_t ret;

    for( std::uint8_t index = 0;
         index < UINT256_SIZE_IN_BYTES;
         ++index
       )
        {
            ret[ index ] = comp[ index ] ^ data[ index ];
        }

    return ret;
}

CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator|( uint256_t comp )
{
    uint256_t ret;

    for( std::uint8_t index = 0;
         index < UINT256_SIZE_IN_BYTES;
         ++index
       )
        {
            ret[ index ] = comp[ index ] | data[ index ];
        }

    return ret;
}

CUDA_CALLABLE_MEMBER bool uint256_t::operator==( uint256_t comp )
{
    uint8_t ret = 0xFF;
    for( uint8_t byte = 0; byte < UINT256_SIZE_IN_BYTES; ++byte )
        {
            // we want this branchless because CUDA
            ret = ret & ( ~data[ byte ] ^ comp[ byte ] );
        }
    return ret == 0xFF;
}

CUDA_CALLABLE_MEMBER uint256_data_t& uint256_t::get_data()
{
    return data;
}

CUDA_CALLABLE_MEMBER bool uint256_t::operator!=( uint256_t comp )
{
    return !( *this == comp );
}

__host__ void uint256_t::dump()
{
    for( const auto& x : data )
        {
            std::cout
                << "0x"
                << std::setfill('0')
                << std::setw(2)
                << std::hex
                << unsigned( x )
                << " ";
        }
    std::cout << "\n"; 
}