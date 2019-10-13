#include <iostream>
#include <iomanip>

#include "uint256_t.h"

uint256_t::uint256_t()
{
    set_all( 0 );
}
CUDA_CALLABLE_MEMBER uint256_t::uint256_t( std::uint8_t set )
{
    set_all( set );
}

CUDA_CALLABLE_MEMBER void uint256_t::set_all( std::uint8_t val )
{
    memset( data, val, UINT256_SIZE_IN_BYTES );
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
    bool ret = true;
    for( uint8_t byte = 0; byte < UINT256_SIZE_IN_BYTES; ++byte )
        {
            ret = ret && ( data[ byte ] == comp[ byte ] );
        }
    return ret;
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

CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator>>( int shift )
{
    uint256_t ret;

    std::uint8_t limb_shifts  = shift / UINT256_LIMB_SIZE;
    std::uint8_t shift_length = shift % UINT256_LIMB_SIZE;

    std::uint8_t byte = 0;

    for( byte = limb_shifts; byte < UINT256_SIZE_IN_BYTES; ++byte )
        {
            ret[ byte - limb_shifts ] = data[ byte ];
        }

    // leading limbs are alread zero

    for( byte = 0; byte < UINT256_SIZE_IN_BYTES - 1; ++byte )
        {
            ret[ byte ] = ( ret.at( byte ) >> shift_length
                            | ( ret.at( byte + 1 ) << ( UINT256_LIMB_SIZE - shift_length ) )
                          );
        }

    ret[ UINT256_SIZE_IN_BYTES - 1 ] >>= shift;

    return ret;
}

CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator<<( int shift )
{
    uint256_t ret;

    std::uint8_t limb_shifts  = shift / UINT256_LIMB_SIZE;
    std::uint8_t shift_length = shift % UINT256_LIMB_SIZE;

    std::uint8_t byte = 0;

    for( byte = 0; byte < UINT256_SIZE_IN_BYTES - limb_shifts; ++byte )
        {
            ret[ byte + limb_shifts ] = data[ byte ];
        }

    // trailing limbs are alread zero

    for( byte = UINT256_SIZE_IN_BYTES - 1; byte > 0; --byte )
        {
            ret[ byte ] = ret.at( byte ) << shift_length
                | ( ret.at( byte - 1 ) >> ( UINT256_LIMB_SIZE - shift_length ) );
        }
    ret[ 0 ] <<= shift_length;
    
    return ret;

}

__device__ int uint256_t::popc()
{
    int total_ones = 0;
    uint32_t current = 0;

    for( std::uint8_t index = 0; index < 32; index += 4 )
        {
            current |= data[ index ];
            current = current << 8;

            current |= data[ index + 1 ];
            current = current << 8;

            current |= data[ index + 2 ];
            current = current << 8;

            current |= data[ index + 3 ];

            total_ones += __popc( current );
            current = 0;
        }

    return total_ones;
}

CUDA_CALLABLE_MEMBER std::uint8_t uint256_t::at( int loc )
{
    return data[ loc ];
}

__device__ int uint256_t::ctz()
{
    return 256 - popc();
}