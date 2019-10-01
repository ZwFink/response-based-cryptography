#include "uint256_t.h"

CUDA_CALLABLE_MEMBER uint256_t::uint256_t()
{
    data = { 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00
           };
}

CUDA_CALLABLE_MEMBER uint8_t uint256_t::operator[]( std::size_t idx )
{
    return data[ idx ];
}


CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator&( const uint256_t comp )
{
    uint256_t ret;

    for( std::uint8_t index = 0;
         index < UINT_256_SIZE_IN_BYTES;
         ++index
       )
        {
            ret[ index ] = comp[ index ] & data[ index ];
        }

    return ret;
}


