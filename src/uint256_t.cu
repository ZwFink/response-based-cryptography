#include <iostream>
#include <iomanip>

#include "uint256_t.h"

namespace uint256_ctz_table
{
    #ifdef USE_CONSTANT
    __constant__
    #endif
    __device__ 
    std::uint8_t lookup[ 37 ] = 
    {
        32, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13, 4,
        7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9, 5,
        20, 8, 19, 18
    };

    INLINE DEVICE_ONLY int ctz( const std::uint32_t loc )
    {
        return lookup[ ( -loc & loc ) % 37 ];
    }

}

uint256_t::uint256_t()
{
    set_all( 0 );
}

CUDA_CALLABLE_MEMBER uint256_t::uint256_t( std::uint8_t set, std::uint8_t index )
{
    data[ index ] = set;
}

CUDA_CALLABLE_MEMBER void uint256_t::from_string( const unsigned char *string )
{
    for( std::uint8_t index = 0; index < UINT256_SIZE_IN_BYTES; ++index )
        {
            data[ index ] = string[ index ];
        }
}

CUDA_CALLABLE_MEMBER uint256_t::uint256_t( std::uint8_t set )
{
    set_all( set );
}

CUDA_CALLABLE_MEMBER void uint256_t::set_all( std::uint8_t val )
{
    for( std::uint8_t x = 0; x < UINT256_SIZE_IN_BYTES; ++x )
        {
            data[ x ] = val;
        }
}

// copy constructor
CUDA_CALLABLE_MEMBER void uint256_t::copy( const uint256_t& copied )
{
    for( std::uint8_t idx = 0; idx < UINT256_SIZE_IN_BYTES; ++idx )
    {
       data[ idx ] = copied[ idx ];
    }
}

// PRECONDITION: 0 <= idx <= 3
CUDA_CALLABLE_MEMBER void uint256_t::copy_64( uint64_t ref, uint8_t idx )
{
    uint64_t *data_ptr = (uint64_t *) &data;
    
    data_ptr[ idx ] |= ref; // bitwise OR 
}

CUDA_CALLABLE_MEMBER void uint256_t::set( std::uint8_t set, std::uint8_t index ){
    data[ index ] = set;
}

CUDA_CALLABLE_MEMBER const std::uint8_t& uint256_t::operator[]( std::uint8_t idx ) const
{
    return data[ idx ];
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

CUDA_CALLABLE_MEMBER std::uint8_t *uint256_t::get_data_ptr()
{
    return data;
}

CUDA_CALLABLE_MEMBER void uint256_t::operator=( const uint256_t& set )
{
    for( std::uint8_t a = 0; a < UINT256_SIZE_IN_BYTES / 4; ++a )
        {
            *((uint32_t*)data + a )  = *((uint32_t*)set.data + a );
        }
}

CUDA_CALLABLE_MEMBER bool uint256_t::operator!=( uint256_t comp )
{
    return !( *this == comp );
}

__host__ void uint256_t::dump_hex()
{
    char buff[ 163 ] = { 0 };

    for( int x = 0; x < 32; ++x )
        {
            snprintf( buff + ( x * 5 ), 
                      6,
                      "0x%02x ", data[ x ]
                    );
                    
        }
    printf( "%s\n", buff );
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
    std::uint32_t *current = nullptr;

    for( std::uint8_t index = 0; index < UINT256_SIZE_IN_BYTES / 4; ++index )
        {
            current = (std::uint32_t*) data + index;
            total_ones += __popc( *current );
        }

    return total_ones;
}

CUDA_CALLABLE_MEMBER std::uint8_t uint256_t::at( int loc )
{
    return data[ loc ];
}

__device__ int uint256_t::ctz()
{
    int ret = 0;
    int count_limit = 0;
    for( std::uint8_t idx = 0;
         ret == count_limit
          && idx < UINT256_SIZE_IN_BYTES / 4;
         ++idx
       )
        {
            count_limit += sizeof( uint32_t ) * 8;
            ret += uint256_ctz_table::ctz( *((std::uint32_t*) data + idx ) );
        }

    return ret;
}

CUDA_CALLABLE_MEMBER void uint256_t::to_32_bit_arr( std::uint32_t* dest )
{
    memcpy( dest, &(data), 32 );
}

CUDA_CALLABLE_MEMBER int uint256_t::compare( const uint256_t& comp ) const
{
    std::uint32_t *my_data = (std::uint32_t*) &data;
    std::uint32_t *comp_data = (std::uint32_t*) &comp.data;

    int result = 0;

    for( int index = ( UINT256_SIZE_IN_BYTES / 4 ) - 1;
         !result && index >= 0;
         --index
       )
        {
            result = ( my_data[ index ] > comp_data[ index ] )
                     - ( my_data[ index ] < comp_data[ index ] );
        }
    return result;
}

CUDA_CALLABLE_MEMBER bool uint256_t::operator<( const uint256_t& comp ) const
{
    return compare( comp ) < 0;
}

CUDA_CALLABLE_MEMBER bool uint256_t::operator>( const uint256_t& comp ) const
{
    return compare( comp ) > 0;
}

__device__ bool uint256_t::add( uint256_t& dest, const uint256_t augend ) const
{
    uint256_t ret;

    std::uint32_t *self_32   = (uint32_t*) &data;
    std::uint32_t *augend_32 = (uint32_t*) &augend.data;
    std::uint32_t *dest_32   = (uint32_t*) &dest.data;

    asm ("add.cc.u32      %0, %8, %16;\n\t"
         "addc.cc.u32     %1, %9, %17;\n\t"
         "addc.cc.u32     %2, %10, %18;\n\t"
         "addc.cc.u32     %3, %11, %19;\n\t"
         "addc.cc.u32     %4, %12, %20;\n\t"
         "addc.cc.u32     %5, %13, %21;\n\t"
         "addc.cc.u32     %6, %14, %22;\n\t"
         "addc.u32        %7, %15, %23;\n\t"
         : "=r"(dest_32[ 0 ]), "=r"(dest_32[ 1 ]), "=r"(dest_32[ 2 ]),   
           "=r"(dest_32[ 3 ]), "=r"(dest_32[ 4 ]), "=r"(dest_32[ 5 ]),   
           "=r"(dest_32[ 6 ]), "=r"(dest_32[ 7 ])
         : "r"(self_32[ 0 ]), "r"(self_32[ 1 ]), "r"(self_32[ 2 ]),   
           "r"(self_32[ 3 ]), "r"(self_32[ 4 ]), "r"(self_32[ 5 ]),   
           "r"(self_32[ 6 ]), "r"(self_32[ 7 ]),
           "r"(augend_32[ 0 ]), "r"(augend_32[ 1 ]), "r"(augend_32[ 2 ]),   
           "r"(augend_32[ 3 ]), "r"(augend_32[ 4 ]), "r"(augend_32[ 5 ]),   
           "r"(augend_32[ 6 ]), "r"(augend_32[ 7 ])
         );

    return dest_32[ 7 ] < self_32[ 7 ];
}

__device__ void uint256_t::neg( uint256_t& dest )
{
    uint256_t complement = ~(*this);

    complement.add( dest, UINT256_ONE );
}

// intended for use with permutation creation in function decode_ordinal
CUDA_CALLABLE_MEMBER void uint256_t::set_bit( std::uint8_t bit_idx )
{
    std::uint8_t block = bit_idx / 8;
    std::uint8_t ndx_in_block = bit_idx - ( block * 8 );
   
    data[ block ] |= ( 1 << ndx_in_block );
}


