#include <iostream>
#include <iomanip>

#include "uint256_t.h"

// Convention: data is stored in a 'little-endian' manner
//             - the least significant byte is stored first




// namespace below has not been adapted
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




/* Constructors and methods */

// default constructor
uint256_t::uint256_t()
{
    set_all( 0 );
}
// initialization constructor - set all indices of data to the same value
CUDA_CALLABLE_MEMBER uint256_t::uint256_t( std::uint32_t set )
{
    set_all( set );
}
// initialization constructor helper
CUDA_CALLABLE_MEMBER void uint256_t::set_all( std::uint32_t val )
{
    for( std::uint8_t x = 0; x < UINT256_SIZE_IN_BYTES; ++x )
        {
            data[ x ] = val;
        }
}
// initialization constructor - set specified index of data to specified value
CUDA_CALLABLE_MEMBER uint256_t::uint256_t( std::uint32_t set, std::uint8_t index )
    : uint256_t()
{
    data[ index ] = set;
}
// alternative method to set value at index
CUDA_CALLABLE_MEMBER void uint256_t::set( std::uint32_t set, std::uint8_t index )
{
    data[ index ] = set;
}
// conversion from string
CUDA_CALLABLE_MEMBER void uint256_t::from_string( const unsigned char *string )
{
    for( std::uint8_t index = 0; index < UINT256_SIZE_IN_BYTES; ++index )
        {
            data[ index ] = string[ index ];
        }
}
// intended for use with permutation creation in function decode_ordinal
CUDA_CALLABLE_MEMBER void uint256_t::set_bit( std::uint8_t bit_idx )
{
    std::uint8_t block = bit_idx / UINT256_LIMB_SIZE;
    std::uint8_t ndx_in_block = bit_idx - ( block * UINT256_LIMB_SIZE );
   
    data[ block ] |= ( 1 << ndx_in_block );
}
// host method only - for printing
__host__ void uint256_t::dump()
{
    for( const auto& x : data )
        {
            std::cout
               << "0x"
               << std::setfill('0')
               << std::setw(8)
               << std::hex
               << unsigned( x )
               << " ";
        }
    std::cout << "\n"; 
}
// count trailing zeros - utilizes namespace uint256_ctz_table
__device__ int uint256_t::ctz()
{
    int ret = 0;
    int count_limit = 0;
    for( std::uint8_t idx = 0;
         ret == count_limit
          && idx < UINT256_SIZE_IN_BYTES;
         ++idx
       )
        {
            count_limit += sizeof( uint32_t ) * 8;
            ret += uint256_ctz_table::ctz( *(data + idx) );
        }

    return ret;
}




/* Operators and their associated helper methods */

CUDA_CALLABLE_MEMBER int uint256_t::compare( const uint256_t& comp ) const
{
    int result = 0;
    for( int index = UINT256_SIZE_IN_BYTES - 1; !result && index >= 0; --index )
    {
        result = ( data[ index ] > comp[ index ] )
                 - ( data[ index ] < comp[ index ] );
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
CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator~() const
{
    uint256_t ret;
    for( std::uint8_t index = 0; index < UINT256_SIZE_IN_BYTES; ++index )
    {
        ret[ index ] = ~data[ index ];
    }
    return ret;
}
CUDA_CALLABLE_MEMBER const std::uint32_t& uint256_t::operator[]( std::uint8_t idx ) const
{
    return data[ idx ];
}
CUDA_CALLABLE_MEMBER std::uint32_t& uint256_t::operator[]( std::uint8_t idx )
{
    return data[ idx ];
}
CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator&( const uint256_t& comp ) const
{
    uint256_t ret;
    for( std::uint8_t index = 0; index < UINT256_SIZE_IN_BYTES; ++index )
    {
        ret[ index ] = comp[ index ] & data[ index ];
    }
    return ret;
}
CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator^( const uint256_t& comp ) const
{
    uint256_t ret;
    for( std::uint8_t index = 0; index < UINT256_SIZE_IN_BYTES; ++index )
    {
        ret[ index ] = comp[ index ] ^ data[ index ];
    }
    return ret;
}
CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator|( const uint256_t& comp ) const
{
    uint256_t ret;
    for( std::uint8_t index = 0; index < UINT256_SIZE_IN_BYTES; ++index )
    {
        ret[ index ] = comp[ index ] | data[ index ];
    }
    return ret;
}
CUDA_CALLABLE_MEMBER bool uint256_t::operator==( const uint256_t& comp ) const
{
    bool ret = true;
    for( uint8_t byte = 0; byte < UINT256_SIZE_IN_BYTES; ++byte )
    {
        ret = ret && ( data[ byte ] == comp[ byte ] );
    }
    return ret;
}
CUDA_CALLABLE_MEMBER void uint256_t::operator=( const uint256_t& set )
{
    for( std::uint8_t a = 0; a < UINT256_SIZE_IN_BYTES; ++a )
    {
        *(data + a) = *(set.data + a);
    }
}
CUDA_CALLABLE_MEMBER bool uint256_t::operator!=( const uint256_t& comp ) const
{
    return !( *this == comp );
}
__device__ void uint256_t::neg( uint256_t& dest ) const
{
    (~(*this)).add( dest, UINT256_ONE );
}
__device__ uint256_t uint256_t::operator-() const
{
    uint256_t tmp;
    neg( tmp );
    return tmp;
}
__device__ bool uint256_t::add( uint256_t& dest, const uint256_t augend ) const
{
    asm ("add.cc.u32      %0, %8, %16;\n\t"
         "addc.cc.u32     %1, %9, %17;\n\t"
         "addc.cc.u32     %2, %10, %18;\n\t"
         "addc.cc.u32     %3, %11, %19;\n\t"
         "addc.cc.u32     %4, %12, %20;\n\t"
         "addc.cc.u32     %5, %13, %21;\n\t"
         "addc.cc.u32     %6, %14, %22;\n\t"
         "addc.u32        %7, %15, %23;\n\t"
         : "=r"(dest.data[ 0 ]), "=r"(dest.data[ 1 ]), "=r"(dest.data[ 2 ]),   
           "=r"(dest.data[ 3 ]), "=r"(dest.data[ 4 ]), "=r"(dest.data[ 5 ]),   
           "=r"(dest.data[ 6 ]), "=r"(dest.data[ 7 ])
         : "r"(this->data[ 0 ]), "r"(this->data[ 1 ]), "r"(this->data[ 2 ]),   
           "r"(this->data[ 3 ]), "r"(this->data[ 4 ]), "r"(this->data[ 5 ]),   
           "r"(this->data[ 6 ]), "r"(this->data[ 7 ]),
           "r"(augend.data[ 0 ]), "r"(augend.data[ 1 ]), "r"(augend.data[ 2 ]),   
           "r"(augend.data[ 3 ]), "r"(augend.data[ 4 ]), "r"(augend.data[ 5 ]),   
           "r"(augend.data[ 6 ]), "r"(augend.data[ 7 ])
        );

    return dest.data[ 7 ] < this->data[ 7 ];
}
__device__ uint256_t uint256_t::operator+( const uint256_t& other ) const
{
    uint256_t ret;
    add( ret, other );
    return ret;
}
CUDA_CALLABLE_MEMBER std::uint32_t uint256_t::at( int loc )
{
    return data[ loc ];
}
// shifts are suspect for hidden bugs
CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator>>( int shift ) const
{
    uint256_t ret;

    std::uint8_t limb_shifts  = shift / UINT256_LIMB_SIZE;
    std::uint8_t shift_length = shift % UINT256_LIMB_SIZE;

    std::uint8_t byte = 0;

    for( byte = limb_shifts; byte < UINT256_SIZE_IN_BYTES; ++byte )
        {
            ret[ byte - limb_shifts ] = data[ byte ];
        }

    // leading limbs are already zero

    for( byte = 0; byte < UINT256_SIZE_IN_BYTES - 1; ++byte )
        {
            ret[ byte ] = ( ret.at( byte ) >> shift_length
             | ( ret.at( byte + 1 ) << ( UINT256_LIMB_SIZE - shift_length ) ) );
        }

    ret[ UINT256_SIZE_IN_BYTES - 1 ] >>= shift;

    return ret;
}
CUDA_CALLABLE_MEMBER uint256_t uint256_t::operator<<( int shift ) const
{
    uint256_t ret;

    std::uint8_t limb_shifts  = shift / UINT256_LIMB_SIZE;
    std::uint8_t shift_length = shift % UINT256_LIMB_SIZE;

    std::uint8_t byte = 0;

    for( byte = 0; byte < UINT256_SIZE_IN_BYTES - limb_shifts; ++byte )
        {
            ret[ byte + limb_shifts ] = data[ byte ];
        }

    // trailing limbs are already zero

    for( byte = UINT256_SIZE_IN_BYTES - 1; byte > 0; --byte )
        {
            ret[ byte ] = ret.at( byte ) << shift_length
               | ( ret.at( byte - 1 ) >> ( UINT256_LIMB_SIZE - shift_length ) );
        }
    ret[ 0 ] <<= shift_length;
    
    return ret;

}




/* Methods below are NOT used in this program (currently) */
/* ------------------------------------------------------ */
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
// begin here
CUDA_CALLABLE_MEMBER uint256_data_t& uint256_t::get_data()
{
    return data;
}
CUDA_CALLABLE_MEMBER std::uint32_t *uint256_t::get_data_ptr()
{
    return data;
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
CUDA_CALLABLE_MEMBER void uint256_t::to_32_bit_arr( std::uint32_t* dest )
{
    memcpy( dest, &(data), 32 );
}





