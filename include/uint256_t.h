#ifndef UINT256_T_HH_INCLUDED
#define UINT256_T_HH_INCLUDED

#define UINT256_SIZE_IN_BYTES 8
#define UINT256_SIZE_IN_BITS 256
#define UINT256_LIMB_SIZE 32

#define UINT256_MAX_INT uint256_t( 0xFFFFFFFF )
#define UINT256_ZERO uint256_t( 0x00000000 )
#define UINT256_ONE uint256_t( 1, 0 )
#define UINT256_NEGATIVE_ONE UINT256_MAX_INT

#include <cstddef>
#include <string>
#include <x86intrin.h>
#include <cstdint>

#include "cuda_defs.h"

// Convention: data is stored in a 'little-endian' manner
//             - the least significant byte is stored first

typedef std::uint32_t uint256_data_t[ UINT256_SIZE_IN_BYTES ];

namespace uint256_ctz_table
{
    extern std::uint8_t lookup[ 37 ];

    INLINE DEVICE_ONLY int ctz( const std::uint32_t loc );
}

class uint256_t
{
 public:

	CUDA_CALLABLE_MEMBER void copy( const uint256_t& copied );
    CUDA_CALLABLE_MEMBER void copy_64( uint64_t ref, uint8_t index );
    CUDA_CALLABLE_MEMBER uint256_data_t& get_data();
    CUDA_CALLABLE_MEMBER std::uint32_t *get_data_ptr();
    __host__ void dump_hex();
    __device__ int popc();
    CUDA_CALLABLE_MEMBER void to_32_bit_arr( std::uint32_t* dest );


    /* The usage of methods below has been verified in this program */
    
    // Constructors and methods
    CUDA_CALLABLE_MEMBER uint256_t();
    CUDA_CALLABLE_MEMBER ~uint256_t() = default;
    CUDA_CALLABLE_MEMBER uint256_t( std::uint32_t set, std::uint8_t index );
    CUDA_CALLABLE_MEMBER uint256_t( std::uint32_t set );
    CUDA_CALLABLE_MEMBER void set_all( std::uint32_t val );
    CUDA_CALLABLE_MEMBER void set( std::uint32_t set, std::uint8_t index );
    CUDA_CALLABLE_MEMBER void from_string( const unsigned char *string );
    CUDA_CALLABLE_MEMBER void set_bit( std::uint8_t bit_idx );
    __device__ int ctz();
    __host__ void dump();

    // Operators and their associated helper methods
    CUDA_CALLABLE_MEMBER std::uint32_t& operator[]( std::uint8_t idx );
    CUDA_CALLABLE_MEMBER const std::uint32_t& operator[]( std::uint8_t idx ) const;
    CUDA_CALLABLE_MEMBER bool operator==( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER bool operator!=( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER void operator=( const uint256_t& set );
    CUDA_CALLABLE_MEMBER uint256_t operator&( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER uint256_t operator|( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER uint256_t operator^( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER std::uint32_t at( int loc );
    CUDA_CALLABLE_MEMBER uint256_t operator<<( int shift ) const;
    CUDA_CALLABLE_MEMBER uint256_t operator>>( int shift ) const;
    CUDA_CALLABLE_MEMBER uint256_t operator~() const;
    CUDA_CALLABLE_MEMBER int compare( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER bool operator<( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER bool operator>( const uint256_t& comp ) const;
    __device__ bool add( uint256_t& dest, const uint256_t augend ) const;
    __device__ uint256_t operator+( const uint256_t& other ) const;
    // two methods below must be device-only because uint256_t::add is used
    __device__ void neg( uint256_t& dest ) const;
    __device__ uint256_t operator-() const;

    // data storage
    std::uint32_t data[ UINT256_SIZE_IN_BYTES ];
};

#endif // UINT256_T_HH_INCLUDED
