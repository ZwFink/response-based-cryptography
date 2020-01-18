#ifndef UINT256_T_HH_INCLUDED
#define UINT256_T_HH_INCLUDED
#define UINT256_SIZE_IN_BYTES 32
#define UINT256_SIZE_IN_BITS 256
#define UINT256_LIMB_SIZE 8
#define UINT256_MAX_INT uint256_t( 0xFF )
#define UINT256_ZERO uint256_t( 0x00 )
#define UINT256_ONE uint256_t( 0x01, 0 )
#define UINT256_NEGATIVE_ONE UINT256_MAX_INT

#include <cstddef>
#include <string>
#include <x86intrin.h>
#include <cstdint>

#include "cuda_defs.h"

typedef std::uint8_t uint256_data_t[ UINT256_SIZE_IN_BYTES ];

namespace uint256_ctz_table
{
    extern std::uint8_t lookup[ 37 ];

    INLINE DEVICE_ONLY int ctz( const std::uint32_t loc );
}

class uint256_t
{
 public:
    CUDA_CALLABLE_MEMBER uint256_t();
    CUDA_CALLABLE_MEMBER uint256_t( std::uint8_t set );

    // set the member at index to set
    CUDA_CALLABLE_MEMBER uint256_t( std::uint8_t set, std::uint8_t index );
    CUDA_CALLABLE_MEMBER ~uint256_t() = default;
	 CUDA_CALLABLE_MEMBER void copy( const uint256_t& copied );
    // copy 64 bit integer into member starting at index
    CUDA_CALLABLE_MEMBER void copy_64( uint64_t ref, uint8_t index );
    CUDA_CALLABLE_MEMBER void set( std::uint8_t set, std::uint8_t index );

    CUDA_CALLABLE_MEMBER void from_string( const unsigned char *string );

    CUDA_CALLABLE_MEMBER uint256_t operator&( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER uint256_t operator|( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER uint256_t operator^( const uint256_t& comp ) const;

    CUDA_CALLABLE_MEMBER uint256_t operator<<( int shift ) const;
    CUDA_CALLABLE_MEMBER uint256_t operator>>( int shift ) const;

    CUDA_CALLABLE_MEMBER uint256_t operator~() const;

    CUDA_CALLABLE_MEMBER std::uint8_t& operator[]( std::uint8_t idx );
    CUDA_CALLABLE_MEMBER const std::uint8_t& operator[]( std::uint8_t idx ) const;
    CUDA_CALLABLE_MEMBER std::uint8_t at( int loc );

    CUDA_CALLABLE_MEMBER bool operator==( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER bool operator!=( const uint256_t& comp ) const;

    CUDA_CALLABLE_MEMBER int compare( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER bool operator<( const uint256_t& comp ) const;
    CUDA_CALLABLE_MEMBER bool operator>( const uint256_t& comp ) const;

    CUDA_CALLABLE_MEMBER uint256_data_t& get_data();
    CUDA_CALLABLE_MEMBER std::uint8_t *get_data_ptr();

    CUDA_CALLABLE_MEMBER void set_all( std::uint8_t value );

    CUDA_CALLABLE_MEMBER void operator=( const uint256_t& set );
    __host__ void dump();
    __host__ void dump_hex();

    __device__ int ctz();
    __device__ int popc();

    CUDA_CALLABLE_MEMBER void to_32_bit_arr( std::uint32_t* dest );

    CUDA_CALLABLE_MEMBER void set_bit( std::uint8_t bit_idx );

    __device__ bool add( uint256_t& dest,
                         const uint256_t augend
                       ) const;

    // this must be device-only because uint256_t::add is used
    __device__ void neg( uint256_t& dest );


    std::uint8_t data[ UINT256_SIZE_IN_BYTES ];

};

#endif // UINT256_T_HH_INCLUDED
