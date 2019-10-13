#ifndef UINT256_T_HH_INCLUDED
#define UINT256_T_HH_INCLUDED
#define UINT256_SIZE_IN_BYTES 32
#define UINT256_SIZE_IN_BITS 256
#define UINT256_LIMB_SIZE 8
#define UINT256_MAX_INT uint256_t( 0xFF );
#define UINT256_ZERO uint256_t( 0x00 );

#include <cstddef>
#include <string>
#include <x86intrin.h>
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER 
#endif

#define INLINE __forceinline__

typedef std::uint8_t uint256_data_t[ UINT256_SIZE_IN_BYTES ];

class uint256_t
{
 public:
    CUDA_CALLABLE_MEMBER uint256_t();
    CUDA_CALLABLE_MEMBER uint256_t( std::uint8_t set );
    CUDA_CALLABLE_MEMBER ~uint256_t() = default;

    CUDA_CALLABLE_MEMBER uint256_t operator&( uint256_t comp );
    CUDA_CALLABLE_MEMBER uint256_t operator|( uint256_t comp );
    CUDA_CALLABLE_MEMBER uint256_t operator^( uint256_t comp );

    CUDA_CALLABLE_MEMBER uint256_t operator<<( int shift );

    CUDA_CALLABLE_MEMBER uint256_t operator~();

    CUDA_CALLABLE_MEMBER std::uint8_t& operator[]( std::uint8_t idx );
    CUDA_CALLABLE_MEMBER std::uint8_t at( int loc );

    CUDA_CALLABLE_MEMBER bool operator==( uint256_t comp );
    CUDA_CALLABLE_MEMBER bool operator!=( uint256_t comp );

    CUDA_CALLABLE_MEMBER uint256_data_t& get_data();

    CUDA_CALLABLE_MEMBER void set_all( std::uint8_t value );

    __host__ void dump();

    __device__ int ctz();
    __device__ int popc();


 private:
    std::uint8_t data[ UINT256_SIZE_IN_BYTES ];
};


#endif // UINT256_T_HH_INCLUDED
