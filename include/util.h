#ifndef UTIL_CUDA_HH_INCLUDED
#define UTIL_CUDA_HH_INCLUDED

#include <type_traits>
#include <cstdint>
// GTX 1080: smem 40x1024, gmem 80x512
// GTX 980: 32x1024
// K40: 30x1024
// TRX 2080: smem 92x1024
#ifndef NUM_BLOCKS
#define NUM_BLOCKS 30
#endif

#ifdef _MSC_VER
#define SWAP(x) (_lrotl(x, 8) & 0x00ff00ff | _lrotr(x, 8) & 0xff00ff00)
#define GETWORD(p) SWAP(*((uint *)(p)))
#define PUTWORD(ct, st) (*((uint *)(ct)) = SWAP((st)))
#else
#define GETWORD(pt) (((uint)(pt)[0] << 24) ^ ((uint)(pt)[1] << 16) ^ ((uint)(pt)[2] <<  8) ^ ((uint)(pt)[3]))
#define PUTWORD(ct, st) ((ct)[0] = (uchar)((st) >> 24), (ct)[1] = (uchar)((st) >> 16), (ct)[2] = (uchar)((st) >>  8), (ct)[3] = (uchar)(st), (st))
#endif

#define MAX_RK_SIZE 182

using uchar = unsigned char;

template<uint N_ELEM,
         typename A
        >
    __device__ void load_smem_arr( A *smem, A *gmem )
{
    const int tid = threadIdx.x;
    static_assert( std::is_same<typename std::is_integral<A>::value, bool>::value,
                   "load_smem_arr can only be called for integral types"
                 );

    if( tid < N_ELEM )
        {
            ((std::uint32_t*)smem)[ tid ]
                = ((std::uint32_t*)gmem)[ tid ];
        }

    __syncthreads();
}

__device__ void load_smem(uchar *st0, uchar *gt0, uchar *st1, uchar *gt1, uchar *st2, uchar *gt2, uchar *st3, uchar *gt3);
__device__ void load_smem(uint *st0, uint *gt0, uint *st1, uint *gt1, uint *st2, uint *gt2, uint *st3, uint *gt3);
#endif // UTIL_HH_INCLUDED
