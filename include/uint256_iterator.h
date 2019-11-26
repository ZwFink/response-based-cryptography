#ifndef UINT256_ITERATOR_HH_INCLUDED
#define UINT256_ITERATOR_HH_INCLUDED
#include "uint256_t.h"
#include "cuda_defs.h"

class uint256_iter
{
 public:

    CUDA_CALLABLE_MEMBER uint256_iter( const uint256_t& key,
                                       const uint256_t& first_perm,
                                       const uint256_t& final_perm
                                     );

    __device__ void next();
    __device__ bool end();

    uint256_t curr_perm;
    uint256_t last_perm;
    uint256_t t;
    uint256_t tmp;
    uint256_t key_uint;
    uint256_t corrupted_key;
    bool overflow;
};

#endif // UINT256_ITERATOR_HH_INCLUDED
