#ifndef AES_PER_ROUND_HH_INCLUDED
#define AES_PER_ROUND_HH_INCLUDED
#include <cstdint>

#include "uint256_t.h"
#include "cuda_defs.h"

#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))

namespace aes_per_round
{

}; // namespace aes_per_round 

#endif // AES_PER_ROUND_HH_INCLUDED 
