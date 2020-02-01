#ifndef MAIN_UTIL_HH_INCLUDED
#define MAIN_UTIL_HH_INCLUDED

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include <thread>
#include <cstdint>
#include <utility>


// thrust inclusions
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "perm_util.h"
#include "uint256_iterator.h"
#include "aes_per_round.h"
#include "sbox.h"



void warm_up_gpu( int device );

__device__ int validator( uint256_t *starting_perm,
                          uint256_t *ending_perm,
                          uint256_t *key_for_encryp,
                          uint256_t *key_to_find,
                          const aes_per_round::message_128 *user_id,
                          const aes_per_round::message_128 *auth_cipher 
                        );

__global__ void kernel_rbc_engine( uint256_t *key_for_encryp,
                                   uint256_t *key_to_find,
                                   const int mismatch,
                                   const aes_per_round::message_128 *user_id,
                                   const aes_per_round::message_128 *auth_cipher,
                                   const std::size_t key_sz_bits,
                                   const std::size_t num_blocks,
                                   const std::size_t threads_per_block,
                                   const std::size_t keys_per_thread,
                                   std::uint64_t num_keys,
                                   std::uint64_t extra_keys,
                                   std::uint64_t *iter_count
                                 );

#endif // MAIN_UTIL_HH_INCLUDED

