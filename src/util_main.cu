// utility file for host driver funciton (main)

#include "util_main.h"
#include "aes_tables.h"
#include "util.h"
#include <stdio.h>

__global__ void kernel_startend_perms( uint256_t *starting_perm,
                                       uint256_t *ending_perm,
                                       const int mismatch,
                                       const std::size_t num_blocks,
                                       const std::size_t threads_per_block,
                                       const std::size_t keys_per_thread,
                                       const std::uint32_t extra_keys,
                                       const uint64_t offset,
                                       const uint64_t bound,
                                       const int key_size_bits,
                                       int *found_key_Flag
                                     )
{
    unsigned int tid = threadIdx.x + ( blockIdx.x * blockDim.x ) + offset;

    (starting_perm + tid)->set_all(0);
    (ending_perm + tid)->set_all(0);

    if( tid < bound && !(EARLY_EXIT && *found_key_Flag) )
    {
        get_perm_pair( starting_perm + tid, 
                       ending_perm + tid, 
                       (uint64_t) tid, 
                       (uint64_t) num_blocks * threads_per_block,
                       mismatch,
                       keys_per_thread,
                       extra_keys,
                       key_size_bits
                     );
    }

}

__global__ void kernel_iterate_keyspace( uint256_t *key_for_encryp,
                                         uint256_t *key_to_find,
                                         const int mismatch,
                                         const uint *user_id,
                                         const uint *auth_cipher,
                                         const std::size_t num_blocks,
                                         const std::size_t threads_per_block,
                                         const std::size_t keys_per_thread,
                                         const std::uint64_t num_keys,
                                         const std::uint32_t extra_keys,
                                         std::uint64_t *iter_count,
                                         const uint64_t offset,
                                         const uint64_t bound,
                                         const int key_size_bits,
                                         int *found_key_Flag,
                                         uint256_t *starting_perm,
                                         uint256_t *ending_perm
                                       )
{
    if( EARLY_EXIT && *found_key_Flag == 1 ) return;
    
    unsigned int tid = threadIdx.x + ( blockIdx.x * blockDim.x ) + offset;

    aes_tables tabs;
    
#ifdef USE_SMEM
    __shared__ std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ];
    if( threadIdx.x < SBOX_SIZE_IN_BYTES )
        {
#if THREADS_PER_BLOCK == 128 

            sbox[ 2 * threadIdx.x ] = Tsbox_256[ 2 * threadIdx.x ];
            sbox[ ( 2 * threadIdx.x ) + 1 ] = Tsbox_256[ ( 2 * threadIdx.x ) + 1 ];

#elif THREADS_PER_BLOCK == 256

            sbox[ threadIdx.x ] = Tsbox_256[ threadIdx.x ];

#endif
        }
    __shared__ uint Te0[256], Te1[256], Te2[256], Te3[256];
    load_smem(Te0, cTe0, Te1, cTe1, Te2, cTe2, Te3, cTe3);
    // NOTE: __syncthreads not used here because it's called in
    // util::load_smem

    tabs.Te0 = Te0;
    tabs.Te1 = Te1;
    tabs.Te2 = Te2;
    tabs.Te3 = Te3;

#else
    // just get a reference to it
    uint8_t *sbox = Tsbox_256;

    tabs.Te0 = cTe0;
    tabs.Te1 = cTe1;
    tabs.Te2 = cTe2;
    tabs.Te3 = cTe3;
#endif 

    if( tid < bound && !(EARLY_EXIT && *found_key_Flag) )
    {
        tabs.sbox = sbox;
        std::uint8_t idx = 0;
        std::uint8_t match = 0;
        int total = 0;
        uint cyphertext[ 4 ] = {0,0,0,0};

        uint256_t * start_perm = starting_perm + tid;
        uint256_t * end_perm   = ending_perm + tid;

        uint256_iter iter ( *key_for_encryp,
                            *start_perm,
                            *end_perm
                          );
        
        while( !iter.end() && !(EARLY_EXIT && (total%ITERCOUNT)==0 && *found_key_Flag) )
            {

                ++total;
                // encrypt
                aes_gpu::encrypt( user_id,
                                  cyphertext,
                                  (uint*)(iter.corrupted_key.data),
                                  &tabs
                                );

                // check for match! 
                for( idx = 0; idx < 4; ++idx )
                    {
                        match += ( cyphertext[ idx ] == auth_cipher[ idx ] );
                    }

                if( match == 4 )
                    {
                        *key_to_find = iter.corrupted_key;

                        if( EARLY_EXIT )
                            atomicAdd( (unsigned long long int*) found_key_Flag, 1 );
                    }

                match = 0;

                // get next key
                iter.next();

            }

        atomicAdd( (unsigned long long int*) iter_count, total );
        
    }

}

__host__ __device__ uint bytes_to_int( const std::uint8_t *bytes )
{
    uint ret_val;
    ret_val =  ((int)bytes[3] << 24) | ((int)bytes[2] << 16) | ((int)bytes[1] << 8) | ((int)bytes[0]);
    return ret_val;
}

void warm_up_gpu( int device, int verbose )
{
    cudaSetDevice( device ); 		
    // initialize all ten integers of a device_vector to 1 
    thrust::device_vector<int> D(10, 1); 
    // set the first seven elements of a vector to 9 
    thrust::fill(D.begin(), D.begin() + 7, 9); 
    // initialize a host_vector with the first five elements of D 
    thrust::host_vector<int> H(D.begin(), D.begin() + 5); 
    // set the elements of H to 0, 1, 2, 3, ... 
    thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D 
    thrust::copy(H.begin(), H.end(), D.begin()); 
    // print D 

    if( verbose ) printf("\nDevice: %d\n",device);

    for(int i = 0; i < D.size(); i++) 
        if( verbose ) std::cout << " D[" << i << "] = " << D[i]; 


    // empty the vector
    D.clear();

    // deallocate any capacity which may currently be associated with vec
    D.shrink_to_fit();

    if( verbose ) printf("\n");

    return;
}


