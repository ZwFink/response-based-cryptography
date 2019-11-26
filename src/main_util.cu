// utility file for host driver funciton (main)

#include "main_util.h"

__global__ void kernel_rbc_engine( uint256_t *key_for_encryp,
                                   size_t first_mismatch,
                                   size_t last_mismatch,
                                   const aes_per_round::message_128 *user_id,
                                   const aes_per_round::message_128 *auth_cipher,
                                   const size_t key_sz_bytes,
                                   const size_t key_sz_bits
                                 )
{
    unsigned int tid = threadIdx.x + ( blockIdx.x * blockDim.x );

    uint256_t starting_perm, ending_perm;

    size_t mismatch   = 0;
    uint64_t num_keys = 0;
    int result        = 0;

    num_keys = get_bin_coef( key_sz_bits, mismatch ); 
   
    // only run thread if tid is less than cardinality of current keyspace
    if( tid < num_keys )
    {
        get_perm_pair( &starting_perm, 
                       &ending_perm, 
                       (size_t) tid, 
                       (size_t) NBLOCKS*BLOCKSIZE,
                       mismatch,
                       key_sz_bytes,
                       key_sz_bits
                     );
        
        result = validator( key_for_encryp,
                            &starting_perm,
                            &ending_perm,
                            user_id,
                            auth_cipher
                          );

        // if result is 1 then we found a key matching client's private key
        // signal all threads to stop
    }

}

__device__ int validator( uint256_t *starting_perm,
                          uint256_t *ending_perm,
                          uint256_t *key_for_encryp,
                          const aes_per_round::message_128 *user_id,
                          const aes_per_round::message_128 *auth_cipher 
                        )
{
    aes_per_round::message_128 encrypted;
    int idx = 0;

    for( idx = 0; idx < 4; ++idx )
        {
            ((uint32_t*)&(encrypted.bits))[ idx ] = 0;
        }

    #ifdef USE_SMEM
    __shared__ std::uint8_t sbox[ SBOX_SIZE_IN_BYTES ];
    if( threadIdx.x < SBOX_SIZE_IN_BYTES )
        {
            sbox[ threadIdx.x ] = Tsbox_256[ threadIdx.x ];
        }

    __syncthreads();

    #else
    // just get a reference to it
    uint8_t *sbox = Tsbox_256;
    #endif 

    uint256_iter iter ( *key_for_encryp,
                        *starting_perm,
                        *ending_perm
                      );
    while( !iter.end() )
        {

            // encrypt
            aes_per_round::roundwise_encrypt( &encrypted,
                                              key_for_encryp,
                                              user_id,
                                              sbox
                                            );

            // check for match! 


            // get next key


            for( idx = 0; idx < 4; ++idx )
                {
                    ((uint32_t*)&(encrypted.bits))[ idx ] = 0;
                }

        }

}

void warm_up_gpu( int device )
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

    printf("\nDevice: %d\n",device);

    for(int i = 0; i < D.size(); i++) 
        std::cout << " D[" << i << "] = " << D[i]; 


    // empty the vector
    D.clear();

    // deallocate any capacity which may currently be associated with vec
    D.shrink_to_fit();

    printf("\n");

    return;
}


