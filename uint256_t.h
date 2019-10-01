#ifndef UINT246_T_HH_INCLUDED
#define UINT246_T_HH_INCLUDED
#define UINT256_SIZE_IN_BYTES 32

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER 
#endif

class uint256_t
{
 public:
    CUDA_CALLABLE_MEMEBER uint256_t();
    CUDA_CALLABLE_MEMEBER ~uint256_t();


 private:
    CUDA_CALLABLE_MEMEBER uint8_t[ UINT256_SIZE_IN_BYTES ] data;
}


#endif // UINT246_T_HH_INCLUDED
