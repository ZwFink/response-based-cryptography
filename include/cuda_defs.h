#ifndef CUDA_DEFS_HH_INCLUDED
#define CUDA_DEFS_HH_INCLUDED

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER 
#endif

#define INLINE __forceinline__


#endif // CUDA_DEFS_HH_INCLUDED
