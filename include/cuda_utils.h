#ifndef CUDA_UTILS_HH_INCLUDED
#define CUDA_UTILS_HH_INCLUDED
#include <cstddef>

namespace cuda_utils
{
    template<typename T, typename V>
        cudaError_t HtoD( T *dest, const V *src, std::size_t bytes )
    {
        cudaError_t err = cudaMemcpy( dest, src, bytes, cudaMemcpyHostToDevice );
        return err;
    }

    template<typename T, typename V>
        cudaError_t DtoH( T *dest, const V *src, std::size_t bytes )
    {
        cudaError_t err = cudaMemcpy( dest, src, bytes, cudaMemcpyDeviceToHost );
        return err;
    }

}; // namespace cuda_utils

#endif // CUDA_UTILS_HH_INCLUDED
