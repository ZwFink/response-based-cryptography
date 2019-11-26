#ifndef CUDA_UTILS_HH_INCLUDED
#define CUDA_UTILS_HH_INCLUDED

namespace cuda_utils
{
    template<class T>
        cudaError_t HtoD( T *dest, T *src, std::size_t bytes )
    {
        cudaError_t err = cudaMemcpy( dest, src, bytes, cudaMemcpyHostToDevice );
        return err;
    }

    template<class T>
        cudaError_t DtoH( T *dest, T *src, std::size_t bytes )
    {
        cudaError_t err = cudaMemcpy( dest, src, bytes, cudaMemcpyDeviceToHost );
        return err;
    }

    template<class T>
        cudaError_t HtoD( T *dest, T *src )
    {
        cudaError_t err = cudaMemcpy( dest, src, sizeof( T ), cudaMemcpyHostToDevice );
        return err;
    }

    template<class T>
        cudaError_t DtoH( T *dest, T *src )
    {
        cudaError_t err = cudaMemcpy( dest, src, sizeof( T ), cudaMemcpyDeviceToHost );
        return err;
    }


}; // namespace cuda_utils

#endif // CUDA_UTILS_HH_INCLUDED
