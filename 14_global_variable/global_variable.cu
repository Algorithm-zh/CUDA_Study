#include <cuda_runtime.h>
#include <stdio.h>
__device__ float devData;//全局内存
/*
 *CUDA中每个线程都有自己的私有的本地内存；
  线程块有自己的共享内存，对线程块内所有线程可见；
  所有线程都能访问读取常量内存和纹理内存，但是不能写，因为他们是只读的；
  全局内存，常量内存和纹理内存空间有不同的用途。对于一个应用来说，全局内存，常量内存和纹理内存有相同的生命周期
 */
__global__ void checkGlobalVariable()
{
    printf("Device: The value of the global variable is %f\n",devData);
    devData+=2.0;
}
int main()
{
    float value=3.14f;
    cudaMemcpyToSymbol(devData,&value,sizeof(float));//静态变量拷贝
    printf("Host: copy %f to the global variable\n",value);
    checkGlobalVariable<<<1,1>>>();
    cudaMemcpyFromSymbol(&value,devData,sizeof(float));
    printf("Host: the value changed by the kernel to %f \n",value);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
