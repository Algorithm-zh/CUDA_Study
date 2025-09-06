#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/freshman.h"



void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}
__global__ void sumArraysGPU(float*a,float*b,float*res,int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i < N)
    res[i]=a[i]+b[i];
}
int main(int argc,char **argv)
{
  // set up device
  initDevice(0);

  int nElem=1<<24;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *res_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  //memset(res_from_gpu_h,0,nByte);

  /*
   * 统一内存的基本思路就是减少指向同一个地址的指针，比如我们经常见到的，
   * 在本地分配内存，然后传输到设备，然后在从设备传输回来，
   * 使用统一内存，就没有这些显式的需求了，而是驱动程序帮我们完成。
   * 使用cudaMallocManaged分配内存，可以直接在设备端和主机端访问(内部是他自己调用了copy,这只是封装起来了)
   * 虽然统一内存管理给我们写代码带来了方便而且速度也很快，但是实验表明，手动控制还是要优于统一内存管理，换句话说，人脑的控制比编译器和目前的设备更有效
   */
  float *a_d,*b_d,*res_d;
  CHECK(cudaMallocManaged((float**)&a_d,nByte));
  CHECK(cudaMallocManaged((float**)&b_d,nByte));
  CHECK(cudaMallocManaged((float**)&res_d,nByte));

  initialData(a_d,nElem);
  initialData(b_d,nElem);

  //CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
  //CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

  dim3 block(512);
  dim3 grid((nElem-1)/block.x+1);

  double iStart,iElaps;
  iStart=cpuSecond();
  sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d,nElem);
  cudaDeviceSynchronize();
  iElaps=cpuSecond()-iStart;
  printf("Execution configuration<<<%d,%d>>> Time elapsed %f sec\n",grid.x,block.x,iElaps);

  //CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
  sumArrays(b_d,b_d,res_h,nElem);

  checkResult(res_h,res_d,nElem);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(res_h);

  return 0;
}
