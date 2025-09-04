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
__global__ void sumArraysGPU(float*a,float*b,float*res)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  res[i]=a[i]+b[i];
}
int main(int argc,char **argv)
{
  int dev = 0;
  cudaSetDevice(dev);
  int power=10;
  if(argc>=2)
    power=atoi(argv[1]);
  int nElem=1<<power;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *res_from_gpu_h=(float*)malloc(nByte);
  float *res_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  memset(res_from_gpu_h,0,nByte);

  float *a_host,*b_host,*res_d;
  double iStart,iElaps;
  dim3 block(1024);
  dim3 grid(nElem/block.x);
  res_from_gpu_h=(float*)malloc(nByte);
  float *a_dev,*b_dev;
  //flags的选值为：
  /*
   * cudaHostAllocDefalt 选这个时相当于cudaMallocHost函数,即创建固定内存
   * cudaHostAllocPortable 返回能被所有CUDA上下文使用的固定内存
   * cudaHostAllocWriteCombined  返回写结合内存，在某些设备上这种内存传输效率更高
   * cudaHostAllocMapped  产生零拷贝内存
   */
  CHECK(cudaHostAlloc((float**)&a_host,nByte,cudaHostAllocMapped));
  CHECK(cudaHostAlloc((float**)&b_host,nByte,cudaHostAllocMapped));
  CHECK(cudaMalloc((float**)&res_d,nByte));
  initialData(a_host,nElem);
  initialData(b_host,nElem);

 //=============================================================//
  iStart = cpuSecond();
  /*
   * 零拷贝内存虽然不需要显式的传递到设备上，但是设备还不能通过pHost直接访问对应的内存地址，
   * 设备需要访问主机上的零拷贝内存，需要先获得另一个地址，这个地址帮助设备访问到主机对应的内存 
   * pDevice就是设备上访问主机零拷贝内存的指针
   */
  CHECK(cudaHostGetDevicePointer((void**)&a_dev,(void*) a_host,0));
  CHECK(cudaHostGetDevicePointer((void**)&b_dev,(void*) b_host,0));
  sumArraysGPU<<<grid,block>>>(a_dev,b_dev,res_d);
  CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
  iElaps = cpuSecond() - iStart;
 //=============================================================//
  printf("zero copy memory elapsed %lf ms \n", iElaps);
  printf("Execution configuration<<<%d,%d>>>\n",grid.x,block.x);
//-----------------------normal memory---------------------------
  float *a_h_n=(float*)malloc(nByte);
  float *b_h_n=(float*)malloc(nByte);
  float *res_h_n=(float*)malloc(nByte);
  float *res_from_gpu_h_n=(float*)malloc(nByte);
  memset(res_h_n,0,nByte);
  memset(res_from_gpu_h_n,0,nByte);

  float *a_d_n,*b_d_n,*res_d_n;
  CHECK(cudaMalloc((float**)&a_d_n,nByte));
  CHECK(cudaMalloc((float**)&b_d_n,nByte));
  CHECK(cudaMalloc((float**)&res_d_n,nByte));

  initialData(a_h_n,nElem);
  initialData(b_h_n,nElem);
//=============================================================//
  iStart = cpuSecond();
  CHECK(cudaMemcpy(a_d_n,a_h_n,nByte,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d_n,b_h_n,nByte,cudaMemcpyHostToDevice));
  sumArraysGPU<<<grid,block>>>(a_d_n,b_d_n,res_d_n);
  CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
  iElaps = cpuSecond() - iStart;
//=============================================================//
  printf("device memory elapsed %lf ms \n", iElaps);
  printf("Execution configuration<<<%d,%d>>>\n",grid.x,block.x);
//--------------------------------------------------------------------

  /*
   *
      Vector size:1024
      zero copy memory elapsed 0.000339 ms 
      Execution configuration<<<1,1024>>>
      device memory elapsed 0.000034 ms 
      Execution configuration<<<1,1024>>>
      Check result success!

      Vector size:1024
      zero copy memory elapsed 0.000371 ms 
      Execution configuration<<<1,1024>>>
      device memory elapsed 0.000054 ms 
      Execution configuration<<<1,1024>>>
      Check result success!

      Vector size:1024
      zero copy memory elapsed 0.000346 ms 
      Execution configuration<<<1,1024>>>
      device memory elapsed 0.000104 ms 
      Execution configuration<<<1,1024>>>
      Check result success!
      总结来看就是零拷贝内存比设备主存储器更慢
      但是在一些gpu和cpu集成设备上，他们的物理内存是公用的，零拷贝内存会很有效,而离散设备，通过pcie连接的，就很慢
   */
  sumArrays(a_host,b_host,res_h,nElem);
  checkResult(res_h,res_from_gpu_h,nElem);

  cudaFreeHost(a_host);
  cudaFreeHost(b_host);
  cudaFree(res_d);
  free(res_h);
  free(res_from_gpu_h);

  cudaFree(a_d_n);
  cudaFree(b_d_n);
  cudaFree(res_d_n);

  free(a_h_n);
  free(b_h_n);
  free(res_h_n);
  free(res_from_gpu_h_n);
  return 0;
}
