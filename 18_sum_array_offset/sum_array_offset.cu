#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/freshman.h"


void sumArrays(float * a,float * b,float * res,int offset,const int size)
{

    for(int i=0,k=offset;k<size;i++,k++)
    {
        res[i]=a[k]+b[k];
    }

}
/*
 * 不启用一级缓存的缓存粒度是32字节，启用后是128字节，也就是一次读取的内存字节数目，
 * 当一级缓存被禁用的时候，对全局内存的加载请求直接进入二级缓存，如果二级缓存缺失，则由DRAM完成请求
 * 内存访问特点：
 *    是否使用缓存：一级缓存是否介入加载过程
 *    对齐与非对齐的：如果访问的第一个地址是32的倍数
 *    合并与非合并，访问连续数据块则是合并的
 * 非对齐内存读取实例
 */
__global__ void sumArraysGPU(float*a,float*b,float*res,int offset,int n)
{
  //int i=threadIdx.x;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int k=i+offset;
  if(k<n)
    res[i]=a[k]+b[k];
}
__global__ void copyKernel(float * in,float* out)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    out[idx]=__ldg(&in[idx]);//__ldg是使用只读缓存来进行读取(代替一级缓存)
}
/*
 * 内存的写入和读取（或者叫做加载）是完全不同的，并且写入相对简单很多。
 * 一级缓存不能用在 Fermi 和 Kepler GPU上进行存储操作，发送到设备前，只经过二级缓存，
 * 存储操作在32个字节的粒度上执行，内存事物也被分为一段两端或者四段，
 * 如果两个地址在一个128字节的段内但不在64字节范围内，则会产生一个四段的事务，其他情况以此类推
 */
int main(int argc,char **argv)
{
  int dev = 0;
  cudaSetDevice(dev);

  int nElem=1<<18;
  int offset=0;
  if(argc>=2)
    offset=atoi(argv[1]);
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *a_h=(float*)malloc(nByte);
  float *b_h=(float*)malloc(nByte);
  float *res_h=(float*)malloc(nByte);
  float *res_from_gpu_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  memset(res_from_gpu_h,0,nByte);

  float *a_d,*b_d,*res_d;
  CHECK(cudaMalloc((float**)&a_d,nByte));
  CHECK(cudaMalloc((float**)&b_d,nByte));
  CHECK(cudaMalloc((float**)&res_d,nByte));
  CHECK(cudaMemset(res_d,0,nByte));
  initialData(a_h,nElem);
  initialData(b_h,nElem);

  CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

  dim3 block(1024);
  dim3 grid(nElem/block.x);
  double iStart,iElaps;
  iStart=cpuSecond();
  sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d,offset,nElem);
  cudaDeviceSynchronize();
  iElaps=cpuSecond()-iStart;
  CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
  printf("Execution configuration<<<%d,%d>>> Time elapsed %f sec --offset:%d \n",grid.x,block.x,iElaps,offset);


  sumArrays(a_h,b_h,res_h,offset,nElem);

  checkResult(res_h,res_from_gpu_h,nElem);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(a_h);
  free(b_h);
  free(res_h);
  free(res_from_gpu_h);

  return 0;
}
