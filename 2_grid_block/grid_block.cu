#include <cuda_runtime.h>
#include <stdio.h>
int main(int argc,char ** argv)
{
  //一个核心函数对应一个grid
  int nElem=1024;
  //dim3是手动定义的 在设备端是uint3类型，uint3是常类型结构，不能修改
  dim3 block(1024);
  dim3 grid((nElem-1)/block.x+1);
  //grid.x = 1, block.x = 1024, grid里有一个block,这个block里有1024个线程
  printf("grid.x %d block.x %d\n",grid.x,block.x);

  block.x=512;
  grid.x=(nElem-1)/block.x+1;
  //grid.x = 2, block.x = 512, grid里有二个block,block里有512个线程
  printf("grid.x %d block.x %d\n",grid.x,block.x);

  block.x=256;
  grid.x=(nElem-1)/block.x+1;
  //grid.x = 4, block.x = 256, grid里有四个block,block里有256个线程
  printf("grid.x %d block.x %d\n",grid.x,block.x);

  block.x=128;
  grid.x=(nElem-1)/block.x+1;
  //grid.x = 8, block.x = 128, grid里有八个block,block里有128个线程
  printf("grid.x %d block.x %d\n",grid.x,block.x);

  cudaDeviceReset();
  return 0;
}
