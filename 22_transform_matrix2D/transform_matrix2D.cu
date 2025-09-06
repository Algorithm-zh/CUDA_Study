#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/freshman.h"
//cpu transform
void transformMatrix2D_CPU(float * MatA,float * MatB,int nx,int ny)
{
  for(int j=0;j<ny;j++)
  {
    for(int i=0;i<nx;i++)
    {
      MatB[i*nx+j]=MatA[j*nx+i];
    }
  }
}
//按行读(比按列快的多，因为合并读取)
__global__ void copyRow(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*nx;
    if (ix<nx && iy<ny)
    {
      MatB[idx]=MatA[idx];
    }
}
//按列读
__global__ void copyCol(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx]=MatA[idx];
    }
}
//按行朴素转置
__global__ void transformNaiveRow(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_col]=MatA[idx_row];
    }
}
//按列朴素转置(按列比按行转置快，因为L1会缓存，缓存了的数据就不用在读取)
__global__ void transformNaiveCol(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_row]=MatA[idx_col];
    }
}
//按行展开转置(展开后发现无论行还是列都会比最开始的按行读更快)
__global__ void transformNaiveRowUnroll(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x*4;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_col]=MatA[idx_row];
      //对 B 来说，行号 = ix，往右走 blockDim.x，就是「行号增加了 blockDim.x」：
      MatB[idx_col+ny*1*blockDim.x]=MatA[idx_row+1*blockDim.x];//B[(ix+blockDim.x)∗ny+iy]
      MatB[idx_col+ny*2*blockDim.x]=MatA[idx_row+2*blockDim.x];
      MatB[idx_col+ny*3*blockDim.x]=MatA[idx_row+3*blockDim.x];
    }
}
//按列展开转置
__global__ void transformNaiveColUnroll(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x*4;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
        MatB[idx_row]=MatA[idx_col];
        MatB[idx_row+1*blockDim.x]=MatA[idx_col+ny*1*blockDim.x];
        MatB[idx_row+2*blockDim.x]=MatA[idx_col+ny*2*blockDim.x];
        MatB[idx_row+3*blockDim.x]=MatA[idx_col+ny*3*blockDim.x];
    }
}
//对角读取，使用 f(x,y)=(m,n)函数来将坐标打乱，是为了打乱线程块，防止连续的线程块访问相近的DRAM，
//因为DRAM过多的访问同一个分区(256个字节一个分区)的话可能会出现排队的情况，所以让他分散开,.调整块的ID
__global__ void transformNaiveRowDiagonal(float * MatA,float * MatB,int nx,int ny)
{
    int block_y=blockIdx.x;
    int block_x=(blockIdx.x+blockIdx.y)%gridDim.x;
    int ix=threadIdx.x+blockDim.x*block_x;
    int iy=threadIdx.y+blockDim.y*block_y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_col]=MatA[idx_row];
    }
}
__global__ void transformNaiveColDiagonal(float * MatA,float * MatB,int nx,int ny)
{
    int block_y=blockIdx.x;
    int block_x=(blockIdx.x+blockIdx.y)%gridDim.x;
    int ix=threadIdx.x+blockDim.x*block_x;
    int iy=threadIdx.y+blockDim.y*block_y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_row]=MatA[idx_col];
    }
}



int main(int argc,char** argv)
{
  printf("strating...\n");
  initDevice(0);
  int nx=1<<12;
  int ny=1<<12;
  int dimx=32;
  int dimy=32;
  int nxy=nx*ny;
  int nBytes=nxy*sizeof(float);
  int transform_kernel=0;
  if(argc==2)
    transform_kernel=atoi(argv[1]);
  if(argc>=4)
  {
      transform_kernel=atoi(argv[1]);
      dimx=atoi(argv[2]);
      dimy=atoi(argv[3]);
  }

  //Malloc
  float* A_host=(float*)malloc(nBytes);
  float* B_host=(float*)malloc(nBytes);
  initialData(A_host,nxy);

  //cudaMalloc
  float *A_dev=NULL;
  float *B_dev=NULL;
  CHECK(cudaMalloc((void**)&A_dev,nBytes));
  CHECK(cudaMalloc((void**)&B_dev,nBytes));

  CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(B_dev,0,nBytes));



  // cpu compute
  double iStart=cpuSecond();
  transformMatrix2D_CPU(A_host,B_host,nx,ny);
  double iElaps=cpuSecond()-iStart;
  printf("CPU Execution Time elapsed %f sec\n",iElaps);

  // 2d block and 2d grid
  dim3 block(dimx,dimy);
  dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1);
  dim3 block_1(dimx,dimy);
  dim3 grid_1((nx-1)/(block_1.x*4)+1,(ny-1)/block_1.y+1);
  iStart=cpuSecond();
  switch(transform_kernel)
  {
  case 0:
    copyRow<<<grid,block>>>(A_dev,B_dev,nx,ny);
    break;
  case 1:
    copyCol<<<grid,block>>>(A_dev,B_dev,nx,ny);
    break;
  case 2:
    transformNaiveRow<<<grid,block>>>(A_dev,B_dev,nx,ny);
    break;
  case 3:
        transformNaiveCol<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
  case 4:
        transformNaiveColUnroll<<<grid_1,block_1>>>(A_dev,B_dev,nx,ny);
        break;
  case 5:

        transformNaiveColUnroll<<<grid_1,block_1>>>(A_dev,B_dev,nx,ny);
        break;
  case 6:
        transformNaiveRowDiagonal<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
  case 7:
        transformNaiveColDiagonal<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
  default:
    break;
  }
  CHECK(cudaDeviceSynchronize());
  iElaps=cpuSecond()-iStart;
  printf(" Time elapsed %f sec\n",iElaps);
  CHECK(cudaMemcpy(B_host,B_dev,nBytes,cudaMemcpyDeviceToHost));
  checkResult(B_host,B_host,nxy);

  cudaFree(A_dev);
  cudaFree(B_dev);
  free(A_host);
  free(B_host);
  cudaDeviceReset();
  return 0;
}
