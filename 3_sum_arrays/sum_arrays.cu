#include "../include/freshman.h"
#include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>

void sumArrays(float *a, float *b, float *res, const int size) {
  for (int i = 0; i < size; i += 4) {
    res[i] = a[i] + b[i];
    res[i + 1] = a[i + 1] + b[i + 1];
    res[i + 2] = a[i + 2] + b[i + 2];
    res[i + 3] = a[i + 3] + b[i + 3];
  }
}
/*
 *
 * __global__设备端执行，可以从主机调用也可以从计算能力3以上的设备调用
 * __device__设备端执行设备端调用
 * __host__主机端执行主机端调用，可以不写
 * 可以同时定义为host和device,这种函数可以同时被主机端和设备端的代码调用
 * Kernel核函数编写有以下限制
 *   1.只能访问设备内存
 *   2.必须有void返回类型
 *   3.不支持可变数量的参数
 *   4.不支持静态变量
 *   5.显示异步行为
 */
// 核函数返回值必须是void
__global__ void sumArraysGPU(float *a, float *b, float *res) {
  // int i=threadIdx.x;
  //(0-15) * 1024 + (0-1023)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  res[i] = a[i] + b[i];
}
int main(int argc, char **argv) {
  int dev = 0;
  cudaSetDevice(dev);

  int nElem = 1 << 14; // 16384
  printf("Vector size:%d\n", nElem);
  int nByte = sizeof(float) * nElem;
  float *a_h = (float *)malloc(nByte);
  float *b_h = (float *)malloc(nByte);
  float *res_h = (float *)malloc(nByte);
  float *res_from_gpu_h = (float *)malloc(nByte);
  memset(res_h, 0, nByte);
  memset(res_from_gpu_h, 0, nByte);

  float *a_d, *b_d, *res_d;
  CHECK(cudaMalloc((float **)&a_d, nByte));
  CHECK(cudaMalloc((float **)&b_d, nByte));
  CHECK(cudaMalloc((float **)&res_d, nByte));

  initialData(a_h, nElem);
  initialData(b_h, nElem);

  CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

  dim3 block(1024);
  dim3 grid(nElem / block.x); // 16
  // 启动核函数 <<<>>>是对设备代码执行的线程结构的配置
  sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
  printf("Execution configuration<<<%d,%d>>>\n", grid.x, block.x);

  // cudaMemcpy从设备复制数据回到主机，主机端必须等待设备端计算完成，也就是隐式的同步
  CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
  sumArrays(a_h, b_h, res_h, nElem);

  checkResult(res_h, res_from_gpu_h, nElem);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(a_h);
  free(b_h);
  free(res_h);
  free(res_from_gpu_h);

  return 0;
}
