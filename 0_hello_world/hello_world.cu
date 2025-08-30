//hello_world.cu
#include <cstdio>
#include <stdio.h>
//global告诉编译器这个是可以在设备上执行的核函数
__global__ void hello_world(void)
{
  printf("gpu: hello_world\n");
}
int main()
{
  printf("cpu: hello_world\n");
  hello_world<<<1,10>>>();//<<<>>>是对设备进行配置的参数
  cudaDeviceReset();//必须执行，包含隐式同步，让主机线程等待gpu执行完毕再继续
  return 0;
}
