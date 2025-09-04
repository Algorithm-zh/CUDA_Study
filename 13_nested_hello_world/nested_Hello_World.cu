#include <cuda_runtime.h>
#include <stdio.h>
__global__ void nesthelloworld(int iSize,int iDepth)
{
    unsigned int tid=threadIdx.x;
    printf("depth : %d blockIdx: %d,threadIdx: %d\n",iDepth,blockIdx.x,threadIdx.x);
    if (iSize==1)
        return;
    int nthread=(iSize>>1);
    if (tid==0 && nthread>0)
    {
        //每个block里的第一个线程启动一个子网格,子网格都是一个block, 每个block有nthread个线程,
        nesthelloworld<<<1,nthread>>>(nthread,++iDepth);//动态并行，也就是递归，需要用到一个库，所以这里编辑器会报错
        //同一祖宗的网格都是隐式同步的                                                        
        /*
         * 父线程块启动子网格需要显示的同步，也就是说不通的线程束需要都执行到子网格调用那一句，
         * 这个线程块内的所有子网格才能依据所在线程束的执行，一次执行
         */
        printf("-----------> nested execution depth: %d\n",iDepth);
    }

}

int main(int argc,char* argv[])
{
    int size=64;
    int block_x=2;
    dim3 block(block_x,1);
    dim3 grid((size-1)/block.x+1,1);
    //32个block,每个里面2个thread
    nesthelloworld<<<grid,block>>>(size,0);
    cudaGetLastError();
    cudaDeviceReset();
    return 0;
}
