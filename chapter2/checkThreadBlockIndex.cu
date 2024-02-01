#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>


#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess){\
        fprintf("Error: %s:%d", __FILE__, __LINE__);\
        fprintf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(-10*error);\
    }\
}




__global__ void showIndice(float *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = iy * nx + ix;    //放弃记住吧，不详细分析根本 get不到

    printf("The ix in computing matrix is %d\n", ix);
    printf("The iy in computing matrix is %d\n", iy);
    printf("The index in global memory is %d\n", index);
}


void initializeMatrix(float *A, const int s){
    for (int i=0; i<s; i++){
        A[i] = (float)rand();
    }
}


int main(){
    int nx = 10;
    int ny = 6;
    int s = nx * ny;
    int nBytes = s * sizeof(float);
    float* A;
    A = (float *)malloc(nBytes);
    initializeMatrix(A, s);
    float *device_A;
    cudaMalloc((void **)device_A, nBytes);
    cudaMemcpy(device_A, A, nBytes, cudaMemcpyHostToDevice);
    dim3 block(5, 2);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);
    showIndice <<< grid, block >>>(device_A, nx, ny);
    cudaDeviceSynchronize();
    printf("It finished!");
    cudaFree(device_A);
    free(A);
    cudaDeviceReset();  //suggested to be used only when to check memory leak

    return 0;

}