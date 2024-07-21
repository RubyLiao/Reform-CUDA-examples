#include <cuda_runtime.h>
#include <stdio.h>




void initialInt(int *ip, int size);
void initialData(float *ip,int size);
void printMatrix(int *C, const int nx, const int ny);
__global__ void printThreadIndex(int *A, const int nx, const int ny);
void sumArraysOnHost(float *A, float *B, float *C, const int N);
__global__ void sumArraysOnGPU(float *A, float *B, float *C);
double cpuSecond();
void checkResult(float *hostRef, float *gpuRef, const int N);