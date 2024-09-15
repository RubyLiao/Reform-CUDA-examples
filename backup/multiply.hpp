#ifndef MULTIPLY_HPP
#define MULTIPLY_HPP

#include <cuda_runtime.h>
#include <stdio.h>
#include <ctime> 
#include <cstdlib>  // Required for srand() and rand()
#include <sys/time.h>




inline void multiplyMatrices(float* A, float* B, float* C, int width);
inline __global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int Width);


inline void initialInt(int *ip, int size);
inline void initialData(float *ip, int size);
inline void printMatrix(int *C, const int nx, const int ny);
inline __global__ void printThreadIndex(int *A, const int nx, const int ny);
inline void sumArraysOnHost(float *A, float *B, float *C, const int N);
inline __global__ void sumArraysOnGPU(float *A, float *B, float *C);
inline double cpuSecond();
inline void checkResult(float *hostRef, float *gpuRef, const int N);


#endif