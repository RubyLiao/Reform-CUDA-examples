
#include "matrix.hpp"
#include "utils.hpp"


#include <iostream>

using namespace std;



void multiplyMatrices(float* A, float* B, float* C, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            *(C + i * width + j) = 0;
            for (int k = 0; k < width; ++k) {
                *(C + i * width + j) += *(A + i * width + k) * *(B + k * width + j);
            }
        }
    }
}


__global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {
// Calculate the row index of the d_Pelement and d_M
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    // Calculate the column index of d_P and d_N
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((Row < Width) && (Col < Width)) {
        float Pvalue = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k) {
            Pvalue += d_M[Row*Width+k]*d_N[k*Width+Col];
        }
        d_P[Row*Width+Col] = Pvalue;
    } 
} 


#define BLOCK_WIDTH 16
// Setup the execution configuration


int main()
{

    int nx = 1024; //1<<14;
    int ny = 1024; //1<<14;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n",nx, ny);
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    // initialize data at host side
    double iStart = cpuSecond();
    initialData (h_A, nxy);
    initialData (h_B, nxy);
    double iElaps = cpuSecond() - iStart;
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    // add matrix at host side for result checks
    iStart = cpuSecond();
    multiplyMatrices(h_A, h_B, hostRef, nx);
    iElaps = cpuSecond() - iStart;
    printf("Matrix Multiplication on CPU takes %d", iElaps);
    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    int Width = nx;
    int NumBlocks = Width/BLOCK_WIDTH;
    if (Width % BLOCK_WIDTH) NumBlocks++;
    dim3 dimGrid(NumBlocks, NumBlocks);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    // Launch the device computation threads!
    
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_MatA, d_MatB, d_MatC, Width);

    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    checkResult(gpuRef, hostRef, nxy);

    return 0;
    
}