

#include "utils.hpp"




#define CHECK(call) 
{ 
 const cudaError_t error = call; 
 if (error != cudaSuccess) 
 { 
 printf("Error: %s:%d, ", __FILE__, __LINE__); 
 printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); 
 exit(-10*error);
 }
}


void initialInt(int *ip, int size) {
 for (int i=0; i<size; i++) {
 ip[i] = i;
 }
}

void initialData(float *ip,int size) {
 // generate different seed for random number
 time_t t;
 srand((unsigned) time(&t));
 for (int i=0; i<size; i++) {
 ip[i] = (float)( rand() & 0xFF )/10.0f;
 }
}


void sumArraysOnHost(float *A, float *B, float *C, const int N) {
 for (int idx=0; idx<N; idx++)
 C[idx] = A[idx] + B[idx];
}
__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
 int i = threadIdx.x;
 C[i] = A[i] + B[i];
}

void printMatrix(int *C, const int nx, const int ny) {
 int *ic = C;
 printf("\nMatrix: (%d.%d)\n",nx,ny);
 for (int iy=0; iy<ny; iy++) {
 for (int ix=0; ix<nx; ix++) {
 printf("%3d",ic[ix]);
 }
 ic += nx;
 printf("\n");
 }
 printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
 int ix = threadIdx.x + blockIdx.x * blockDim.x;
 int iy = threadIdx.y + blockIdx.y * blockDim.y;
 unsigned int idx = iy*nx + ix;
 printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) "
 "global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x,
 blockIdx.y, ix, iy, idx, A[idx]);
}


double cpuSecond() {
 struct timeval tp;
 gettimeofday(&tp,NULL);
 return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
 double epsilon = 1.0E-8;
 bool match = 1;
 for (int i=0; i<N; i++) {
 if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
 match = 0;
 printf("Arrays do not match!\n");
 printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
 break;
 }
 }
 if (match) printf("Arrays match.\n\n");
}

