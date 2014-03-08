#include "ArrayBuffer.h"




/*
Initialize the 2D buffer in device
*/
__global__ void initDeviceMemoryElementsKernel(
ArrayBuffer::weighted_d* devPtr, 
int width, 
int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//ArrayBuffer::weighted_d* ref = &((ArrayBuffer::weighted_d*)((char*)devPtr + y * pitch))[x];

	ArrayBuffer::weighted_d* ref = &devPtr[x + y * width];

	ref->d = 0.0;
	ref->w = 0.0;
}




void ArrayBuffer::initDeviceMemoryElements(){
	
	initDeviceMemoryElementsKernel<<<gridSize, blockSize>>>(devPtr, width, height);
}