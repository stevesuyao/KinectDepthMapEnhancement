#include "ArrayBuffer.h"

ArrayBuffer::ArrayBuffer(int width, int height)
{
	this->width = width;
	this->height = height;
	blockSize = dim3(32, 24);
	gridSize = dim3(width / 32, height / 24);
}



void ArrayBuffer::initDeviceMemory(){
	cudaMalloc(&devPtr, width * height * sizeof(weighted_d));
	//cudaMallocPitch(&devPtr, &pitch, width * sizeof(weighted_d), height);
	initDeviceMemoryElements();
}

ArrayBuffer::weighted_d* ArrayBuffer::getRawPointer(){
	return devPtr;
}