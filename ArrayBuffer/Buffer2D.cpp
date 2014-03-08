#include "Buffer2D.h"


Buffer2D::Buffer2D(int width, int height):
ArrayBuffer(width, height),
DEPTH_THRESHOLD(100.0){

	initDeviceMemory();
	initDeviceMemoryElements();
}


void Buffer2D::insertData(weighted_d* data){
	cudaMemcpy(devPtr, data, width * height * sizeof(weighted_d), cudaMemcpyDeviceToDevice);
}


void Buffer2D::insertData(xn::DepthMetaData* data){
	float* temp; cudaMalloc(&temp, width * height * sizeof(float));
	float* host_temp = new float[width * height];

	for(int x = 0; x < width; x++)
		for(int y = 0; y < height; y++)
			host_temp[x + y * width] = (*data)(x, y);

	cudaMemcpy(temp, host_temp, width * height * sizeof(float), cudaMemcpyHostToDevice);

	updateData(temp);
	
	delete[] host_temp;
	cudaFree(temp);
}