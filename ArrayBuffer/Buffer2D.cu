#include "Buffer2D.h"


//Locate the referencing element from pitched 2D device array
__device__ ArrayBuffer::weighted_d* getReference(
	int x, 
	int y, 
	ArrayBuffer::weighted_d* devPtr, 
	size_t pitch){
		return &((ArrayBuffer::weighted_d*)((char*)devPtr + y * pitch))[x];
}

__device__ void updateWaitedDepth(ArrayBuffer::weighted_d* ref, float d, const float THRESH){
	if(d > 50.0){
		//if(fabs(d - ref->d) < THRESH){
		//if(true){
		//mimic that of TSDF in KinectFusion
		if(ref->d != 0.0f){
			if((float)abs((int)ref->d-(int)d) < d*0.01f){
				ref->d = 
					((ref->d * (ref->w + 1)) + (d * ref->w)) / (ref->w * 2 + 1);
				ref->w++;
			}
		}
		else {
			ref->d = d;
			ref->w = 1.0;
		}
	}
}


__global__ void insertDataKernel(
	ArrayBuffer::weighted_d* devPtr, 
	float* data, 
	int width, 
	int height, 
	size_t pitch,
	const float THRESHOLD){	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		ArrayBuffer::weighted_d* ref = &devPtr[x + y * width];
		float d = data[x + y * width];
		//if(d > 50.0){
		ref->d = d;
		ref->w = 1.0;
		//}
		//updateWaitedDepth(ref, d, THRESHOLD);
}


void Buffer2D::insertData(float* data){
	insertDataKernel<<<gridSize, blockSize>>>
		(devPtr, data, width, height, pitch, DEPTH_THRESHOLD);
}


__global__ void getDepthMapKernel(
	ArrayBuffer::weighted_d* devPtr, 
	float* data, 
	int width, 
	int height, 
	size_t pitch){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		ArrayBuffer::weighted_d* ref = &devPtr[x + y * width];
		data[x + y * width] = ref->d;
}


void Buffer2D::getDepthMap(float* out){
	getDepthMapKernel<<<gridSize, blockSize>>>
		(devPtr, out, width, height, pitch);

}

__global__ void getWeightMapKernel(
	ArrayBuffer::weighted_d* devPtr, 
	float* data, 
	int width, 
	int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		ArrayBuffer::weighted_d* ref = &devPtr[x + y * width];
		data[x + y * width] = ref->w;
}

void Buffer2D::getWeightMap(float* out){
	getWeightMapKernel<<<gridSize, blockSize>>>
		(devPtr, out, width, height);
}


__global__ void updateDataKernel(
	ArrayBuffer::weighted_d* devPtr, 
	float* data, 
	int width, 
	int height, 
	size_t pitch, 
	const float THRESHOLD){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		ArrayBuffer::weighted_d* ref = &devPtr[x + y * width];
		float d = data[x + y * width];


		updateWaitedDepth(ref, d, THRESHOLD);

}


void Buffer2D::updateData(float* data){
	updateDataKernel<<<gridSize, blockSize>>>
		(devPtr, data, width, height, pitch, DEPTH_THRESHOLD);

}


__global__ void insertDataKernel(
	ArrayBuffer::weighted_d* devPtr, 
	float2* data, 
	int width, 
	int height, 
	size_t pitch){	
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		ArrayBuffer::weighted_d* ref = &devPtr[x + y * width];
		float d = data[x + y * width].x;
		float w = data[x + y * width].y;
		//if(d > 50.0){
		ref->d = d;
		ref->w = y;

		//}
}



void Buffer2D::insertData(float2* data){
	insertDataKernel<<<gridSize, blockSize>>>
		(devPtr, data, width, height, pitch);
}