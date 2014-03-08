#ifndef ARRAYBUFFER_H
#define ARRAYBUFFER_H
#include <cuda.h>
#include <cuda_runtime.h>

/*
Abstract base class for all buffer classes
*/
class ArrayBuffer{

public:
	typedef struct __align__(8){
		float d;
		float w;
	}weighted_d;

	ArrayBuffer(int width, int height);

	virtual void insertData(float* data)		= 0;
	virtual void insertData(weighted_d* data)	= 0;
	virtual void getDepthMap(float* out)		= 0;
	virtual void getWeightMap(float* out)		= 0;
	virtual void updateData(float* data)		= 0;

	weighted_d* getRawPointer();

protected:


	dim3 blockSize;
	dim3 gridSize;

	weighted_d* devPtr;
	size_t pitch;

	int width;
	int height;



	void initDeviceMemory();
	void initDeviceMemoryElements();


};
#endif
