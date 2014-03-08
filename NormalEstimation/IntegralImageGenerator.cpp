//////////////////////////////////////////////////
// contents :generate integral image from depthmap
// create 	:2013/03/17
// modefied :
// writer   :Takuya Ikeda 
// other	:
//////////////////////////////////////////////////


#include "IntegralImagegenerator.h"

IntegralImagegenerator::IntegralImagegenerator(int w, int h){
	this->width = w;
	this->height = h;
	initMemory();
}

IntegralImagegenerator::~IntegralImagegenerator(){
	cudaFree(vertexMap);
	cudaFree(IntegralZ);
	cudaFree(IntegralXYZ);
	cudaFree(IntegralXXXYXZ);
	cudaFree(IntegralYYYZZZ);
	cudaFree(dinMd);
	cudaFree(doutMd);
	cudaFree(dinMui);
	cudaFree(doutMui);
}

void IntegralImagegenerator::initMemory(void){
	cudaMalloc((void**) &vertexMap , sizeof(float3)*width*height);

	cudaMalloc((void**) &IntegralZ , sizeof(double)*width*height);

	cudaMalloc((void**) &IntegralXYZ , sizeof(double3)*width*height);
	cudaMalloc((void**) &IntegralXXXYXZ , sizeof(double3)*width*height);
	cudaMalloc((void**) &IntegralYYYZZZ , sizeof(double3)*width*height);

	cudaMalloc((void**) &IntegralCount , sizeof(unsigned)*width*height);

	cudaMalloc((void**) &dinMd , sizeof(double)*M_HEIGHT*M_WIDTH);
	cudaMalloc((void**) &doutMd, sizeof(double)*M_HEIGHT*M_WIDTH);
	cudaMalloc((void**) &dinMui , sizeof(unsigned)*M_HEIGHT*M_WIDTH);
	cudaMalloc((void**) &doutMui, sizeof(unsigned)*M_HEIGHT*M_WIDTH);
	return;
}

void IntegralImagegenerator::setInput(float3* in){
	cudaMemcpy(vertexMap, in, sizeof(float3)*width*height, cudaMemcpyDeviceToDevice);
	return;
}

double* IntegralImagegenerator::getIntegralZ(void){
	return IntegralZ;
}

double3* IntegralImagegenerator::getIntegralXYZ(void){
	return IntegralXYZ;
}

double3* IntegralImagegenerator::getIntegralXXXYXZ(void){
	return IntegralXXXYXZ;
}

double3* IntegralImagegenerator::getIntegralYYYZZZ(void){
	return IntegralYYYZZZ;
}

unsigned* IntegralImagegenerator::getIntegralCount(void){
	return IntegralCount;
}
