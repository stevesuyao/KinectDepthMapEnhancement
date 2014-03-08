//////////////////////////////////////////////////////
// contents :generate SmoothingAreaMap from depth map
// create 	:2013/03/17
// modefied :
// writer   :Takuya Ikeda 
// other	:
//////////////////////////////////////////////////////

#include "SmoothingAreaMapGenerator.h"
//#include "../header/OpenCV.h"

SmoothingAreaMapGenerator::SmoothingAreaMapGenerator(int w, int h){
	width = w, height = h;

	max_depth_change_factor_ = 0.03f;//ÉÅÅ[ÉgÉã
	normal_smoothing_size_ = 20.0f;//pixel
	//max_depth_change_factor_ = 300.0f;
	//normal_smoothing_size_ = 10000.0f;

	//normal_smoothing_size_ = 20.0f;
	initMemory();
}

SmoothingAreaMapGenerator::~SmoothingAreaMapGenerator(){

}

void SmoothingAreaMapGenerator::initMemory(void){
	cudaMalloc((void**) &verticeMap , sizeof(float3)*width*height);

	cudaMalloc((void**) &depthChangeIndicationMap , sizeof(int)*width*height);
	cudaMalloc((void**) &distanceTransformMap , sizeof(float)*width*height);
	cudaMalloc((void**) &depthDependentSmoothingAreaMap , sizeof(float)*width*height);
	cudaMalloc((void**) &finalSmoothingMap , sizeof(float)*width*height);
	return;
}
void SmoothingAreaMapGenerator::setVerticeMap(float3* in){
	cudaMemcpy(verticeMap, in, sizeof(float3)*width*height, cudaMemcpyDeviceToDevice);
	return;
}

void SmoothingAreaMapGenerator::generateFinalSmoothingAreaMap(){
	computeDCIMap();
	computeDTMap();
	computeDDSAIMap();
	computeFSIMap();
	return;
}

float* SmoothingAreaMapGenerator::getFinalSmoothingMap(void){
	return finalSmoothingMap;
}