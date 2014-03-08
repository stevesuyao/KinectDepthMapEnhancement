/////////////////////////////////////////////////////////////////////////////////
// contents :NormalMap generation from SmoothingAreaMap, IntegralImage, vertexMap
// create 	:2013/03/17
// modefied :  
// writer   :Takuya Ikeda
// other	:
/////////////////////////////////////////////////////////////////////////////////

//#include "../header/OpenCV.h"
#include "NormalMapGenerator.h"

NormalMapGenerator::NormalMapGenerator(int w, int h): iig(w, h), samg(w, h){
	this->width = w;
	this->height = h;
	normal_estimation_method_ = BILATERAL; // default setting

	initMemory();
}

NormalMapGenerator::~NormalMapGenerator(){
	cudaFree(verticeMap);
	cudaFree(normalMap);
	cudaFree(segmentNormalMap);
}


//void NormalMapGenerator::setCameraParams(unsigned long long F, float p_size){
//	dc.setCameraParameters(F, p_size);
//}

//void NormalMapGenerator::setInput(float3* in){
//	//compute Vetexmap
//	//dc.projectiveToReal(in, verticeMap);
//}

void NormalMapGenerator::setNormalEstimationMethods(int method){
	normal_estimation_method_ = method;
}


void NormalMapGenerator::initMemory(){

	cudaMalloc((void**) &verticeMap, width * height * sizeof(float3));
	cudaMalloc((void**) &normalMap, width * height * sizeof(float3));
	cudaMalloc((void**) &segmentNormalMap, width * height * sizeof(float3));
	
}

float3* NormalMapGenerator::getNormalMap(void){
	return normalMap;	
}

