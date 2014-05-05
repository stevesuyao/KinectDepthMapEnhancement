#include "DepthAdaptiveSuperpixel.h"
#include <ctime>

DepthAdaptiveSuperpixel::DepthAdaptiveSuperpixel(int width, int height):
	SuperpixelSegmentation(width, height),
	Intrinsic_Device(cv::gpu::createContinuous(3, 3, CV_32F)){}
DepthAdaptiveSuperpixel::~DepthAdaptiveSuperpixel(){
	cudaFree(superpixelCenters_Host);
	cudaFree(superpixelCenters_Device);
}
void DepthAdaptiveSuperpixel::SetParametor(int rows, int cols, cv::Mat_<double> intrinsic){
	//number of clusters
	ClusterNum.x = cols;
	ClusterNum.y = rows;
	//grid(window) size
	Window_Size.x = width/cols;
	Window_Size.y = height/rows;
	//Init GPU memory
	initMemory();						
	//Random colors
	for(int i=0; i<ClusterNum.x*ClusterNum.y; i++){
		int3 tmp;
		tmp.x = rand()%255;
		tmp.y = rand()%255;
		tmp.z = rand()%255;
		RandomColors[i] = tmp;
	}
	////////////////////////////////Virtual//////////////////////////////////////////
	//set intrinsic mat
	cv::Mat_<float> intr;
	intrinsic.convertTo(intr, CV_32F);
	Intrinsic_Device.upload(intr);
}
void DepthAdaptiveSuperpixel::initMemory(){
	//superpixel data
	cudaMallocHost(&meanData_Host, sizeof(superpixel) * ClusterNum.x*ClusterNum.y);	
	cudaMalloc(&meanData_Device, sizeof(superpixel) * ClusterNum.x*ClusterNum.y);
	//Random color
	RandomColors = new int3[ClusterNum.x*ClusterNum.y];
	/////////////////////////////////Virtual/////////////////////////////////////////
	//superpixel centers
	cudaMallocHost(&superpixelCenters_Host, sizeof(float3) * ClusterNum.x*ClusterNum.y);
	cudaMalloc(&superpixelCenters_Device, sizeof(float3) * ClusterNum.x*ClusterNum.y);
}
