#include "DepthAdaptiveSuperpixel.h"
#include <ctime>

DepthAdaptiveSuperpixel::DepthAdaptiveSuperpixel(int width, int height):
	SuperpixelSegmentation(width, height){
	cudaMalloc(&intrinsicDevice, sizeof(float)*3*3);
	cudaMallocHost(&intrinsicHost, sizeof(float)*3*3);
	}
DepthAdaptiveSuperpixel::~DepthAdaptiveSuperpixel(){
	cudaFree(superpixelCenters_Host);
	cudaFree(superpixelCenters_Device);
	cudaFree(intrinsicDevice);
	cudaFree(intrinsicHost);
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
	for(int y=0; y<3; y++){
		for(int x=0; x<3; x++){
			intrinsicHost[y*3+x] = (float)(intrinsic.at<double>(y,x));
		}
	}
	cudaMemcpy(intrinsicDevice, intrinsicHost, sizeof(float)*3*3, cudaMemcpyHostToDevice);
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
