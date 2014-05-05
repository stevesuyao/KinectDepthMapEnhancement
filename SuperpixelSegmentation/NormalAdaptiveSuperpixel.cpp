#include "NormalAdaptiveSuperpixel.h"
#include <ctime>

NormalAdaptiveSuperpixel::NormalAdaptiveSuperpixel(int width, int height):
	DepthAdaptiveSuperpixel(width, height),
	normalImage(height, width){
		cudaMalloc(&reliabilityMap, sizeof(float)*width*height);
	}
NormalAdaptiveSuperpixel::~NormalAdaptiveSuperpixel(){
	cudaFree(superpixelCenters_Host);
	cudaFree(superpixelCenters_Device);
	cudaFree(superpixelNormals_Host);
	cudaFree(superpixelNormals_Device);
	cudaFree(NormalsVariance_Host);
	cudaFree(NormalsVariance_Device);
	cudaFree(reliabilityMap);
}

void NormalAdaptiveSuperpixel::initMemory(){
	//superpixel data
	cudaMallocHost(&meanData_Host, sizeof(superpixel) * ClusterNum.x*ClusterNum.y);	
	cudaMalloc(&meanData_Device, sizeof(superpixel) * ClusterNum.x*ClusterNum.y);
	//Random color
	RandomColors = new int3[ClusterNum.x*ClusterNum.y];
	/////////////////////////////////Virtual/////////////////////////////////////////
	//superpixel centers
	cudaMallocHost(&superpixelCenters_Host, sizeof(float3) * ClusterNum.x*ClusterNum.y);
	cudaMalloc(&superpixelCenters_Device, sizeof(float3) * ClusterNum.x*ClusterNum.y);
	/////////////////////////////////Virtual/////////////////////////////////////////
	//superpixel normals
	cudaMallocHost(&superpixelNormals_Host, sizeof(float3) * ClusterNum.x*ClusterNum.y);
	cudaMalloc(&superpixelNormals_Device, sizeof(float3) * ClusterNum.x*ClusterNum.y);
	////////////////////////////////Virtural/////////////////////////////////////////
	//normals variance
	cudaMallocHost(&NormalsVariance_Host, sizeof(float) * ClusterNum.x*ClusterNum.y);
	cudaMalloc(&NormalsVariance_Device, sizeof(float) * ClusterNum.x*ClusterNum.y);
}
cv::Mat_<cv::Vec3b>	NormalAdaptiveSuperpixel::getNormalImg(){
	cudaMemcpy(superpixelNormals_Host, superpixelNormals_Device, sizeof(float3)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
	//cudaMemcpy(Labels_Host, Labels_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			int id = Labels_Host[y*width+x];
			normalImage.at<cv::Vec3b>(y,x).val[0] = (unsigned char)(255.0f*(superpixelNormals_Host[id].x+1.0f)/2.0f);
			normalImage.at<cv::Vec3b>(y,x).val[1] = (unsigned char)(255.0f*(superpixelNormals_Host[id].y+1.0f)/2.0f);
			normalImage.at<cv::Vec3b>(y,x).val[2] = (unsigned char)(255.0f*(superpixelNormals_Host[id].z+1.0f)/2.0f);
		}
	}
	//for(int i=0; i<ClusterNum.x*ClusterNum.y; i++){
	//	normalImage.at<cv::Vec3b>(meanData_Host[i].y,meanData_Host[i].x) = cv::Vec3b(0, 0, 0);
	//}
	//cv::imshow("NASP_normals", normalImage);
	return normalImage;
}
float3*	NormalAdaptiveSuperpixel::getCentersHost(){
	cudaMemcpy(superpixelCenters_Host, superpixelCenters_Device, sizeof(float3)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
	return superpixelCenters_Host;
}
float3*	NormalAdaptiveSuperpixel::getCentersDevice(){
	return superpixelCenters_Device;
}
float3*	NormalAdaptiveSuperpixel::getNormalsHost(){
	//cudaMemcpy(superpixelNormals_Host, superpixelNormals_Device, sizeof(float3)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
	return superpixelNormals_Host;
}
float3*	NormalAdaptiveSuperpixel::getNormalsDevice(){
	return superpixelNormals_Device;
}
float* NormalAdaptiveSuperpixel::getNormalsVarianceHost(){
	cudaMemcpy(NormalsVariance_Host, NormalsVariance_Device, sizeof(float)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
	return NormalsVariance_Host;
}
float* NormalAdaptiveSuperpixel::getNormalsVarianceDevice(){
	return NormalsVariance_Device;
}