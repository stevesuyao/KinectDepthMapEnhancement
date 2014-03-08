#include "LabelEquivalenceSeg.h"
#include <stdlib.h>
#include <iostream>

#include <cuda_runtime.h>

LabelEquivalenceSeg::LabelEquivalenceSeg(int width, int height):
	show(height, width),
	normalImage(height, width){
		this->width = width;
		this->height = height;

		cudaMalloc(&InputND_Device, sizeof(float4) * width * height);
		cudaMalloc(&MergedClusterND_Device, sizeof(float4) * width * height);
		cudaMallocHost(&MergedClusterND_Host, sizeof(float4) * width * height);
		cudaMalloc(&MergedClusterLabel_Device, sizeof(int) * width * height);
		cudaMallocHost(&MergedClusterLabel_Host, sizeof(int) * width * height);
		cudaMalloc(&MergedClusterCenters_Device, sizeof(float3) * width * height);
		cudaMallocHost(&MergedClusterCenters_Host, sizeof(float3) * width * height);
		cudaMalloc(&MergedClusterVariance_Device, sizeof(float) * width * height);
		cudaMallocHost(&MergedClusterVariance_Host, sizeof(float) * width * height);
		cudaMalloc(&ref, sizeof(int) * width * height);
		cudaMallocHost(&ref_host, sizeof(int) * width * height);
		cudaMalloc(&merged_cluster_size, sizeof(int) * width * height);
		cudaMalloc(&sum_of_merged_cluster_normals, sizeof(float3) * width * height);
		cudaMalloc(&sum_of_merged_cluster_centers, sizeof(float3) * width * height);
		cudaMalloc(&spColor_Device, sizeof(rgb) * width * height);

		color_pool = new unsigned char*[width * height];
		//width*height‚Ìcolor 
		for(int i = 0; i < width * height; i++){
			color_pool[i] = new unsigned char[3];
			color_pool[i][0] = (unsigned char)rand();
			color_pool[i][1] = (unsigned char)rand();
			color_pool[i][2] = (unsigned char)rand();
		}
		Writer = cv::VideoWriter::VideoWriter("merged_superpixel.avi", CV_FOURCC('X','V','I','D'), 3.0, cv::Size(width, height));
}


LabelEquivalenceSeg::~LabelEquivalenceSeg(){
	cudaFree(InputND_Device);
	cudaFree(spColor_Device);
	cudaFree(MergedClusterND_Device);
	cudaFree(MergedClusterND_Host);
	cudaFree(MergedClusterLabel_Device);
	cudaFree(MergedClusterLabel_Host);
	cudaFree(MergedClusterCenters_Device);
	cudaFree(MergedClusterCenters_Host);
	cudaFree(MergedClusterVariance_Device);
	cudaFree(MergedClusterVariance_Host);
	cudaFree(ref);
	cudaFree(ref_host);
	cudaFree(merged_cluster_size);
	cudaFree(sum_of_merged_cluster_normals);
	cudaFree(sum_of_merged_cluster_centers);
	show.~Mat_<cv::Vec3b>();
	delete [] color_pool;
}
void LabelEquivalenceSeg::releaseVideo(){
	Writer.release();
}
void LabelEquivalenceSeg::viewSegmentResult(){

	cv::Vec3b* pixel;
	unsigned char* color;
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			pixel = &show.at<cv::Vec3b>(cv::Point2d(j, i));
			if(MergedClusterLabel_Host[j + i*width] > -1){
				//std::cout << "nd: "<<MergedClusterND_Host[i*width+j].x<< ", "<<MergedClusterND_Host[i*width+j].y<<", "<<MergedClusterND_Host[i*width+j].z<<std::endl;
				color = color_pool[MergedClusterLabel_Host[j + i * width]];
				pixel->val[0] = color[0];
				pixel->val[1] = color[1];
				pixel->val[2] = color[2];
			} else {
				pixel->val[0] = 0;
				pixel->val[1] = 0;
				pixel->val[2] = 0;
			}
		}
	}
	Writer << show;
	cv::imshow("Labeled surfaces", show);
	// cv::waitKey(1);
}
cv::Mat_<cv::Vec3b> LabelEquivalenceSeg::getSegmentResult(){

	cv::Vec3b* pixel;
	unsigned char* color;
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			pixel = &show.at<cv::Vec3b>(cv::Point2d(j, i));
			if(MergedClusterLabel_Host[j + i*width] > -1){
				//std::cout << "nd: "<<MergedClusterND_Host[i*width+j].x<< ", "<<MergedClusterND_Host[i*width+j].y<<", "<<MergedClusterND_Host[i*width+j].z<<std::endl;
				color = color_pool[MergedClusterLabel_Host[j + i * width]];
				pixel->val[0] = color[0];
				pixel->val[1] = color[1];
				pixel->val[2] = color[2];
			} else {
				pixel->val[0] = 0;
				pixel->val[1] = 0;
				pixel->val[2] = 0;
			}
		}
	}
	return show;
}
cv::Mat_<cv::Vec3b>	LabelEquivalenceSeg::getNormalImg(){
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			int id = MergedClusterLabel_Host[y*width+x];
			if(id==-1)
				normalImage.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 0, 0);
			else{
				normalImage.at<cv::Vec3b>(y,x).val[0] = (unsigned char)(255.0f*(MergedClusterND_Host[y*width+x].x+1.0f)/2.0f);
				normalImage.at<cv::Vec3b>(y,x).val[1] = (unsigned char)(255.0f*(MergedClusterND_Host[y*width+x].y+1.0f)/2.0f);
				normalImage.at<cv::Vec3b>(y,x).val[2] = (unsigned char)(255.0f*(MergedClusterND_Host[y*width+x].z+1.0f)/2.0f);
			}
		}
	}
	//cv::imshow("NASP_normals", normalImage);
	return normalImage;
}
float4* LabelEquivalenceSeg::getMergedClusterND_Device()const{
	return MergedClusterND_Device;
}
int* LabelEquivalenceSeg::getMergedClusterLabel_Device()const{
	return MergedClusterLabel_Device;
}
float4* LabelEquivalenceSeg::getMergedClusterND_Host()const{
	return MergedClusterND_Host;
}
int* LabelEquivalenceSeg::getMergedClusterLabel_Host()const{
	return MergedClusterLabel_Host;
}
float* LabelEquivalenceSeg::getMergedClusterVariance_Device()const{
	return MergedClusterVariance_Device;
}
float* LabelEquivalenceSeg::getMergedClusterVariance_Host()const{
	return MergedClusterVariance_Host;
}