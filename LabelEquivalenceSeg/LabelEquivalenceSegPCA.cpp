#include "LabelEquivalenceSegPCA.h"
#include <stdlib.h>
#include <iostream>

#include <cuda_runtime.h>

LabelEquivalenceSegPCA::LabelEquivalenceSegPCA(int width, int height):
		show(height, width){
        this->width = width;
        this->height = height;

		cudaMalloc(&InputND_Device, sizeof(float4) * width * height);
        cudaMalloc(&MergedClusterND_Device, sizeof(float4) * width * height);
		cudaMallocHost(&MergedClusterND_Host, sizeof(float4) * width * height);
		cudaMalloc(&MergedClusterLabel_Device, sizeof(int) * width * height);
		cudaMallocHost(&MergedClusterLabel_Host, sizeof(int) * width * height);
        cudaMalloc(&ref, sizeof(int) * width * height);
		cudaMallocHost(&ref_host, sizeof(int) * width * height);
		cudaMalloc(&MergedClusterEigenvalues_Device, sizeof(float) * width * height);
        cudaMalloc(&merged_cluster_size, sizeof(int) * width * height);
		cudaMalloc(&sum_of_merged_cluster_normals, sizeof(float3) * width * height);
		cudaMalloc(&sum_of_merged_cluster_centers, sizeof(float3) * width * height);
		cudaMalloc(&sum_of_merged_cluster_eigenvalues, sizeof(float) * width * height);

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


LabelEquivalenceSegPCA::~LabelEquivalenceSegPCA(){
		cudaFree(InputND_Device);
        cudaFree(MergedClusterND_Device);
		cudaFree(MergedClusterND_Host);
		cudaFree(MergedClusterLabel_Device);
		cudaFree(MergedClusterLabel_Host);
		cudaFree(MergedClusterEigenvalues_Device);
        cudaFree(ref);
		cudaFree(ref_host);
        cudaFree(merged_cluster_size);
		cudaFree(sum_of_merged_cluster_normals);
		cudaFree(sum_of_merged_cluster_centers);
		cudaFree(sum_of_merged_cluster_eigenvalues);
        show.~Mat_<cv::Vec3b>();
        delete [] color_pool;
}
void LabelEquivalenceSegPCA::releaseVideo(){
	Writer.release();
}
void LabelEquivalenceSegPCA::viewSegmentResult(){
		
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
        cv::waitKey(1);
}

float4* LabelEquivalenceSegPCA::getMergedClusterND_Device()const{
        return MergedClusterND_Device;
}
int* LabelEquivalenceSegPCA::getMergedClusterLabel_Device()const{
	return MergedClusterLabel_Device;
}
float4* LabelEquivalenceSegPCA::getMergedClusterND_Host()const{
        return MergedClusterND_Host;
}
int* LabelEquivalenceSegPCA::getMergedClusterLabel_Host()const{
	return MergedClusterLabel_Host;
}
float* LabelEquivalenceSegPCA::getMergedClusterEigenvalues_Device()const{
	return MergedClusterEigenvalues_Device;
}
cv::Mat_<cv::Vec3b> LabelEquivalenceSegPCA::getSegmentResult(){
		
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