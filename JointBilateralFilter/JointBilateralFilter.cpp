#include "JointBilateralFilter.h"

const int JointBilateralFilter::WindowSize = 5;
const float JointBilateralFilter::SpatialSigma = 70.0f;
const float JointBilateralFilter::ColorSigma = 50.0f;
const float JointBilateralFilter::DepthSigma = 20.0f;

JointBilateralFilter::JointBilateralFilter(int width, int height):
	Width(width),
	Height(height),
	InputDepth(height, width),
	OutputDepth(height, width),
	smooth_Device(cv::gpu::createContinuous(height, width, CV_8UC3)),
	smooth_Host(height, width){
	SpatialFilter_Host = new float[WindowSize*WindowSize];
	cudaMalloc(&SpatialFilter_Device, sizeof(float)*WindowSize*WindowSize);
	cudaMallocHost(&Filtered_Host, sizeof(float)*Width*Height);
	cudaMalloc(&Filtered_Device, sizeof(float)*Width*Height);
	calcSpatialFilter();
	}
JointBilateralFilter::~JointBilateralFilter(){
	delete [] SpatialFilter_Host;
	SpatialFilter_Host = 0;
	cudaFree(SpatialFilter_Device);
	SpatialFilter_Device = 0;
	cudaFree(Filtered_Host);
	Filtered_Host = 0;
	cudaFree(Filtered_Device);
	Filtered_Device = 0;
}
void JointBilateralFilter::calcSpatialFilter(){
	for(int i=0; i<WindowSize; i++){
		for(int j=0; j< WindowSize; j++){
			float dis_x = powf((float)(j - WindowSize/2), 2.0f);
			float dis_y = powf((float)(i - WindowSize/2), 2.0f);
			SpatialFilter_Host[i*WindowSize+j] = expf(-(dis_x+dis_y)/(2.0f*powf(SpatialSigma, 2.0f)));
		}
	}
	cudaMemcpy(SpatialFilter_Device, SpatialFilter_Host, sizeof(float)*WindowSize*WindowSize, cudaMemcpyHostToDevice);
}
float* JointBilateralFilter::getFiltered_Device()const{
	return Filtered_Device;
}
float* JointBilateralFilter::getFiltered_Host()const{
	cudaMemcpy(Filtered_Host, Filtered_Device, sizeof(float)*Width*Height, cudaMemcpyDeviceToHost);
	return Filtered_Host;
}
cv::gpu::GpuMat	JointBilateralFilter::getSmoothImage_Device(){
	return smooth_Device;
}
void JointBilateralFilter::visualize(float* depth_host){

	cudaMemcpy(Filtered_Host, Filtered_Device, sizeof(float)*Width*Height, cudaMemcpyDeviceToHost);
	for(int y = 0; y < Height; y++){
		for(int x = 0; x < Width; x++){
			cv::Vec3b color;
			if(depth_host[y * Width + x] > 50.0f){
				getRGB(depth_host[y * Width + x]/ 5000.0f, color);
				InputDepth.at<cv::Vec3b>(y, x) = color;
			}
			else
				InputDepth.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			
			}
		}
	for(int y = 0; y < Height; y++){
		for(int x = 0; x < Width; x++){
			cv::Vec3b color2;
			if(Filtered_Host[y * Width + x] > 50.0f){
				getRGB(Filtered_Host[y * Width + x]/ 5000.0f, color2);
				OutputDepth.at<cv::Vec3b>(y, x) = color2;
			}
			else
				OutputDepth.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			}
		}
	cv::imshow("depth_input", InputDepth);
	cv::imshow("depth_filtered", OutputDepth);
	cv::waitKey(1);
}

void JointBilateralFilter::getRGB(float ratio, cv::Vec3b& color){
	ratio = ratio >= 0.99f ? 0.99f : ratio;
	if(ratio < 0.33f){
		color.val[0] = (unsigned char)(ratio/0.33f * 255.0f);
	} else if(ratio < 0.66f){
		color.val[0] = (unsigned char)((0.66f-ratio)/0.33f * 255.0f);
		color.val[1] = (unsigned char)((ratio-=0.33f)/0.33f * 255.0f);
	} else {
		color.val[1] = (unsigned char)((0.99f-ratio)/0.33f * 255.0f);
		color.val[2] = (unsigned char)((ratio-=0.66f)/0.33f * 255.0f);
	}
}
