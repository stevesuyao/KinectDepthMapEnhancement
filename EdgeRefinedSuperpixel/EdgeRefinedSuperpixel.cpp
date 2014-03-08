#include "EdgeRefinedSuperpixel.h"
#include <ctime>

const int EdgeRefinedSuperpixel::WindowSize = 15;
const float EdgeRefinedSuperpixel::SpatialSigma = 70.0f;
const float EdgeRefinedSuperpixel::ColorSigma = 50.0f;
const float EdgeRefinedSuperpixel::DepthSigma = 20.0f;

EdgeRefinedSuperpixel::EdgeRefinedSuperpixel(int width, int height):
	Width(width),
	Height(height),
	segmentedImage(Height, Width){
		SpatialFilter_Host = new float[WindowSize*WindowSize];
		cudaMalloc(&SpatialFilter_Device, sizeof(float)*WindowSize*WindowSize);
		cudaMallocHost(&refinedLabels_Host, sizeof(int)*width*height);
		cudaMalloc(&refinedLabels_Device, sizeof(int)*width*height);
		cudaMallocHost(&refinedDepth_Host, sizeof(float)*width*height);
		cudaMalloc(&refinedDepth_Device, sizeof(float)*width*height);
		calcSpatialFilter();
		//cudaMallocHost(&colorLabels_Host, sizeof(int)*width*height);
		//cudaMallocHost(&depthLabels_Host, sizeof(int)*width*height);
		//cudaMallocHost(&Depth_Host, sizeof(float)*width*height);
	}
EdgeRefinedSuperpixel::~EdgeRefinedSuperpixel(){
	delete [] SpatialFilter_Host;
	cudaFree(SpatialFilter_Device);
	cudaFree(refinedLabels_Host);
	cudaFree(refinedLabels_Device);
	cudaFree(refinedDepth_Host);
	cudaFree(refinedDepth_Device);
	//cudaFree(colorLabels_Host);
	//cudaFree(depthLabels_Host);
	//cudaFree(Depth_Host);
}
void EdgeRefinedSuperpixel::calcSpatialFilter(){
	for(int i=0; i<WindowSize; i++){
		for(int j=0; j< WindowSize; j++){
			float dis_x = powf((float)(j - WindowSize/2), 2.0f);
			float dis_y = powf((float)(i - WindowSize/2), 2.0f);
			SpatialFilter_Host[i*WindowSize+j] = expf(-(dis_x+dis_y)/(2.0f*powf(SpatialSigma, 2.0f)));
		}
	}
	cudaMemcpy(SpatialFilter_Device, SpatialFilter_Host, sizeof(float)*WindowSize*WindowSize, cudaMemcpyHostToDevice);
}
int* EdgeRefinedSuperpixel::getRefinedLabels_Device(){
	return refinedLabels_Device;
}
int* EdgeRefinedSuperpixel::getRefinedLabels_Host(){
	//cudaMemcpy(refinedLabels_Host, refinedLabels_Device, sizeof(int)*Width*Height, cudaMemcpyDeviceToHost);
	return refinedLabels_Host;
}
float* EdgeRefinedSuperpixel::getRefinedDepth_Device(){
	return refinedDepth_Device;
}
float* EdgeRefinedSuperpixel::getRefinedDepth_Host(){
	//cudaMemcpy(refinedDepth_Host, refinedDepth_Device, sizeof(float)*Width*Height, cudaMemcpyDeviceToHost);
	return refinedDepth_Host;
}

cv::Mat_<cv::Vec3b> EdgeRefinedSuperpixel::getSegmentedImage(const int max_depth){
	
	//cudaMemcpy(refinedLabels_Host, refinedLabels_Device, sizeof(int)*Width*Height, cudaMemcpyDeviceToHost);
	//input_host.copyTo(segmentedImage);
	//cudaMemcpy(refinedDepth_Host, refinedDepth_Device, sizeof(float)*Width*Height, cudaMemcpyDeviceToHost);
	//std::cout << max_depth <<std::endl;
	for(int y=0; y<Height; y++){
		for(int x=0; x<Width; x++){
			cv::Vec3b color;
			//getRGB(refinedDepth_Host[y * Width + x]/ (float)max_depth, color);
			getRGB(refinedDepth_Host[y * Width + x]/ 3000.0f, color);
			segmentedImage.at<cv::Vec3b>(y, x) = color;
		}
	}
			
	for(int y=0; y<Height-1; y++){
		for(int x=0; x<Width-1; x++){
			if(refinedDepth_Host[y*Width+x] == 0.0f)
				segmentedImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			if(refinedLabels_Host[y*Width+x] !=  refinedLabels_Host[(y+1)*Width+x])
				segmentedImage.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			if(refinedLabels_Host[y*Width+x] !=  refinedLabels_Host[y*Width+x+1])
				segmentedImage.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			if(refinedLabels_Host[y*Width+x] == -100)
				segmentedImage.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
		}
	}

	return segmentedImage;
}
void EdgeRefinedSuperpixel::getRGB(float ratio, cv::Vec3b& color){
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
