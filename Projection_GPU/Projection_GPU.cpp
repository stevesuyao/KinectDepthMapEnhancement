#include "Projection_GPU.h"

const float Projection_GPU::SpatialSigma = 20.0f;
const int Projection_GPU::WindowSize = 7;
const float Projection_GPU::DepthSigma = 100.0f;

Projection_GPU::Projection_GPU(int width, int height, const cv::Mat intrinsic){
	this->width = width;
	this->height = height;
	//è≈ì_ãóó£focal_length
	Fx = (float)intrinsic.at<double>(0, 0);
	Fy = (float)intrinsic.at<double>(1, 1);
	//âÊëúíÜêS
	Cx = (int)intrinsic.at<double>(0, 2);
	Cy = (int)intrinsic.at<double>(1, 2);
	initMemory();
	initNormalized3D();
	SpatialFilter_Host = new float[WindowSize*WindowSize];
	cudaMalloc(&SpatialFilter_Device, sizeof(float)*WindowSize*WindowSize);
	calcSpatialFilter();
}

Projection_GPU::~Projection_GPU(){
	cudaFree(PlaneFitted3D_Host);
	cudaFree(PlaneFitted3D_Device);
	cudaFree(Normalized3D_Device);
	cudaFree(Optimized3D_Host);
	cudaFree(Optimized3D_Device);
	delete [] SpatialFilter_Host;
	SpatialFilter_Host = 0;
	cudaFree(SpatialFilter_Device);
	SpatialFilter_Device = 0;
}

void Projection_GPU::calcSpatialFilter(){
	for(int i=0; i<WindowSize; i++){
		for(int j=0; j< WindowSize; j++){
			float dis_x = powf((float)(j - WindowSize/2), 2.0f);
			float dis_y = powf((float)(i - WindowSize/2), 2.0f);
			SpatialFilter_Host[i*WindowSize+j] = expf(-(dis_x+dis_y)/(2.0f*powf(SpatialSigma, 2.0f)));
		}
	}
	cudaMemcpy(SpatialFilter_Device, SpatialFilter_Host, sizeof(float)*WindowSize*WindowSize, cudaMemcpyHostToDevice);
}
void Projection_GPU::initMemory(){	
	cudaMallocHost(&PlaneFitted3D_Host, width * height * sizeof(float3));
	cudaMalloc(&PlaneFitted3D_Device, width * height * sizeof(float3));
	cudaMalloc(&Normalized3D_Device, width * height * sizeof(float3));
	cudaMallocHost(&Optimized3D_Host, width * height * sizeof(float3));
	cudaMalloc(&Optimized3D_Device, width * height * sizeof(float3));
}
float3*	Projection_GPU::GetPlaneFitted3D_Host()const{
	return PlaneFitted3D_Host;
}

float3*	Projection_GPU::GetPlaneFitted3D_Device()const{
	return PlaneFitted3D_Device;
}
float3*	Projection_GPU::GetOptimized3D_Host()const{
	return Optimized3D_Host;
}
float3*	Projection_GPU::GetOptimized3D_Device()const{
	return Optimized3D_Device;
}
