#include "Projection_PCA.h"

Projection_PCA::Projection_PCA(int width, int height, const cv::Mat intrinsic){
	this->width = width;
	this->height = height;
	//Ε_£focal_length
	Fx = (float)intrinsic.at<double>(0, 0);
	Fy = (float)intrinsic.at<double>(1, 1);
	//ζS
	Cx = (int)intrinsic.at<double>(0, 2);
	Cy = (int)intrinsic.at<double>(1, 2);
	initMemory();
	initNormalized3D();
}

Projection_PCA::~Projection_PCA(){
	cudaFree(PlaneFitted3D_Host);
	cudaFree(PlaneFitted3D_Device);
	cudaFree(Normalized3D_Device);
	cudaFree(Optimized3D_Host);
	cudaFree(Optimized3D_Device);
}


void Projection_PCA::initMemory(){	
	cudaMallocHost(&PlaneFitted3D_Host, width * height * sizeof(float3));
	cudaMalloc(&PlaneFitted3D_Device, width * height * sizeof(float3));
	cudaMalloc(&Normalized3D_Device, width * height * sizeof(float3));
	cudaMallocHost(&Optimized3D_Host, width * height * sizeof(float3));
	cudaMalloc(&Optimized3D_Device, width * height * sizeof(float3));
}
float3*	Projection_PCA::GetPlaneFitted3D_Host()const{
	return PlaneFitted3D_Host;
}

float3*	Projection_PCA::GetPlaneFitted3D_Device()const{
	return PlaneFitted3D_Device;
}
float3*	Projection_PCA::GetOptimized3D_Host()const{
	return Optimized3D_Host;
}
float3*	Projection_PCA::GetOptimized3D_Device()const{
	return Optimized3D_Device;
}
