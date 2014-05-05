#ifndef PROJECTION_PCA_H
#define PROJECTION_PCA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2\opencv.hpp>

class Projection_PCA{
public:
	Projection_PCA(int width, int height, const cv::Mat intrinsic);
	~Projection_PCA();
	void					PlaneProjection(const float4* nd_device, const int* labels_device, const float* eigenvalues_device, const float3* points3d_device);	
	float3*					GetPlaneFitted3D_Host()const;
	float3*					GetPlaneFitted3D_Device()const;
	float3*					GetOptimized3D_Host()const;
	float3*					GetOptimized3D_Device()const;
private:
	void initMemory();
	void getProjectedMap();
	void initNormalized3D();
	
	int width;
	int height;
	float Fx;
	float Fy;
	int Cx;
	int Cy;
	float3*					PlaneFitted3D_Host;
	float3*					PlaneFitted3D_Device;
	float3*					Normalized3D_Device;
	float3*					Optimized3D_Host;
	float3*					Optimized3D_Device;

};
#endif