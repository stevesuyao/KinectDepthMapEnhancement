#ifndef PROJECTION_GPU_H
#define PROJECTION_GPU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2\opencv.hpp>

class Projection_GPU{
public:
	Projection_GPU(int width, int height, const cv::Mat intrinsic);
	~Projection_GPU();
	void					PlaneProjection(const float4* nd_device, const int* labels_device, const float* variance_device, const float3* points3d_device);	
	void					PlaneProjection(const float3* normals_device, const float3* centers_device, const int* labels_device, 
													const float* variance_device, const float3* points3d_device);	
	float3*					GetPlaneFitted3D_Host()const;
	float3*					GetPlaneFitted3D_Device()const;
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

};
#endif