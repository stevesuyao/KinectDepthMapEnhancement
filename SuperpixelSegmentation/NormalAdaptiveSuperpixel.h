#ifndef NASP_H
#define NASP_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <XnCppWrapper.h>
#include <opencv2\opencv.hpp>
#include "thrust\fill.h"
#include "thrust\device_ptr.h"
#include "thrust\iterator\zip_iterator.h"
#include "thrust\transform.h"
#include <opencv2\gpu\gpu.hpp>
#include "DepthAdaptiveSuperpixel.h"


class NormalAdaptiveSuperpixel: public DepthAdaptiveSuperpixel{
public:	
	NormalAdaptiveSuperpixel(int width, int height);	
	~NormalAdaptiveSuperpixel();
	void SetParametor(int rows, int cols, cv::Mat_<double> intrinsic);
	void Segmentation(cv::gpu::GpuMat color_image, float3* points3d_device, float3* normals_device, 
						float color_sigma, float spatial_sigma, float depth_sigma, float normal_sigma, int iteration);
	cv::Mat_<cv::Vec3b>		getNormalImg();
	float3*		getCentersHost();
	float3*		getCentersDevice();
	float3*		getNormalsHost();
	float3*		getNormalsDevice();
	float*		getNormalsVarianceHost();
	float*		getNormalsVarianceDevice();
private:
	cv::Mat_<cv::Vec3b>			normalImage;
	cv::gpu::GpuMat				Intrinsic_Device;
	float3*						superpixelCenters_Host;
	float3*						superpixelCenters_Device;
	float3*						superpixelNormals_Host;
	float3*						superpixelNormals_Device;
	float*						NormalsVariance_Host;
	float*						NormalsVariance_Device;
	void  initMemory();
};
#endif