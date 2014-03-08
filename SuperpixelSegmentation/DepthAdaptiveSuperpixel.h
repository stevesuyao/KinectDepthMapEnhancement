#ifndef DASP_H
#define DASP_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <XnCppWrapper.h>
#include <opencv2\opencv.hpp>
#include "thrust\fill.h"
#include "thrust\device_ptr.h"
#include "thrust\iterator\zip_iterator.h"
#include "thrust\transform.h"
#include <opencv2\gpu\gpu.hpp>
#include "SuperpixelSegmentation.h"


class DepthAdaptiveSuperpixel: public SuperpixelSegmentation{
public:	
	
	DepthAdaptiveSuperpixel(int width, int height);	
	virtual ~DepthAdaptiveSuperpixel();
	void virtual SetParametor(int rows, int cols, cv::Mat_<double> intrinsic);
	void virtual Segmentation(cv::gpu::GpuMat color_image, float3* points3d_device, float color_sigma, float spatial_sigma, float depth_sigma, int iteration);
protected:
	cv::gpu::GpuMat				Intrinsic_Device;
	float3*						superpixelCenters_Host;
	float3*						superpixelCenters_Device;
	void						initMemory();
};
#endif