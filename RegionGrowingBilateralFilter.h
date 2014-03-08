#ifndef BGRF_H
#define RGBF_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>

class DepthAdaptiveSuperpixel;
class EdgeRefinedSuperpixel;

class RegionGrowingBilateralFilter{
public:
	RegionGrowingBilateralFilter(int width, int height);
	~RegionGrowingBilateralFilter();
	void SetParametor(int rows, int cols, cv::Mat_<double> intrinsic);
	void Process(float* depth_device, float3* points_device, cv::gpu::GpuMat color_device);
	float*	getRefinedDepth_Device();
	float*	getRefinedDepth_Host();
private:
	int Width;
	int Height;
	int sp_cols;
	int sp_rows;
	DepthAdaptiveSuperpixel* DASP;
	DepthAdaptiveSuperpixel* SP;
	EdgeRefinedSuperpixel* ERS;
};
#endif