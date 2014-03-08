#ifndef EDGE_REFINED_SUPERPIXEL_H
#define EDGE_REFINED_SUPERPIXEL_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <XnCppWrapper.h>
#include <opencv2\opencv.hpp>
#include "thrust\fill.h"
#include "thrust\device_ptr.h"
#include "thrust\iterator\zip_iterator.h"
#include "thrust\transform.h"
#include <opencv2\gpu\gpu.hpp>


class EdgeRefinedSuperpixel{
public:	
	EdgeRefinedSuperpixel(int width, int height);	
	~EdgeRefinedSuperpixel();
	void EdgeRefining(int* color_label_device, int* depth_label_device, float* depth_device, cv::gpu::GpuMat color_image); 
	int*	getRefinedLabels_Device();
	int*	getRefinedLabels_Host();
	float*	getRefinedDepth_Device();
	float*	getRefinedDepth_Host();
	cv::Mat_<cv::Vec3b> getSegmentedImage(const int max_depth);
private:
	static const int	WindowSize;
	static const float	SpatialSigma;
	static const float  ColorSigma;
	static const float	DepthSigma;
	int Width;
	int Height;
	float*				SpatialFilter_Host;
	float*				SpatialFilter_Device;
	int*	refinedLabels_Device;
	int*	refinedLabels_Host;
	float*	refinedDepth_Device;
	float*	refinedDepth_Host;
	void				calcSpatialFilter();
	cv::Mat_<cv::Vec3b> segmentedImage;
	void getRGB(float ratio, cv::Vec3b& color);
	//int*	colorLabels_Host;
	//int*	depthLabels_Host;
	//float*	Depth_Host;
};
#endif