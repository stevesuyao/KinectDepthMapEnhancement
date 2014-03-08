#ifndef MARKOV_RANDOM_FIELD_H
#define MARKOV_RANDOM_FIELD_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>

class MarkovRandomField{
public:
	MarkovRandomField(int width, int height);
	~MarkovRandomField();
	void Process(float* depth_device, cv::gpu::GpuMat color_image);
	//void Upsampling(float* depthlow_device, cv::gpu::GpuMat colorhigh_image);
	float*				getFiltered_Device()const;
	float*				getFiltered_Host()const;
	cv::gpu::GpuMat		getSmoothImage_Device();
	void				visualize(float* depth_host);
private:
	int					Width;
	int					Height;
	cv::gpu::GpuMat		smooth_Device;
	cv::Mat_<cv::Vec3b> smooth_Host;
	cv::Mat_<cv::Vec3b> InputDepth;
	cv::Mat_<cv::Vec3b> OutputDepth;
	static const int	WindowSize;
	//static const float	SpatialSigma;
	static const float  ColorSigma;
	static const float	SmoothSigma;
	//float*				SpatialFilter_Host;
	//float*				SpatialFilter_Device;
	float*				Filtered_Host;
	float*				Filtered_Device;
	//void				calcSpatialFilter();
	void				getRGB(float ratio, cv::Vec3b& color);
};
#endif 