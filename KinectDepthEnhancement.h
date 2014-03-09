#ifndef KINECTDEPTHENHANCEMENT_H
#define KINECTDEPTHENHANCEMENT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>

class NormalAdaptiveSuperpixel;
class EdgeRefinedSuperpixel;
class DimensionConvertor;
class NormalMapGenerator;
class LabelEquivalenceSeg;
class Projection_GPU;

class KinectDepthEnhancement{
public:
	KinectDepthEnhancement(int width, int height);
	~KinectDepthEnhancement();
	void SetParametor(int rows, int cols, cv::Mat_<double> intrinsic);
	void Process(float* depth_device, float3* points_device, cv::gpu::GpuMat color_device);
	float*	getRefinedDepth_Device();
	float*	getRefinedDepth_Host();
	float3*	getOptimizedPoints_Device();
	float3*	getOptimizedPoints_Host();
private:
	int Width;
	int Height;
	int sp_cols;
	int sp_rows;
	float3* EdgeEnhanced3DPoints_Device;
	NormalAdaptiveSuperpixel* NASP;
	NormalAdaptiveSuperpixel* SP;
	EdgeRefinedSuperpixel* ERS;
	DimensionConvertor* Convertor;
	NormalMapGenerator* NormalGenerator;
	LabelEquivalenceSeg* spMerging;
	Projection_GPU* Projector;
};
#endif