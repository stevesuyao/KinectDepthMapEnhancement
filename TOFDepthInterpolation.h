#ifndef TOF_DEPTH_INTERPOLATION_H
#define TOF_DEPTH_INTERPOLATION_H


#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>

class DepthAdaptiveSuperpixel;
class EdgeRefinedSuperpixel;
class DimensionConvertor;
class LabelEquivalenceSegPCA;
class Projection_PCA;
class Cluster;

class TOFDepthInterpolation{
public:
	TOFDepthInterpolation(int width, int height);
	~TOFDepthInterpolation();
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
	float3* EdgeEnhanced3DPoints_Host;
	DepthAdaptiveSuperpixel* DASP;
	DepthAdaptiveSuperpixel* SP;
	EdgeRefinedSuperpixel* ERS;
	DimensionConvertor* Convertor;
	LabelEquivalenceSegPCA* spMerging;
	Projection_PCA* Projector;
	Cluster* Cluster_Array;
	float4*							ClusterND_Host;					
	float4*							ClusterND_Device;
	float3*							ClusterCenter_Host;
	float3*							ClusterCenter_Device;
	float*							ClusterEigenvalues_Host;
	float*							ClusterEigenvalues_Device;
};


#endif