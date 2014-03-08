#include "RegionGrowingBilateralFilter.h"
#include "SuperpixelSegmentation\DepthAdaptiveSuperpixel.h"
#include "EdgeRefinedSuperpixel\EdgeRefinedSuperpixel.h"

RegionGrowingBilateralFilter::RegionGrowingBilateralFilter(int width, int height):
	Width(width),
	Height(height){
		DASP = new DepthAdaptiveSuperpixel(width, height);
		SP = new DepthAdaptiveSuperpixel(width, height);
		ERS = new EdgeRefinedSuperpixel(width, height);
	}
RegionGrowingBilateralFilter::~RegionGrowingBilateralFilter(){
	delete DASP;
	DASP = 0;
	delete SP;
	SP = 0;
	delete ERS;
	ERS = 0;

}
void RegionGrowingBilateralFilter::SetParametor(int rows, int cols, cv::Mat_<double> intrinsic){
	sp_rows = rows;
	sp_cols = cols;
	DASP->SetParametor(sp_rows, sp_cols, intrinsic);
}
void RegionGrowingBilateralFilter::Process(float* depth_device, float3* points_device, cv::gpu::GpuMat color_device){
	SP->Segmentation(color_device, points_device, 200.0f, 10.0f, 0.0f, 5);
	DASP->Segmentation(color_device, points_device, 0.0f, 10.0f, 200.0f, 5);
	//edge refinement
	ERS->EdgeRefining(SP->getLabelDevice(), DASP->getLabelDevice(), depth_device, color_device);
}
float*	RegionGrowingBilateralFilter::getRefinedDepth_Device(){
	return ERS->getRefinedDepth_Device();
}
float*	RegionGrowingBilateralFilter::getRefinedDepth_Host(){
	return ERS->getRefinedDepth_Host();
}
	