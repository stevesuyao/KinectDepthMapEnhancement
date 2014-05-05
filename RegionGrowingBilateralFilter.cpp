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
	SP->SetParametor(sp_rows, sp_cols, intrinsic);
	DASP->SetParametor(sp_rows, sp_cols, intrinsic);
}
void RegionGrowingBilateralFilter::Process(float* depth_device, float3* points_device, cv::gpu::GpuMat color_device){
	SP->Segmentation(color_device, points_device, 200.0f, 40.0f, 0.0f, 1);
	DASP->Segmentation(color_device, points_device, 100.0f, 20.0f, 200.0f, 1);
	//edge refinement
	ERS->EdgeRefining(SP->getLabelDevice(), DASP->getLabelDevice(), depth_device, color_device);
	//cv::Mat_<cv::Vec3b> color(Height, Width);
	//color_device.download(color);
	//cv::imshow("color_segmentation", SP->getSegmentedImage(color, SuperpixelSegmentation::Line));
	//cv::imshow("color_segmentation", SP->getRandomColorImage());
	//cv::imshow("depth_segmentation", DASP->getRandomColorImage());
	//cv::imshow("enhanced_segmenatation", ERS->getRandomColorImage());
}
float*	RegionGrowingBilateralFilter::getRefinedDepth_Device(){
	return ERS->getRefinedDepth_Device();
}
float*	RegionGrowingBilateralFilter::getRefinedDepth_Host(){
	return ERS->getRefinedDepth_Host();
}
	