#include "KinectDepthEnhancement.h"
#include "SuperpixelSegmentation\NormalAdaptiveSuperpixel.h"
#include "EdgeRefinedSuperpixel\EdgeRefinedSuperpixel.h"
#include "DimensionConvertor\DimensionConvertor.h"
#include "NormalEstimation\NormalMapGenerator.h"
#include "LabelEquivalenceSeg\LabelEquivalenceSeg.h"
#include "Projection_GPU\Projection_GPU.h"
#include "SuperpixelSegmentation\DepthAdaptiveSuperpixel.h"
#include "JointBilateralFilter\JointBilateralFilter.h"

KinectDepthEnhancement::KinectDepthEnhancement(int width, int height):
	Width(width),
	Height(height){
		SP = new DepthAdaptiveSuperpixel(width, height);
		DASP = new DepthAdaptiveSuperpixel(width, height);
		NASP = new NormalAdaptiveSuperpixel(width, height);
		JBF = new JointBilateralFilter(width, height);
		ERS = new EdgeRefinedSuperpixel(width, height);
		Convertor = new DimensionConvertor();
		NormalGenerator = new NormalMapGenerator(width, height);
		spMerging = new LabelEquivalenceSeg(width, height);
		cudaMalloc(&EdgeEnhanced3DPoints_Device, width * height * sizeof(float3));
	
	}
KinectDepthEnhancement::~KinectDepthEnhancement(){
	delete SP;
	SP = 0;
	delete DASP;
	DASP = 0;
	delete NASP;
	NASP = 0;
	delete JBF;
	JBF = 0;
	delete ERS;
	ERS = 0;
	delete NormalGenerator;
	NormalGenerator = 0;
	delete spMerging;
	spMerging = 0;
	delete Projector;
	Projector = 0;
	cudaFree(EdgeEnhanced3DPoints_Device);


}
void KinectDepthEnhancement::SetParametor(int rows, int cols, cv::Mat_<double> intrinsic){
	sp_rows = rows;
	sp_cols = cols;
	SP->SetParametor(sp_rows, sp_cols, intrinsic);
	DASP->SetParametor(sp_rows, sp_cols, intrinsic);
	NASP->SetParametor(sp_rows, sp_cols, intrinsic);
	Convertor->setCameraParameters(intrinsic, Width, Height);
	NormalGenerator->setNormalEstimationMethods(NormalGenerator->CM);
	Projector = new Projection_GPU(Width, Height, intrinsic);
}
void KinectDepthEnhancement::Process(float* depth_device, cv::gpu::GpuMat color_device){
	//filtering
	JBF->Process(depth_device, color_device);
	//convert to realworld
	Convertor->projectiveToReal(JBF->getFiltered_Device(), EdgeEnhanced3DPoints_Device);
	//segmentation
	//SP->Segmentation(color_device, EdgeEnhanced3DPoints_Device, 200.0f, 1.0f, 0.0f, 5);
	//DASP->Segmentation(color_device, EdgeEnhanced3DPoints_Device, 0.0f, 10.0f, 200.0f, 5);
	//normal estimation
	NormalGenerator->generateNormalMap(EdgeEnhanced3DPoints_Device);
	//cv::imwrite("normalImage.jpg", NormalGenerator->getNormalImg());
	NASP->Segmentation(color_device, EdgeEnhanced3DPoints_Device, NormalGenerator->getNormalMap(), 10.0f, 50.0f, 50.0f, 150.0f, 1);

	cv::imwrite("segmentation.jpg", NASP->getRandomColorImage());
	//edge refinement
	//ERS->EdgeRefining(SP->getLabelDevice(), NASP->getLabelDevice(), depth_device, color_device);
	//convert to realworld
	//Convertor->projectiveToReal(ERS->getRefinedDepth_Device(), EdgeEnhanced3DPoints_Device);
	//superpixel merging
	//spMerging->labelImage(NASP->getNormalsDevice(), ERS->getRefinedLabels_Device(), NASP->getCentersDevice(), NASP->getNormalsVarianceDevice());
	spMerging->labelImage(NASP->getNormalsDevice(), NASP->getLabelDevice(), NASP->getCentersDevice(), NASP->getNormalsVarianceDevice());
	cv::imwrite("labelImage.jpg", spMerging->getSegmentResult());
	//plane projection
	Projector->PlaneProjection(spMerging->getMergedClusterND_Device(), spMerging->getMergedClusterLabel_Device(), 
		spMerging->getMergedClusterVariance_Device(), EdgeEnhanced3DPoints_Device, spMerging->getMergedClusterSize_Device());
}
float*	KinectDepthEnhancement::getRefinedDepth_Device(){
	return ERS->getRefinedDepth_Device();
}
float*	KinectDepthEnhancement::getRefinedDepth_Host(){
	return ERS->getRefinedDepth_Host();
}
float3*	KinectDepthEnhancement::getOptimizedPoints_Device(){
	return Projector->GetOptimized3D_Device();
}
float3*	KinectDepthEnhancement::getOptimizedPoints_Host(){
	return Projector->GetOptimized3D_Host();
}