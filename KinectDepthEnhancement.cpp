#include "KinectDepthEnhancement.h"
#include "SuperpixelSegmentation\NormalAdaptiveSuperpixel.h"
#include "EdgeRefinedSuperpixel\EdgeRefinedSuperpixel.h"
#include "DimensionConvertor\DimensionConvertor.h"
#include "NormalEstimation\NormalMapGenerator.h"
#include "LabelEquivalenceSeg\LabelEquivalenceSeg.h"
#include "Projection_GPU\Projection_GPU.h"

KinectDepthEnhancement::KinectDepthEnhancement(int width, int height):
	Width(width),
	Height(height){
		NASP = new NormalAdaptiveSuperpixel(width, height);
		SP = new NormalAdaptiveSuperpixel(width, height);
		ERS = new EdgeRefinedSuperpixel(width, height);
		Convertor = new DimensionConvertor();
		NormalGenerator = new NormalMapGenerator(width, height);
		spMerging = new LabelEquivalenceSeg(width, height);
		cudaMalloc(&EdgeEnhanced3DPoints_Device, width * height * sizeof(float3));
	
	}
KinectDepthEnhancement::~KinectDepthEnhancement(){
	delete NASP;
	NASP = 0;
	delete SP;
	SP = 0;
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
	NASP->SetParametor(sp_rows, sp_cols, intrinsic);
	Convertor->setCameraParameters(intrinsic, Width, Height);
	NormalGenerator->setNormalEstimationMethods(NormalGenerator->CM);
	Projector = new Projection_GPU(Width, Height, intrinsic);
}
void KinectDepthEnhancement::Process(float* depth_device, float3* points_device, cv::gpu::GpuMat color_device){
	//normal estimation
	NormalGenerator->generateNormalMap(points_device);
	//segmentation
	SP->Segmentation(color_device, points_device, NormalGenerator->getNormalMap(), 200.0f, 10.0f, 0.0f, 0.0f, 5);
	NASP->Segmentation(color_device, points_device, NormalGenerator->getNormalMap(), 100.0f, 50.0f, 100.0f, 200.0f, 5);
	//edge refinement
	ERS->EdgeRefining(SP->getLabelDevice(), NASP->getLabelDevice(), depth_device, color_device);
	//convert to realworld
	Convertor->projectiveToReal(ERS->getRefinedDepth_Device(), EdgeEnhanced3DPoints_Device);
	//superpixel merging
	spMerging->labelImage(NASP->getNormalsDevice(), ERS->getRefinedLabels_Device(), NASP->getCentersDevice(), NASP->getNormalsVarianceDevice());
	//plane projection
	Projector->PlaneProjection(spMerging->getMergedClusterND_Device(), spMerging->getMergedClusterLabel_Device(), spMerging->getMergedClusterVariance_Device(), EdgeEnhanced3DPoints_Device);
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