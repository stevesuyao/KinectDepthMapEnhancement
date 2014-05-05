#include "Kinect/Kinect.h"
#include "SuperpixelSegmentation/SuperpixelSegmentation.h"
#include "SuperpixelSegmentation/DepthAdaptiveSuperpixel.h"
#include "JointBilateralFilter/JointBilateralFilter.h"
#include "DimensionConvertor/DimensionConvertor.h"
#include <time.h>
#include "NormalEstimation\NormalMapGenerator.h"
#include "SuperpixelSegmentation/NormalAdaptiveSuperpixel.h"
#include "ArrayBuffer\Buffer2D.h"
#include "LabelEquivalenceSeg/LabelEquivalenceSeg.h"
#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>
#include "Projection_GPU/Projection_GPU.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "EdgeRefinedSuperpixel/EdgeRefinedSuperpixel.h"
#include "MarkovRandomField/MarkovRandomField.h"
#include "RegionGrowingBilateralFilter.h"
#include "KinectDepthEnhancement.h"
#include "SPDepthSuperResolution.h"
#include "TOFDepthInterpolation.h"

float color_sigma = 150.0f;
float spatial_sigma = 55.0f;
float depth_sigma = 20.0f;
float normal_sigma = 50.0f;
int iteration = 2;
const int sp_rows = 15;
const int sp_cols = 20;
const bool capture = false; 


int main(){

	///////////////////////////////////////////////////initialization////////////////////////////////////////////////////////////
	SingleKinect kinect;
	//input buffer
	float* inputDepth_Host, *inputDepth_Device, *bufferDepth_Host, *bufferDepth_Device;
	float3* inputPoints_Host, *inputPoints_Device, *bufferPoints_Device, *bufferPoints_Host;

	cudaMallocHost(&inputDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMalloc(&inputDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&bufferDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMalloc(&bufferDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&inputPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&inputPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&bufferPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&bufferPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	//JBF, MRF, RGBF
	float3 *jbfPoints_Host, *jbfPoints_Device, *mrfPoints_Host, *mrfPoints_Device, *rgbfPoints_Host, *rgbfPoints_Device, *resultPoints_Host, *resultPoints_Device; 
	cudaMallocHost(&jbfPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&jbfPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&mrfPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&mrfPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&rgbfPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&rgbfPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&resultPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&resultPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height);

	cv::gpu::GpuMat Color_Device = cv::gpu::createContinuous(Kinect::Height, Kinect::Width, CV_8UC3);
	
	//array buffer
	Buffer2D Buffer(Kinect::Width, Kinect::Height);
	//joint bilateral filter
	JointBilateralFilter JBF(Kinect::Width, Kinect::Height);
	//markov random field
	MarkovRandomField MRF(Kinect::Width, Kinect::Height);
	//dimension convertor
	DimensionConvertor convertor;
	convertor.setCameraParameters(kinect.GetIntrinsicMatrix(), Kinect::Width, Kinect::Height);
	//region growing bilateral filter
	RegionGrowingBilateralFilter RGBF(Kinect::Width, Kinect::Height);
	RGBF.SetParametor(sp_rows, sp_cols, kinect.GetIntrinsicMatrix());
	//kinect depth map enhancement
	KinectDepthEnhancement KDE(Kinect::Width, Kinect::Height);
	KDE.SetParametor(sp_rows, sp_cols, kinect.GetIntrinsicMatrix());
	////super-pixel based depth image super-resolution
	//SPDepthSuperResolution SPDSP(Kinect::Width, Kinect::Height);
	//SPDSP.SetParametor(sp_rows, sp_cols, kinect.GetIntrinsicMatrix());
	////TOF depth interpolation
	//TOFDepthInterpolation TOF(Kinect::Width, Kinect::Height);
	//TOF.SetParametor(sp_rows, sp_cols, kinect.GetIntrinsicMatrix());
	//////////////////////////////////////////////////////capture///////////////////////////////////////////////////////////
	if(capture){
		int count=0;
		while(count<1000){
			kinect.UpdateContextAndData();
			//depth input
			for(int y=0; y<Kinect::Height; y++){
				for(int x=0; x<Kinect::Width; x++){
					inputDepth_Host[y*Kinect::Width+x] = (float)(*kinect.GetDepthMD())(x, y);
				}
			}
			cudaMemcpy(inputDepth_Device, inputDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height, cudaMemcpyHostToDevice);
			//array buffer
			//Buffer.insertData(inputDepth_Device);
			Buffer.updateData(inputDepth_Device);
			count++;
		}
		cv::Mat_<float> averaged_depth(Kinect::Height, Kinect::Width);
		cv::Mat_<float> depth(Kinect::Height, Kinect::Width);
		Buffer.getDepthMap(bufferDepth_Device);
		cudaMemcpy(bufferDepth_Host, bufferDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height, cudaMemcpyHostToDevice);
		for(int y=0; y<Kinect::Height; y++){
			for(int x=0; x<Kinect::Width; x++){
				averaged_depth.at<float>(y, x) = bufferDepth_Host[y*Kinect::Width+x];
				depth.at<float>(y, x) = inputDepth_Host[y*Kinect::Width+x];
			}
		}
		cv::FileStorage cvfs("experiment/depth.xml", CV_STORAGE_WRITE);
		cv::write(cvfs, "averaged_depth",averaged_depth); 
		cv::write(cvfs, "depth",depth); 
		cv::imwrite("experiment/color.jpg", kinect.GetColorImage());
	}

	///////////////////////////////////////////make noisy data from ground truth///////////////////////
	////read data
	//cv::Mat_<float> ground_truth(Kinect::Height, Kinect::Width);
	//cv::FileStorage cvfs3("experiment/groundtruth/groundtruth.xml", CV_STORAGE_READ);
	//cv::FileNode node3(cvfs3.fs, NULL);
	//cv::read(node3["depth"], ground_truth);
	//cv::Mat_<float> noise(Kinect::Height, Kinect::Width);
	//for(int y=0; y<Kinect::Height; y++){
	//	for(int x=0; x<Kinect::Width; x++){
	//		float depth_variance = 0.45*2.85*pow(ground_truth.at<float>(y, x)/10.0f, 2.0f)/10000.0f;//mm
	//		int random = rand()%((int)(depth_variance*200)) - (int)(depth_variance*100);
	//		float rand_noise = ((float)random)/100.0f;
	//		noise.at<float>(y, x) = ground_truth.at<float>(y,x) + rand_noise;
	//	}
	//}
	//cv::FileStorage cvfs2("experiment/groundtruth/depth.xml", CV_STORAGE_WRITE);
	//cv::write(cvfs2, "averaged_depth", ground_truth); 
	//cv::write(cvfs2, "depth", noise); 
	
	///////////////////////////////////////////////////////////////////////read data////////////////////////////////////////////////////////////////
	//evaluation file
	FILE* fp;
	fp = fopen("evaluation.txt", "w");
	//read data
	cv::Mat_<float> averaged_depth(Kinect::Height, Kinect::Width);
	cv::Mat_<float> depth(Kinect::Height, Kinect::Width);	
	cv::Mat_<cv::Vec3b> color(Kinect::Height, Kinect::Width);	

	cv::FileStorage cvfs("experiment/groundtruth/depth.xml", CV_STORAGE_READ);
	cv::FileNode node(cvfs.fs, NULL);
	cv::read(node["averaged_depth"], averaged_depth);
	cv::read(node["depth"], depth);
	color = cv::imread("experiment/groundtruth/color.jpg", 1);
	//insert data
	for(int y=0; y<Kinect::Height; y++){
		for(int x=0; x<Kinect::Width; x++){
			bufferDepth_Host[y*Kinect::Width+x] = averaged_depth.at<float>(y, x);
			inputDepth_Host[y*Kinect::Width+x] = depth.at<float>(y, x);
		}
	}
	fprintf(fp, "runtime \n");
	clock_t start = clock();
	cudaMemcpy(inputDepth_Device, inputDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height, cudaMemcpyHostToDevice);
	fprintf(fp, "cudaMemcpyrHostToDevice: %f\n", (float)(clock()-start));
	cudaMemcpy(bufferDepth_Device, bufferDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height, cudaMemcpyHostToDevice);
	Color_Device.upload(color);
	///////////////////////////////////////////////////////////////////processing//////////////////////////////////////////////
	//input	
	//project to real
	start = clock();
	convertor.projectiveToReal(inputDepth_Device, inputPoints_Device);
	fprintf(fp, "projectToReal: %f\n", (float)(clock()-start));
	start = clock();
	cudaMemcpy(inputPoints_Host, inputPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	fprintf(fp, "cudaMemcpyDeviceToHost: %f\n", (float)(clock()-start));
	//buffer	
	//project to real
	convertor.projectiveToReal(bufferDepth_Device, bufferPoints_Device);
	cudaMemcpy(bufferPoints_Host, bufferPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	//JBF
	start = clock();
	JBF.Process(inputDepth_Device, Color_Device);
	fprintf(fp, "JBF: %f\n", (float)(clock()-start));
	//project to real
	convertor.projectiveToReal(JBF.getFiltered_Device(), jbfPoints_Device);
	cudaMemcpy(jbfPoints_Host, jbfPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	//MRF
	start = clock();
	MRF.Process(inputDepth_Device, Color_Device);
	fprintf(fp, "MRF: %f\n", (float)(clock()-start));
	//project to real
	convertor.projectiveToReal(MRF.getFiltered_Device(), mrfPoints_Device);
	cudaMemcpy(mrfPoints_Host, mrfPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	//region growing bilateral filter
	start = clock();
	RGBF.Process(inputDepth_Device, inputPoints_Device, Color_Device);
	fprintf(fp, "RGBF: %f\n", (float)(clock()-start));
	//project to real
	convertor.projectiveToReal(RGBF.getRefinedDepth_Device(), rgbfPoints_Device);
	cudaMemcpy(rgbfPoints_Host, rgbfPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	//Denoising of kinect depth map using piecewise planar surface
	start = clock();
	KDE.Process(inputDepth_Device, Color_Device);
	fprintf(fp, "KDE: %f\n", (float)(clock()-start));
	cudaMemcpy(resultPoints_Host, KDE.getOptimizedPoints_Device(), sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	////Superpixel based depth image super-resolution
	//SPDSP.Process(inputDepth_Device, inputPoints_Device, Color_Device);
	//cudaMemcpy(inputPoints_Host, SPDSP.getOptimizedPoints_Device(), sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	////TOF depth interpolation
	////TOF.Process(inputDepth_Device, inputPoints_Device, Color_Device);
	//cudaMemcpy(inputPoints_Host, TOF.getOptimizedPoints_Device(), sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	//////////////////////////////////////////////////////////////////////output//////////////////////////////////////////////////////////////////
	//visualize
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr input (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr buffer (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr jbf (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr mrf (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbf (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGB>);
	int input_count(0), jbf_count(0), mrf_count(0), rgbf_count(0), result_count(0);
	float input_average(0.0f), jbf_average(0.0f), mrf_average(0.0f), rgbf_average(0.0f), result_average(0.0f);
	//convert to realworld
	for(int y=0; y<Kinect::Height; y++){
		for(int x=0; x<Kinect::Width; x++){
			pcl::PointXYZRGB point;
			point.r = color.at<cv::Vec3b>(y, x).val[2];
			point.g = color.at<cv::Vec3b>(y, x).val[1];
			point.b = color.at<cv::Vec3b>(y, x).val[0];
			
			if(bufferPoints_Host[y*Kinect::Width+x].z >50.0f && bufferPoints_Host[y*Kinect::Width+x].z < 15000.0f){
				point.x = bufferPoints_Host[y*Kinect::Width+x].x/1000.0f;
				point.y = bufferPoints_Host[y*Kinect::Width+x].y/1000.0f;
				point.z = -bufferPoints_Host[y*Kinect::Width+x].z/1000.0f;	
				buffer->push_back(point);
			}
			if(inputPoints_Host[y*Kinect::Width+x].z > 50.0f && inputPoints_Host[y*Kinect::Width+x].z < 15000.0f){
				point.x = inputPoints_Host[y*Kinect::Width+x].x/1000.0f;
				point.y = inputPoints_Host[y*Kinect::Width+x].y/1000.0f;
				point.z = -inputPoints_Host[y*Kinect::Width+x].z/1000.0f;	
				input->push_back(point);
				if(bufferPoints_Host[y*Kinect::Width+x].z >50.0f && bufferPoints_Host[y*Kinect::Width+x].z < 15000.0f){
					//input_average += fabs(inputPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z);
					input_average += sqrtf(pow(inputPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z, 2.0f) + 
												pow(inputPoints_Host[y*Kinect::Width+x].y-bufferPoints_Host[y*Kinect::Width+x].y, 2.0f) +
													pow(inputPoints_Host[y*Kinect::Width+x].x-bufferPoints_Host[y*Kinect::Width+x].x, 2.0f));
					input_count ++;
				}
			}
			if(jbfPoints_Host[y*Kinect::Width+x].z >50.0f && jbfPoints_Host[y*Kinect::Width+x].z < 15000.0f){
				point.x = jbfPoints_Host[y*Kinect::Width+x].x/1000.0f;
				point.y = jbfPoints_Host[y*Kinect::Width+x].y/1000.0f;
				point.z = -jbfPoints_Host[y*Kinect::Width+x].z/1000.0f;	
				jbf->push_back(point);
				if(bufferPoints_Host[y*Kinect::Width+x].z >50.0f && bufferPoints_Host[y*Kinect::Width+x].z < 15000.0f){
					//jbf_average += fabs(jbfPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z);
					jbf_average += sqrtf(pow(jbfPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z, 2.0f) + 
												pow(jbfPoints_Host[y*Kinect::Width+x].y-bufferPoints_Host[y*Kinect::Width+x].y, 2.0f) +
													pow(jbfPoints_Host[y*Kinect::Width+x].x-bufferPoints_Host[y*Kinect::Width+x].x, 2.0f));
					
					jbf_count ++;
				}
			}
			if(mrfPoints_Host[y*Kinect::Width+x].z >50.0f && mrfPoints_Host[y*Kinect::Width+x].z < 15000.0f){
				point.x = mrfPoints_Host[y*Kinect::Width+x].x/1000.0f;
				point.y = mrfPoints_Host[y*Kinect::Width+x].y/1000.0f;
				point.z = -mrfPoints_Host[y*Kinect::Width+x].z/1000.0f;	
				mrf->push_back(point);
				if(bufferPoints_Host[y*Kinect::Width+x].z >50.0f && bufferPoints_Host[y*Kinect::Width+x].z < 15000.0f){
					//mrf_average += fabs(mrfPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z);
					mrf_average += sqrtf(pow(mrfPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z, 2.0f) + 
												pow(mrfPoints_Host[y*Kinect::Width+x].y-bufferPoints_Host[y*Kinect::Width+x].y, 2.0f) +
													pow(mrfPoints_Host[y*Kinect::Width+x].x-bufferPoints_Host[y*Kinect::Width+x].x, 2.0f));
					mrf_count ++;
				}
			}
			if(rgbfPoints_Host[y*Kinect::Width+x].z >50.0f && rgbfPoints_Host[y*Kinect::Width+x].z < 15000.0f){
				point.x = rgbfPoints_Host[y*Kinect::Width+x].x/1000.0f;
				point.y = rgbfPoints_Host[y*Kinect::Width+x].y/1000.0f;
				point.z = -rgbfPoints_Host[y*Kinect::Width+x].z/1000.0f;	
				rgbf->push_back(point);
				if(bufferPoints_Host[y*Kinect::Width+x].z >50.0f && bufferPoints_Host[y*Kinect::Width+x].z < 15000.0f){
					//rgbf_average += fabs(rgbfPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z);
					rgbf_average += sqrtf(pow(rgbfPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z, 2.0f) + 
												pow(rgbfPoints_Host[y*Kinect::Width+x].y-bufferPoints_Host[y*Kinect::Width+x].y, 2.0f) +
													pow(rgbfPoints_Host[y*Kinect::Width+x].x-bufferPoints_Host[y*Kinect::Width+x].x, 2.0f));
					rgbf_count ++;
				}
			}
			if(resultPoints_Host[y*Kinect::Width+x].z >50.0f && resultPoints_Host[y*Kinect::Width+x].z < 15000.0f){
				point.x = resultPoints_Host[y*Kinect::Width+x].x/1000.0f;
				point.y = resultPoints_Host[y*Kinect::Width+x].y/1000.0f;
				point.z = -resultPoints_Host[y*Kinect::Width+x].z/1000.0f;	
				result->push_back(point);
				if(bufferPoints_Host[y*Kinect::Width+x].z >50.0f && bufferPoints_Host[y*Kinect::Width+x].z < 15000.0f){
					//result_average += fabs(resultPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z);
					result_average += sqrtf(pow(resultPoints_Host[y*Kinect::Width+x].z-bufferPoints_Host[y*Kinect::Width+x].z, 2.0f) + 
												pow(resultPoints_Host[y*Kinect::Width+x].y-bufferPoints_Host[y*Kinect::Width+x].y, 2.0f) +
													pow(resultPoints_Host[y*Kinect::Width+x].x-bufferPoints_Host[y*Kinect::Width+x].x, 2.0f));
					
					result_count ++;
				}
			}
		}
	}
	//calculate error
	fprintf(fp, "error \n");
	fprintf(fp, "input %f\n", input_average/(float)input_count);
	fprintf(fp, "jbf %f\n", jbf_average/(float)jbf_count);
	fprintf(fp, "mrf %f\n", mrf_average/(float)mrf_count);
	fprintf(fp, "rgbf %f\n", rgbf_average/(float)rgbf_count);
	fprintf(fp, "result %f\n", result_average/(float)result_count);
	fclose(fp);
	////visualize input and result
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Viewer"));
	//viewer->initCameraParameters ();
	//viewer->setBackgroundColor(255, 255, 255);
	//int v1 = 0;
	//viewer->createViewPort(0.0, 0.0, 0.499, 1.0, v1);
	//viewer->addText("input_cloud", 10, 10, "v1 text", v1);
	//viewer->setBackgroundColor (0, 0, 0, v1);
	//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_input(input);
	//viewer->addPointCloud<pcl::PointXYZRGB> (input, rgb_input, "input", v1);
	//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "input", v1);
	//int v2 = 0;
	//viewer->createViewPort(0.501, 0.0, 1.0, 1.0, v2);
	//viewer->addText("result_cloud", 10, 10, "v2 text", v2);
	//viewer->setBackgroundColor (0, 0, 0, v2);
	//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_result(result);
	//viewer->addPointCloud<pcl::PointXYZRGB> (result, rgb_result, "result", v2);
	//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "result", v2);

	////visualize jbf, mrf, rgbf, result
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Viewer"));
	//viewer->initCameraParameters ();
	//viewer->setBackgroundColor(255, 255, 255);
	//int v1 = 0;
	//viewer->createViewPort(0.0, 0.501, 0.499, 1.0, v1);
	//viewer->addText("mrf_cloud", 10, 10, "v1 text", v1);
	//viewer->setBackgroundColor (0, 0, 0, v1);
	//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_mrf(mrf);
	//viewer->addPointCloud<pcl::PointXYZRGB> (mrf, rgb_mrf, "mrf", v1);
	//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "mrf", v1);
	//int v2 = 0;
	//viewer->createViewPort(0.501, 0.501, 1.0, 1.0, v2);
	//viewer->addText("rgbf_cloud", 10, 10, "v2 text", v2);
	//viewer->setBackgroundColor (0, 0, 0, v2);
	//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_rgbf(rgbf);
	//viewer->addPointCloud<pcl::PointXYZRGB> (rgbf, rgb_rgbf, "rgbf", v2);
	//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "rgbf", v2);
	//int v3 = 0;
	//viewer->createViewPort(0.0, 0.0, 0.499, 0.499, v3);
	//viewer->addText("jbf_cloud", 10, 10, "v3 text", v3);
	//viewer->setBackgroundColor (0, 0, 0, v3);
	//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_jbf(jbf);
	//viewer->addPointCloud<pcl::PointXYZRGB> (jbf, rgb_jbf, "jbf", v3);
	//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "jbf", v3);
	//int v4 = 0;
	//viewer->createViewPort(0.501, 0.0, 1.0, 0.499, v4);
	//viewer->addText("result_cloud", 10, 10, "v4 text", v4);
	//viewer->setBackgroundColor (0, 0, 0, v4);
	//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_result(result);
	//viewer->addPointCloud<pcl::PointXYZRGB> (result, rgb_result, "result", v4);
	//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "result", v4);

	//visualize all
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Viewer"));
	viewer->initCameraParameters ();
	viewer->setBackgroundColor(255, 255, 255);
	int v1 = 0;
	viewer->createViewPort(0.331, 0.501, 0.669, 1.0, v1);
	viewer->addText("MRF", 10, 10, 30, 1.0, 1.0, 1.0, "v1 text", v1);
	viewer->setBackgroundColor (0, 0, 0, v1);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_mrf(mrf);
	viewer->addPointCloud<pcl::PointXYZRGB> (mrf, rgb_mrf, "mrf", v1);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "mrf", v1);
	int v2 = 0;
	viewer->createViewPort(0.671, 0.501, 1.0, 1.0, v2);
	viewer->addText("RGBF", 10, 10, 30, 1.0, 1.0, 1.0, "v2 text", v2);
	viewer->setBackgroundColor (0, 0, 0, v2);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_rgbf(rgbf);
	viewer->addPointCloud<pcl::PointXYZRGB> (rgbf, rgb_rgbf, "rgbf", v2);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "rgbf", v2);
	int v3 = 0;
	viewer->createViewPort(0.331, 0.0, 0.669, 0.499, v3);
	viewer->addText("JBF", 10, 10, 30, 1.0, 1.0, 1.0, "v3 text", v3);
	viewer->setBackgroundColor (0, 0, 0, v3);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_jbf(jbf);
	viewer->addPointCloud<pcl::PointXYZRGB> (jbf, rgb_jbf, "jbf", v3);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "jbf", v3);
	int v4 = 0;
	viewer->createViewPort(0.671, 0.0, 1.0, 0.499, v4);
	viewer->addText("PROPOSED", 10, 10, 30, 1.0, 1.0, 1.0, "v4 text", v4);
	viewer->setBackgroundColor (0, 0, 0, v4);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_result(result);
	viewer->addPointCloud<pcl::PointXYZRGB> (result, rgb_result, "result", v4);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "result", v4);
	int v5 = 0;
	viewer->createViewPort(0.0, 0.501, 0.329, 1.0, v5);
	viewer->addText("INPUT", 10, 10, 30, 1.0, 1.0, 1.0, "v5 text", v5);
	viewer->setBackgroundColor (0, 0, 0, v5);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_input(input);
	viewer->addPointCloud<pcl::PointXYZRGB> (input, rgb_input, "input", v5);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "input", v5);
	int v6 = 0;
	viewer->createViewPort(0.0, 0.0, 0.329, 0.499, v6);
	viewer->addText("GROUND TRUTH", 10, 10, 30, 1.0, 1.0, 1.0, "v6 text", v6);
	viewer->setBackgroundColor (0, 0, 0, v6);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_buffer(buffer);
	viewer->addPointCloud<pcl::PointXYZRGB> (buffer, rgb_buffer, "buffer", v6);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "buffer", v6);

	bool ShouldRun = true;
	while (ShouldRun) {  
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}			


}