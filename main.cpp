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

float color_sigma = 250.0f;
float spatial_sigma = 15.0f;
float depth_sigma = 20.0f;
float normal_sigma = 80.0f;
int iteration = 5;
const int sp_rows = 10;
const int sp_cols = 20;
const bool capture = false; 

int main(){
	///////////////////////////////////////////////////initialization////////////////////////////////////////////////////////////
	SingleKinect kinect;
	float* inputDepth_Host, *inputDepth_Device, *bufferDepth_Host, *bufferDepth_Device;
	float3* points_Host, *points_Device, *refinedPoints_Device, *refinedPoints_Host;
	cudaMallocHost(&bufferDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMalloc(&bufferDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&inputDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMalloc(&inputDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&points_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&points_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&refinedPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&refinedPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
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
		cv::FileStorage cvfs("input/depth.xml", CV_STORAGE_WRITE);
		cv::write(cvfs, "averaged_depth",averaged_depth); 
		cv::write(cvfs, "depth",depth); 
		cv::imwrite("input/color.jpg", kinect.GetColorImage());
	}

	///////////////////////////////////////////////////////////////////////read data////////////////////////////////////////////////////////////////
	//read data
	cv::Mat_<float> averaged_depth(Kinect::Height, Kinect::Width);
	cv::Mat_<float> depth(Kinect::Height, Kinect::Width);	
	cv::Mat_<cv::Vec3b> color(Kinect::Height, Kinect::Width);	

	cv::FileStorage cvfs("input/depth.xml", CV_STORAGE_READ);
	cv::FileNode node(cvfs.fs, NULL);
	cv::read(node["averaged_depth"], averaged_depth);
	cv::read(node["depth"], depth);
	color = cv::imread("input/color.jpg", 1);
	//insert data
	for(int y=0; y<Kinect::Height; y++){
			for(int x=0; x<Kinect::Width; x++){
				bufferDepth_Host[y*Kinect::Width+x] = averaged_depth.at<float>(y, x);
				inputDepth_Host[y*Kinect::Width+x] = depth.at<float>(y, x);
			}
		}
	cudaMemcpy(inputDepth_Device, inputDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height, cudaMemcpyHostToDevice);
	cudaMemcpy(bufferDepth_Host, bufferDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height, cudaMemcpyHostToDevice);
	Color_Device.upload(color);
	///////////////////////////////////////////////////////////////////processing//////////////////////////////////////////////
	//JBF
	JBF.Process(inputDepth_Device, Color_Device);
	//MRF
	MRF.Process(inputDepth_Device, Color_Device);
	//project to real
	convertor.projectiveToReal(inputDepth_Device, points_Device);
	//region growing bilateral filter
	RGBF.Process(inputDepth_Device, points_Device, Color_Device);
	convertor.projectiveToReal(RGBF.getRefinedDepth_Device(), refinedPoints_Device);
	cudaMemcpy(refinedPoints_Host, refinedPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
	cudaMemcpy(points_Host, points_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);

	//////////////////////////////////////////////////////////////////////output//////////////////////////////////////////////////////////////////
	//visualize
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr input (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr averaged (new pcl::PointCloud<pcl::PointXYZRGB>);
	//convert to realworld
	for(int y=0; y<Kinect::Height; y++){
		for(int x=0; x<Kinect::Width; x++){
			pcl::PointXYZRGB point;
			point.r = color.at<cv::Vec3b>(y, x).val[0];
			point.g = color.at<cv::Vec3b>(y, x).val[1];
			point.b = color.at<cv::Vec3b>(y, x).val[2];
			//if(depth.at<float>(y, x) > 50.0f){
			//	XnPoint3D proj, real;
			//	proj.X = x;
			//	proj.Y = y;
			//	proj.Z = depth.at<float>(y, x);
			//	kinect.ProjectToReal(proj, real);
			//	point.x = (float)real.X/1000.0f;
			//	point.y = (float)real.Y/1000.0f;
			//	point.z = (float)real.Z/1000.0f;
			//	input->push_back(point);
			//}
			//if(averaged_depth.at<float>(y, x) > 50.0f){
			//	XnPoint3D proj, real;
			//	proj.X = x;
			//	proj.Y = y;
			//	proj.Z = averaged_depth.at<float>(y, x);
			//	kinect.ProjectToReal(proj, real);
			//	point.x = (float)real.X/1000.0f;
			//	point.y = (float)real.Y/1000.0f;
			//	point.z = (float)real.Z/1000.0f;
			//	averaged->push_back(point);
			//}
			if(points_Host[y*Kinect::Width+x].z > 50.0f){
				point.x = points_Host[y*Kinect::Width+x].x/1000.0f;
				point.y = points_Host[y*Kinect::Width+x].y/1000.0f;
				point.z = points_Host[y*Kinect::Width+x].z/1000.0f;	
				input->push_back(point);
			}
			if(refinedPoints_Host[y*Kinect::Width+x].z >50.0f){
				point.x = refinedPoints_Host[y*Kinect::Width+x].x/1000.0f;
				point.y = refinedPoints_Host[y*Kinect::Width+x].y/1000.0f;
				point.z = refinedPoints_Host[y*Kinect::Width+x].z/1000.0f;	
				averaged->push_back(point);
			}
			//if(points_Host[y*Kinect::Width+x].z > 50.0f && refinedPoints_Host[y*Kinect::Width+x].z >50.0f){
			//   std::cout << "input "<< points_Host[y*Kinect::Width+x].x<< ", "<<points_Host[y*Kinect::Width+x].y<<", "<<points_Host[y*Kinect::Width+x].z<<std::endl;
			//   std::cout << "averaged_depth "<< refinedPoints_Host[y*Kinect::Width+x].x<<", "<<refinedPoints_Host[y*Kinect::Width+x].y<<", "<<refinedPoints_Host[y*Kinect::Width+x].z<<std::endl;
			//}
		}
	}
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_input (new pcl::visualization::PCLVisualizer ("Input Viewer"));
	viewer_input->initCameraParameters ();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_input(input);
	viewer_input->addPointCloud<pcl::PointXYZRGB> (input, rgb_input, "input");
	viewer_input->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "input");
	
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_avraged(new pcl::visualization::PCLVisualizer ("Average Viewer"));
	viewer_avraged->initCameraParameters ();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_averaged(averaged);
	viewer_avraged->addPointCloud<pcl::PointXYZRGB> (averaged, rgb_averaged, "averaged");
	viewer_avraged->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "averaged");
	
	bool ShouldRun = true;
	
	while (ShouldRun) {  
		viewer_input->spinOnce (100);
		viewer_avraged->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}			
	
	
}