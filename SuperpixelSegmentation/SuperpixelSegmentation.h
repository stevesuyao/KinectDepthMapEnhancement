#ifndef SUPERPIXEL_SEGMENTATION_H
#define SUPERPIXEL_SEGMENTATION_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <XnCppWrapper.h>
#include <opencv2\opencv.hpp>
#include "thrust\fill.h"
#include "thrust\device_ptr.h"
#include "thrust\iterator\zip_iterator.h"
#include "thrust\transform.h"
#include <opencv2\gpu\gpu.hpp>


class SuperpixelSegmentation{
public:	
	
	typedef struct __align__(16) {		
		unsigned char r;		//1
		unsigned char g;		//1
		unsigned char b;		//1
		int x;		//4
		int y;      //4
		int size;  //4
	}superpixel;	

	typedef struct __align__(8){		//64 bits
		float d;		//4byte		distance + color difference from center point 
		int l;			//4byte		label of cluster
	}label_distance;	//force align 8 bytes


	SuperpixelSegmentation(int width, int height);	
	virtual ~SuperpixelSegmentation();
	void virtual SetParametor(int rows, int cols);
	void virtual Process(cv::gpu::GpuMat color_image, float color_sigma, float spatial_sigma, int iteration);
	static const int Line = 0, Average = 1;
	cv::Mat_<cv::Vec3b> getSegmentedImage(cv::Mat_<cv::Vec3b> input_host, int options);
	cv::Mat_<cv::Vec3b> getRandomColorImage();
	void releaseVideo();
	int*			getLabelDevice();
	superpixel*		getMeanDataDevice();
protected:
	void virtual				initMemory();
	int2 ClusterNum;
	int2 Window_Size;
	int width;
	int height;
	cv::Mat_<cv::Vec3b>			SegmentedRandomColor;
	cv::Mat_<cv::Vec3b>			SegmentedColor;
	cv::VideoWriter				SegmentedWriter;
	int*						Labels_Host;
	int*						Labels_Device;
	superpixel*					meanData_Device;
	superpixel*					meanData_Host;
	label_distance*				LD_Device;
	int3*						RandomColors;
};
#endif