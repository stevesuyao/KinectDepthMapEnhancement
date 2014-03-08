/////////////////////////////////////////////////////////////////////////////////
// contents :NormalMap generation from SmoothingAreaMap, IntegralImage, vertexMap
// create 	:2013/07/02
// modefied :  
// writer   :Takuya Ikeda 
// other	:
/////////////////////////////////////////////////////////////////////////////////

#ifndef _NORMALMAPgenerator_H_
#define _NORMALMAPgenerator_H_


#include <cutil_inline.h>
#include <cuda.h>
//#include "../header/define.h"
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "IntegralImageGenerator.h"
#include "SmoothingAreaMapGenerator.h"
//#include "Dimensionconvertor.h"

#define BLOCKDIM 16

class NormalMapGenerator{
public:
	NormalMapGenerator(int w, int h);
	~NormalMapGenerator();
	static const int SDC = 0, CM = 1, BILATERAL = 2;

	//void setInput(float3* in);
	//void setCameraParams(unsigned long long F, float p_size);
	void setNormalEstimationMethods(int method);

	void generateNormalMap(float3* vertices_device);

	//IO
	float3* getNormalMap(void);

	cv::Mat getNormalImg(void);
	cv::Mat getSegmentNormalImg(bool*mask);
	void saveNormalImg(char* str);
	void saveSegmentNormalImg(char* str, bool*mask);

private:
	int width;
	int height;

	int normal_estimation_method_;

	float3* verticeMap;
	float3* normalMap;
	float3* segmentNormalMap;

	IntegralImagegenerator iig;
	SmoothingAreaMapGenerator samg;
	//Dimensionconvertor dc;

	void initMemory();
	void computeNormal(float3* vertices_device);
};
#endif 