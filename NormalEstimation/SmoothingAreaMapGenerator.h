//////////////////////////////////////////////////////
// contents :generate SmoothingAreaMap from depth map
// create 	:2013/03/17
// modefied :
// writer   :Takuya Ikeda 
// other	:
//////////////////////////////////////////////////////

#ifndef _SMOOTHINGAREAMAPGENERATOR_H_
#define _SMOOTHINGAREAMAPGENERATOR_H_


#include <cutil_inline.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#define BLOCKDIM 16


class SmoothingAreaMapGenerator{
public:
	//parameter
	float normal_smoothing_size_;
	float max_depth_change_factor_;


	SmoothingAreaMapGenerator(int w, int h);
	~SmoothingAreaMapGenerator();

	void setVerticeMap(float3* in);
	void generateFinalSmoothingAreaMap(void);
	float* getFinalSmoothingMap(void);
private:
	//image size
	int width;
	int height;

	//Map
	int* depthChangeIndicationMap;
	float* distanceTransformMap;
	float* depthDependentSmoothingAreaMap;
	float* finalSmoothingMap;
	float3* verticeMap;


	void computeDCIMap();
	void computeDTMap();
	void computeDDSAIMap();
	void computeFSIMap();

	void initMemory(void);
};
#endif 