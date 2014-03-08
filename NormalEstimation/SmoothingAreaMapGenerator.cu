//////////////////////////////////////////////////////
// contents :generate SmoothingAreaMap from depth map
// create 	:2013/03/17
// modefied :
// writer   :Takuya Ikeda 
// other	:GPU part
//////////////////////////////////////////////////////

#include "SmoothingAreaMapGenerator.h"
#include <algorithm>

__global__ void computeDCIMapGPU(const float3* vertice, int* dciMap, float param, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int acs = y*width+x;
	float depthC  = vertice[acs].z;
	float depthR = vertice[acs+1].z;
	float depthD = vertice[(y+1)*width+x].z;
	dciMap[acs] = 255;
	float depthDependentChange = ( param * (fabs(depthC)+1.0f) * 2.0f);

	if (abs(depthC-depthR) > depthDependentChange || depthC == 0.0 || depthR == 0.0) {
		dciMap[acs] = 0;
		dciMap[acs+1] = 0;
	}
	if (abs (depthC - depthD) > depthDependentChange || depthC == 0.0 || depthD == 0.0){
		dciMap[acs] = 0;
		dciMap[acs+width] = 0;
	}
}


void SmoothingAreaMapGenerator::computeDCIMap(){
	dim3 block(BLOCKDIM, BLOCKDIM), grid(width/BLOCKDIM, height/BLOCKDIM);
	computeDCIMapGPU<<<grid, block>>>(verticeMap, depthChangeIndicationMap, max_depth_change_factor_, width, height);
}


void SmoothingAreaMapGenerator::computeDTMap(void){
	int* tmpdciMap = new int[width*height];
	float* tmpdtMap = new float[width*height];
	cudaMemcpy(tmpdciMap, depthChangeIndicationMap, sizeof(int)*width*height, cudaMemcpyDeviceToHost);

	//compute distanceMap
	for (int index = 0; index < width*height; index++){
		if (tmpdciMap[index] == 0)
			tmpdtMap[index] = 0.0f;
		else
			tmpdtMap[index] = width + height;
	}

	// first pass
	float* previous_row = tmpdtMap;
	float* current_row = previous_row + width;
	for (int ri = 1; ri < height; ++ri){
		for (int ci = 1; ci < width; ++ci){
			const float upLeft  = previous_row [ci - 1] + 1.4f; //distanceMap[(ri-1)*input_->width + ci-1] + 1.4f;
			const float up      = previous_row [ci] + 1.0f;     //distanceMap[(ri-1)*input_->width + ci] + 1.0f;
			const float upRight = previous_row [ci + 1] + 1.4f; //distanceMap[(ri-1)*input_->width + ci+1] + 1.4f;
			const float left    = current_row  [ci - 1] + 1.0f; //distanceMap[ri*input_->width + ci-1] + 1.0f;
			const float center  = current_row  [ci];            //distanceMap[ri*input_->width + ci];

			const float minValue = std::min(std::min (upLeft, up), std::min (left, upRight));

			if (minValue < center)
				current_row [ci] = minValue; //distanceMap[ri * input_->width + ci] = minValue;
		}
		previous_row = current_row;
		current_row += width;
	}

	float* next_row    = tmpdtMap + width * (height - 1);
	current_row = next_row - width;
	// second pass
	for (int ri = height-2; ri >= 0; --ri){
		for (int ci = width-2; ci >= 0; --ci){
			const float lowerLeft  = next_row [ci - 1] + 1.4f;    //distanceMap[(ri+1)*input_->width + ci-1] + 1.4f;
			const float lower      = next_row [ci] + 1.0f;        //distanceMap[(ri+1)*input_->width + ci] + 1.0f;
			const float lowerRight = next_row [ci + 1] + 1.4f;    //distanceMap[(ri+1)*input_->width + ci+1] + 1.4f;
			const float right      = current_row [ci + 1] + 1.0f; //distanceMap[ri*input_->width + ci+1] + 1.0f;
			const float center     = current_row [ci];            //distanceMap[ri*input_->width + ci];

			const float minValue = std::min (std::min (lowerLeft, lower), std::min (right, lowerRight));
			if (minValue < center)
				current_row [ci] = minValue; //distanceMap[ri*input_->width + ci] = minValue;
		}
	}
	cudaMemcpy(distanceTransformMap, tmpdtMap, sizeof(float)*width*height, cudaMemcpyHostToDevice);
	delete tmpdciMap; 
	delete tmpdtMap;
}


__global__ void computeDDSAIMapGPU(float3 *vertice, float* ddsaMap, float param, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int acs = y*width+x;
	//ddsaMap[acs] = param * 2.0f;
	ddsaMap[acs] = param + static_cast<float>(vertice[acs].z)/10.0f;

}

void SmoothingAreaMapGenerator::computeDDSAIMap(void){
	dim3 block(BLOCKDIM, BLOCKDIM), grid(width/BLOCKDIM, height/BLOCKDIM);
	computeDDSAIMapGPU<<<grid, block>>>(verticeMap, depthDependentSmoothingAreaMap, normal_smoothing_size_, width, height);
}

__global__ void computeDDSAIMapGPU(float* dtMap, float* ddsaMap, float* fsMap, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int acs = y*width+x;
	if(dtMap[acs] < ddsaMap[acs])
		fsMap[acs] = dtMap[acs];
	else
		fsMap[acs] = ddsaMap[acs];
}

void SmoothingAreaMapGenerator::computeFSIMap(void){
	dim3 block(BLOCKDIM, BLOCKDIM), grid(width/BLOCKDIM, height/BLOCKDIM);
	computeDDSAIMapGPU<<<grid, block>>>(distanceTransformMap, depthDependentSmoothingAreaMap, finalSmoothingMap, width, height);
}
