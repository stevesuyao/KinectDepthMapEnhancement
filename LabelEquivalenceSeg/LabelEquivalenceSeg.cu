#include "LabelEquivalenceSeg.h"
#include <cuda_runtime.h>
#include <thrust\fill.h>
#include <thrust\device_ptr.h>
#include <iostream>


__global__ void initLabel(float4* input_nd, int* merged_cluster_label, int* ref, 
	float3*  cluster_normals, int* cluster_label, float3* cluster_centers, int width, int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		
		if(cluster_normals[cluster_label[x+y*width]].x != -1.0f || 
				cluster_normals[cluster_label[x+y*width]].y !=  -1.0f ||
					cluster_normals[cluster_label[x+y*width]].z != -1.0f){
			input_nd[x+y*width].x = cluster_normals[cluster_label[x+y*width]].x;
			input_nd[x+y*width].y = cluster_normals[cluster_label[x+y*width]].y;
			input_nd[x+y*width].z = cluster_normals[cluster_label[x+y*width]].z;
			input_nd[x+y*width].w = fabs(cluster_normals[cluster_label[x+y*width]].x*cluster_centers[cluster_label[x+y*width]].x + 
											cluster_normals[cluster_label[x+y*width]].y*cluster_centers[cluster_label[x+y*width]].y +
												cluster_normals[cluster_label[x+y*width]].z*cluster_centers[cluster_label[x+y*width]].z);
			ref[x+y*width] = x+y*width;
			merged_cluster_label[x+y*width] = cluster_label[x+y*width];
			
		}
		else {
			input_nd[x+y*width].x = 5.0;
			input_nd[x+y*width].y = 5.0;
			input_nd[x+y*width].z = 5.0;
			input_nd[x+y*width].w = 5.0;
			ref[x+y*width] = x+y*width;
			merged_cluster_label[x+y*width] = -1;
		}
}

__device__ bool compNormal(float4* a, float4* b){
	return 
		acos(a->x * b->x + a->y * b->y + a->z * b->z)>0 && 
		acos(a->x * b->x + a->y * b->y + a->z * b->z) < (3.141592653f / 8.0f)
		&&
		abs(a->w - b->w) < 150.0f;
}


__device__ int getMin(
	float4& up_nd,
	int& merged_up_label,
	int up_label,
	float4& left_nd,
	int& merged_left_label,
	int left_label,
	float4& center_nd,
	int& merged_center_label,
	int center_label,
	float4& right_nd,
	int& merged_right_label,
	int right_label,
	float4& down_nd,
	int& merged_down_label,
	int down_label){
		int c;
		c = merged_up_label > -1 && (up_label==center_label || compNormal(&up_nd, &center_nd)) && merged_up_label < merged_center_label ? merged_up_label : merged_center_label;
		c = merged_left_label > -1 && (left_label==center_label || compNormal(&left_nd, &center_nd)) && merged_left_label < c ? merged_left_label : c;
		c = merged_right_label > -1 && (right_label==center_label || compNormal(&right_nd, &center_nd)) &&merged_right_label < c ? merged_right_label : c;
		return merged_down_label > -1 && (down_label==center_label || compNormal(&down_nd, &center_nd)) && merged_down_label < c ? merged_down_label : c;
}


__global__ void scanKernel(
	float4* input_nd, 
	int* merged_cluster_label, 
	int* ref, 
	int* cluster_label, 
	float* variance,
	int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	int label1 = merged_cluster_label[x + y * width];
	int label2;

	if(label1 > -1/* && acos(variance[label1]) < (3.141592653f / 6.0f)*/){
		label2 = getMin(
			//up
			input_nd[x + (y - 1 > 0 ? y - 1 : 0) * width],
			merged_cluster_label[x + (y - 1 > 0 ? y - 1 : 0) * width],
			cluster_label[x + (y - 1 > 0 ? y - 1 : 0) * width], 
			//left
			input_nd[(x - 1 > 0 ? x - 1 : 0) + y * width],
			merged_cluster_label[(x - 1 > 0 ? x - 1 : 0) + y * width],
			cluster_label[(x - 1 > 0 ? x - 1 : 0) + y * width],
			//center
			input_nd[x + y * width],
			merged_cluster_label[x + y * width],
			cluster_label[x + y*width],
			//right
			input_nd[(x + 1 < width ? x + 1 : width) + y * width],
			merged_cluster_label[(x + 1 < width ? x + 1 : width) + y * width],
			cluster_label[(x + 1 < width ? x + 1 : width) + y * width],
			//down
			input_nd[x + (y + 1 < height ? y + 1 : height) * width],
			merged_cluster_label[x + (y + 1 < height ? y + 1 : height) * width],
			cluster_label[x + (y + 1 < height ? y + 1 : height) * width]);
	
		if(label2 < label1){
			atomicMin(&ref[label1], label2);
		}	
	}
}
__global__ void analysisKernel(
	int* merged_cluster_label, 
	int* ref, 
	int* cluster_label,
	/*float3* normals,
	float3* centers,*/
	int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);

	if(merged_cluster_label[x+y*width] ==  cluster_label[x+y*width]){
	//if(merged_cluster_label[x+y*width] ==  x+y*width){
		//label...¡‚ÌêŠ  //current...‚»‚ÌêŠ‚Ìlabel
		int current = ref[x+y*width];
		//‚»‚ÌêŠ‚Ìlabel‚ª‚Â‚¢‚½—Ìˆæ‚ÅêŠ‚Ì’l(x+y*width)‚Ælabel‚ªˆê’v‚·‚éêŠ‚ð’Tõ
		do{
			current = ref[current];
		} while(current != ref[current]);
		//’Tõ‚µ‚½label(‚Â‚Ü‚è‚»‚Ìlabel”Ô†‚ª‚Â‚¢‚½êŠ‚Ì’l)
		ref[x+y*width] = current;		
	}
	//Labeling phase
	if(merged_cluster_label[x+y*width]> -1){
		merged_cluster_label[x+y*width] = ref[merged_cluster_label[x+y*width]];

	}
}

__device__ inline void fatomicAdd(float* address, float val)
{
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_int, 
						assumed, 
						__float_as_int(val + __int_as_float(assumed)));
	
	}while(assumed != old);
}
__device__ inline void atomicFloatAdd(float *address, float val)
{
    int i_val = __float_as_int(val);
    int tmp0 = 0;
    int tmp1;
 
    while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
    {
        tmp0 = tmp1;
        i_val = __float_as_int(val + __int_as_float(tmp1));
    }
}
__global__ void countKernel(int* merged_cluster_label,
							float4* input_nd,
							int* cluster_label,
							float3* cluster_center,
							float3* sum_of_merged_cluster_normals, 
							float3* sum_of_merged_cluster_centers,
							int* merged_cluster_size, int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);

	if(merged_cluster_label[x + y * width]> -1 && (input_nd[x + y * width].x != -1.0f ||
														input_nd[x + y * width].y != -1.0f ||
															input_nd[x + y * width].y != -1.0f )){

		////count cluster size
		atomicAdd(&merged_cluster_size[merged_cluster_label[x + y * width]], 1);
		////add cluster normal
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x, input_nd[x + y * width].x);
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y, input_nd[x + y * width].y);
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z, input_nd[x + y * width].z);
		//add cluster center
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x, cluster_center[cluster_label[x + y * width]].x);
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y, cluster_center[cluster_label[x + y * width]].y);
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z, cluster_center[cluster_label[x + y * width]].z);
	}
	else{
		merged_cluster_label[x + y * width] = -1;
	}
	
}

__global__ void calculate_nd(int* merged_cluster_label,
							int* ref,
							float4* merged_cluster_nd,
							float3* sum_of_merged_cluster_normals, 
							float3* sum_of_merged_cluster_centers,
							int* merged_cluster_size, 
							float* merged_cluster_variance, 
							float4* input_nd,
							int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if(merged_cluster_label[x + y * width] > -1 ){
		
		//calculate normal
		merged_cluster_nd[x + y * width].x = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		merged_cluster_nd[x + y * width].y = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		merged_cluster_nd[x + y * width].z = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//calculate center
		float3 center;
		center.x = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		center.y = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		center.z = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//calculate variance
		float variance = input_nd[x + y * width].x*merged_cluster_nd[x + y * width].x +
								input_nd[x + y * width].y*merged_cluster_nd[x + y * width].y +
									input_nd[x + y * width].z*merged_cluster_nd[x + y * width].z;
		variance /= (float)merged_cluster_size[merged_cluster_label[x + y * width]];
		atomicAdd(&merged_cluster_variance[merged_cluster_label[x + y * width]], variance);
		//calculate distance
		merged_cluster_nd[x + y * width].w = fabs(merged_cluster_nd[x + y * width].x*center.x + 
														merged_cluster_nd[x + y * width].y*center.y +
															merged_cluster_nd[x + y * width].z*center.z);
	}
}

void LabelEquivalenceSeg::labelImage(float3* cluster_normals_device, int* cluster_label_device, float3* cluster_centers_device, float* variance_device){
	

	//initialize parameter
	initLabel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(InputND_Device, MergedClusterLabel_Device, ref, cluster_normals_device, cluster_label_device, cluster_centers_device, width, height);
	
	for(int i = 0; i < 10; i++){
		//scan(cluster_label_device);
		scanKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(InputND_Device, MergedClusterLabel_Device, ref, cluster_label_device, variance_device, width, height);
		//analysis(cluster_label_device);
		analysisKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(MergedClusterLabel_Device, ref, cluster_label_device, width, height);	
	}
	//init merged_cluster_size
	int int_zero = 0;
	thrust::fill(
		thrust::device_ptr<int>(merged_cluster_size),
		thrust::device_ptr<int>(merged_cluster_size) + width * height,
		int_zero);
	//init normap map, center map
	float3 float3_zero;
	float3_zero.x = 0.0f;
	float3_zero.y = 0.0f;
	float3_zero.z = 0.0f;
	thrust::fill(
		thrust::device_ptr<float3>(sum_of_merged_cluster_normals),
		thrust::device_ptr<float3>(sum_of_merged_cluster_normals) + width * height,
		float3_zero);
	thrust::fill(
		thrust::device_ptr<float3>(sum_of_merged_cluster_centers),
		thrust::device_ptr<float3>(sum_of_merged_cluster_centers) + width * height,
		float3_zero);
	//init variance map
	float float_zero = 0.0f;
	thrust::fill(
		thrust::device_ptr<float>(MergedClusterVariance_Device),
		thrust::device_ptr<float>(MergedClusterVariance_Device) + width * height,
		float_zero);
	//calculate each cluster parametor
	//count cluster size
	countKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(MergedClusterLabel_Device, InputND_Device, cluster_label_device, cluster_centers_device, sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, merged_cluster_size, width, height);
	//calculate normal map, plane distance and variance map
	calculate_nd<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(MergedClusterLabel_Device, ref, MergedClusterND_Device, sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, merged_cluster_size, MergedClusterVariance_Device, InputND_Device, width, height);
	

	//memcpy
	cudaMemcpy(MergedClusterLabel_Host, MergedClusterLabel_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(MergedClusterND_Host, MergedClusterND_Device, sizeof(float4)*width*height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(ref_host, ref, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	
}
