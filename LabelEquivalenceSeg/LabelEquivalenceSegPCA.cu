#include "LabelEquivalenceSegPCA.h"
#include <cuda_runtime.h>
#include <thrust\fill.h>
#include <thrust\device_ptr.h>
#include <iostream>

__global__ void initLabelPCA(float4* input_nd, int* merged_cluster_label, int* ref, 
	float4*  cluster_nd, int* cluster_label, int width, int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		
		if(abs(cluster_nd[cluster_label[x+y*width]].x) < 1.1 && cluster_label[x+y*width] > -1){
			//merged_cluster_label[x+y*width] = x+y*width;
			merged_cluster_label[x+y*width] = cluster_label[x+y*width];
			input_nd[x+y*width] = cluster_nd[cluster_label[x+y*width]];
			ref[x+y*width] = x+y*width;
		} else {
			input_nd[x+y*width].x = 5.0;
			input_nd[x+y*width].y = 5.0;
			input_nd[x+y*width].z = 5.0;
			input_nd[x+y*width].w = 5.0;
			merged_cluster_label[x+y*width] = -1;
			ref[x+y*width] = y*width+x;
		}
}

__device__ bool compNormalPCA(float4* a, float4* b){
	return 
		abs(a->x) < 1.1 && abs(b->x) < 1.1 && 
		abs(acos(a->x * b->x + a->y * b->y 
		+ a->z * b->z)) < (3.141592653f / 8.0f)
		&&
		abs(a->w - b->w) < 700.0f;
}


__device__ int getMinPCA(
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
		c = merged_up_label > -1 && (up_label==center_label || compNormalPCA(&up_nd, &center_nd)) && merged_up_label < merged_center_label ? merged_up_label : merged_center_label;
		c = merged_left_label > -1 && (left_label==center_label || compNormalPCA(&left_nd, &center_nd)) && merged_left_label < c ? merged_left_label : c;
		c = merged_right_label > -1 && (right_label==center_label || compNormalPCA(&right_nd, &center_nd)) &&merged_right_label < c ? merged_right_label : c;
		return merged_down_label > -1 && (down_label==center_label || compNormalPCA(&down_nd, &center_nd)) && merged_down_label < c ? merged_down_label : c;
}


__global__ void scanKernelPCA(float4* input_nd, int* merged_cluster_label, int* ref, int* cluster_label, int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	int label1 = merged_cluster_label[x + y * width];
	int label2;

	if(label1 > -1){
		label2 = getMinPCA(
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

__global__ void analysisKernelPCA(float4* input_nd, int* merged_cluster_label, int* ref, int* cluster_label, int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);

	//if(merged_cluster_label[x+y*width] ==  x+y*width){
	if(merged_cluster_label[x+y*width] ==  cluster_label[x+y*width]){
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
	if(merged_cluster_label[x+y*width]> -1)
		//if(cluster_label[x+y*width] != cluster_label[ref[merged_cluster_label[x+y*width]]])
		merged_cluster_label[x+y*width] = ref[merged_cluster_label[x+y*width]];


}

__device__ inline void fatomicAddPCA(float* address, float val)
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
__device__ inline void atomicFloatAddPCA(float *address, float val)
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
__global__ void countKernelPCA(int* merged_cluster_label,
							float4* input_nd,
							int* cluster_label,
							float3* cluster_center,
							float* cluster_eigenvalues,
							float3* sum_of_merged_cluster_normals, 
							float3* sum_of_merged_cluster_centers,
							float* sum_of_merged_cluster_eigenvalues,
							int* merged_cluster_size, int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);

	if(merged_cluster_label[x + y * width]> -1 && fabs(input_nd[x + y * width].x) < 1.1 ){

		////count cluster size
		atomicAdd(&merged_cluster_size[merged_cluster_label[x + y * width]], 1);
		//add cluster normal
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x, input_nd[x + y * width].x);
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y, input_nd[x + y * width].y);
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z, input_nd[x + y * width].z);
		//add cluster center
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x, cluster_center[cluster_label[x + y * width]].x);
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y, cluster_center[cluster_label[x + y * width]].y);
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z, cluster_center[cluster_label[x + y * width]].z);
		//add cluster eigenvalue
		atomicAdd(&sum_of_merged_cluster_eigenvalues[merged_cluster_label[x + y * width]], cluster_eigenvalues[cluster_label[x + y * width]]);
	}
	else{
		merged_cluster_label[x + y * width] = -1;
	}
	
}

__global__ void calculate_ndPCA(int* merged_cluster_label,
							int* ref,
							float4* merged_cluster_nd,
							float* merged_cluster_eigenvalues,
							float3* sum_of_merged_cluster_normals, 
							float3* sum_of_merged_cluster_centers,
							float* sum_of_merged_cluster_eigenvalues,
							int* merged_cluster_size, int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if(merged_cluster_label[x + y * width] > -1){
		//calculate normal
		merged_cluster_nd[x + y * width].x = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		merged_cluster_nd[x + y * width].y = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		merged_cluster_nd[x + y * width].z = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//calculate center
		float3 center;
		center.x = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		center.y = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		center.z = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//calculate eigenvalues
		merged_cluster_eigenvalues[x + y * width] = sum_of_merged_cluster_eigenvalues[merged_cluster_label[x + y * width]]/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//calculate distance
		merged_cluster_nd[x + y * width].w = fabs(merged_cluster_nd[x + y * width].x*center.x + 
																			merged_cluster_nd[x + y * width].y*center.y +
																				merged_cluster_nd[x + y * width].z*center.z);
	}
}

void LabelEquivalenceSegPCA::postProcess(int* cluster_label_device, float3* cluster_center_device, float* eigenvalues_device){
	
	//count cluster size
	countKernelPCA<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(MergedClusterLabel_Device, InputND_Device, cluster_label_device, cluster_center_device, eigenvalues_device, 
					sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, sum_of_merged_cluster_eigenvalues, merged_cluster_size, width, height);

	//calculate normal map
	calculate_ndPCA<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(MergedClusterLabel_Device, ref, MergedClusterND_Device, MergedClusterEigenvalues_Device, 
				sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, sum_of_merged_cluster_eigenvalues, merged_cluster_size, width, height);
}

void LabelEquivalenceSegPCA::labelImage(float4* cluster_nd_device, int* cluster_label_device, float3* cluster_center_device, float* eigenvalues_device){
	

	//initialize parameter
	initLabelPCA<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(InputND_Device, MergedClusterLabel_Device, ref, cluster_nd_device, cluster_label_device, width, height);
	
	for(int i = 0; i < 10; i++){
		//scan(cluster_label_device);
		scanKernelPCA<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(InputND_Device, MergedClusterLabel_Device, ref, cluster_label_device, width, height);
		//analysis(cluster_label_device);
		analysisKernelPCA<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(InputND_Device, MergedClusterLabel_Device, ref, cluster_label_device, width, height);
			
	}
	//init cluster_nd
	float4 reset;
	reset.x = 0.0;
	reset.y = 0.0;
	reset.z = 0.0;
	reset.w = 0.0;

	thrust::fill(
		thrust::device_ptr<float4>(MergedClusterND_Device),
		thrust::device_ptr<float4>(MergedClusterND_Device) + width * height,
		reset);

	//init merged_cluster_size
	int int_zero = 0;
	thrust::fill(
		thrust::device_ptr<int>(merged_cluster_size),
		thrust::device_ptr<int>(merged_cluster_size) + width * height,
		int_zero);
	//init normap map and center map
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
	//calculate each cluster parametor
	postProcess(cluster_label_device, cluster_center_device, eigenvalues_device);
	//memcpy
	cudaMemcpy(MergedClusterLabel_Host, MergedClusterLabel_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	cudaMemcpy(MergedClusterND_Host, MergedClusterND_Device, sizeof(float4)*width*height, cudaMemcpyDeviceToHost);
	cudaMemcpy(ref_host, ref, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	
	cv::Mat_<cv::Vec3b> normalImage(height, width);
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			//std::cout <<"x:"<<x<<", y:"<<y<<std::endl;
			int id = MergedClusterLabel_Host[y*width+x];
			if(id==-1)
				normalImage.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 0, 0);
			else{
				normalImage.at<cv::Vec3b>(y,x).val[0] = (unsigned char)(255.0f*(MergedClusterND_Host[y*width+x].x+1.0f)/2.0f);
				normalImage.at<cv::Vec3b>(y,x).val[1] = (unsigned char)(255.0f*(MergedClusterND_Host[y*width+x].y+1.0f)/2.0f);
				normalImage.at<cv::Vec3b>(y,x).val[2] = (unsigned char)(255.0f*(MergedClusterND_Host[y*width+x].z+1.0f)/2.0f);
			}
		}
	}
	cv::imshow("label_normal", normalImage);
	
	//for(int i = 0; i < height; i++){
    //            for(int j = 0; j < width; j++){
	//			if(MergedClusterLabel_Host[i*width+j] > -1 && i*width+j == ref_host[MergedClusterLabel_Host[i*width+j]]){
	//				std::cout << "label: "<<MergedClusterLabel_Host[i*width+j]<<std::endl;
	//				std::cout << "nd: "<<MergedClusterND_Host[MergedClusterLabel_Host[i*width+j]].x<< ", "<<MergedClusterND_Host[i*width+j].y<<", "<<MergedClusterND_Host[i*width+j].z<<", "<<MergedClusterND_Host[i*width+j].w<<std::endl;
	//				}
	//			}
	//}


}

