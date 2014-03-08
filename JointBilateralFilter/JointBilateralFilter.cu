#include "JointBilateralFilter.h"

//computeBilateralFiltering
__global__ void joint_bilateral_filtering(
	int width, int height,
	float* depth_device,
	cv::gpu::GpuMat color_image,
	float* spatial_filter_device,
	float* filtered_device,
	int window_size,
	float color_sigma,
	float depth_sigma){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		//calculate weighted average
		float w_average = 0.0f;
		float weight = 0.0f;
		for(int i = - window_size/2; i <= window_size/2; i++){		// y
			for(int j = -window_size/2; j <= window_size/2; j++){		// x
				int xj = x+j, yi = y+i;
				if(xj >= 0 && xj < width && yi >= 0 && yi < height && depth_device[yi*width+xj] > 50.0f){
					float color_diff = powf((float)(color_image.data[(y*width+x)*3+0])-(float)(color_image.data[(yi*width+xj)*3+0]), 2) +
										powf((float)(color_image.data[(y*width+x)*3+1])-(float)(color_image.data[(yi*width+xj)*3+1]), 2) +
											powf((float)(color_image.data[(y*width+x)*3+2])-(float)(color_image.data[(yi*width+xj)*3+2]), 2); 
					float color_filter = 0.0f;
					if(color_sigma != 0.0f)
					color_filter = expf(-color_diff/(2*powf(color_sigma, 2.0f)));
					//calculate filter
					float filter = 1.0f;
					if(spatial_filter_device[(i+window_size/2)*window_size+(j+window_size/2)] != 0.0f)
						filter *= spatial_filter_device[(i+window_size/2)*window_size+(j+window_size/2)];
					if(color_filter != 0.0f)
						filter *= color_filter;
					w_average += depth_device[yi*width+xj]*filter;
					weight += filter;
				}
			}
		}
		if(weight > 0.0f){
			w_average /= weight;
		
		//filtering
		float numerator = 0.0f, denominator = 0.0f;
		//if(depth_device[y*width+x] > 50.0f){
		for(int i = - window_size/2; i <= window_size/2; i++){		// y
			for(int j = -window_size/2; j <= window_size/2; j++){		// x
				int xj = x+j, yi = y+i;
				if(xj >= 0 && xj < width && yi >= 0 && yi < height && depth_device[yi*width+xj] > 50.0f){
					//color filter
					float color_diff = powf((float)(color_image.data[(y*width+x)*3+0])-(float)(color_image.data[(yi*width+xj)*3+0]), 2) +
										powf((float)(color_image.data[(y*width+x)*3+1])-(float)(color_image.data[(yi*width+xj)*3+1]), 2) +
											powf((float)(color_image.data[(y*width+x)*3+2])-(float)(color_image.data[(yi*width+xj)*3+2]), 2); 
					float color_filter = 0.0f;
					if(color_sigma != 0.0f)
					color_filter = expf(-color_diff/(2*powf(color_sigma, 2.0f)));
					//depth filter
					float depth_diff = powf(depth_device[yi*width+xj] - w_average, 2.0f);
					float depth_filter;
					if(depth_sigma!= 0.0f)
					depth_filter = expf(-depth_diff/(2.0f*powf(depth_sigma, 2.0f)));
					//calculate filter
					float filter = 1.0f;
					if(spatial_filter_device[(i+window_size/2)*window_size+(j+window_size/2)] != 0.0f)
						filter *= spatial_filter_device[(i+window_size/2)*window_size+(j+window_size/2)];
					if(color_filter != 0.0f)
						filter *= color_filter;
					if(depth_filter != 0.0f)
						filter *= depth_filter;
					numerator += depth_device[yi*width+xj]*filter; 
					denominator += filter;
				}
			}
		}
		//}
		if(denominator == 0.0f)
			filtered_device[y*width+x] = 0.0f;
		else
			filtered_device[y*width+x] = numerator/denominator;
		}
		else{
			filtered_device[y*width+x] = 0.0f;
		}
}

//template<int blockSize>
//__global__ void joint_bilateral_filtering(
//	int width, int height,
//	float* depth_device,
//	cv::gpu::GpuMat color_image,
//	float* spatial_filter_device,
//	float* filtered_device,
//	int window_size,
//	float color_sigma,
//	float depth_sigma){
//		__shared__ float w_average[blockSize];
//		__shared__ float weight[blockSize];
//
//		int x = blockIdx.x;
//		int y = blockIdx.y;
//		//thread id
//		int tid = threadIdx.y*blockDim.x+threadIdx.x;	
//		w_average[tid] = 0.0f;
//		weight[tid] = 0.0f;
//		//around pixel
//		int2 around;
//		around.x = x-blockDim.x/2+threadIdx.x;
//		around.y = y-blockDim.y/2+threadIdx.y;
//		//calculate weighted average
//		if(around.x>=0 && around.x<width && around.y>=0 && around.y<height){
//			if(abs(around.x-x)<=(window_size/2) && abs(around.y-y)<=(window_size/2)   
//								&& depth_device[around.y*width+around.x] > 50.0f){
//					w_average[tid] = depth_device[around.y*width+around.x]*
//								spatial_filter_device[((around.y-y)+window_size/2)*window_size+((around.x-x)+window_size/2)];
//					weight[tid] = spatial_filter_device[((around.y-y)+window_size/2)*window_size+((around.x-x)+window_size/2)];
//					//w_average[tid] = depth_device[around.y*width+around.x];
//					//weight[tid] = 1.0f;
//				}
//			}
//		__syncthreads();
//
//		//assign cluster label
//		if(blockSize >= 1024){
//			if(tid < 512){
//				w_average[tid] += w_average[tid+512];
//				weight[tid] += weight[tid+512];
//			}
//			__syncthreads();
//		}
//		if(blockSize >= 512){
//			if(tid < 256){
//				w_average[tid] += w_average[tid+256];
//				weight[tid] += weight[tid+256];
//			}
//				__syncthreads();			
//		}
//		if(blockSize >= 256){
//			if(tid < 128){
//				w_average[tid] += w_average[tid+128];
//				weight[tid] += weight[tid+128];
//			}
//				__syncthreads();
//		}
//		if(blockSize >= 128){
//			if(tid < 64){
//				w_average[tid] += w_average[tid+64];
//				weight[tid] += weight[tid+64];
//			}
//				__syncthreads();
//		}
//		if(tid < 32){
//			if(blockSize >= 64){
//				w_average[tid] += w_average[tid+32];
//				weight[tid] += weight[tid+32];
//			}
//			if(blockSize >= 32){
//				w_average[tid] += w_average[tid+16];
//				weight[tid] += weight[tid+16];
//			}
//			if(blockSize >= 16){
//				w_average[tid] += w_average[tid+8];
//				weight[tid] += weight[tid+8];
//			}
//			if(blockSize >= 8){
//				w_average[tid] += w_average[tid+4];
//				weight[tid] += weight[tid+4];
//			}
//			if(blockSize >= 4){
//				w_average[tid] += w_average[tid+2];
//				weight[tid] += weight[tid+2];
//			}
//			if(blockSize >= 2){
//				w_average[tid] += w_average[tid+1];
//				weight[tid] += weight[tid+1];
//			}
//		}
//		//store center point
//		if(tid == 0){
//				if(weight[tid] != 0.0f){
//					 w_average[tid] /= weight[tid];
//					filtered_device[y*width+x] = w_average[tid];
//				}else{
//					filtered_device[y*width+x] = 0.0f;
//				}
//				//filtered_device[y*width+x] = w_average[tid];
//		}
//		__syncthreads();
//
//
//
//	//	if(weight[0] != 0.0f){
//	//	//filtering
//	//	__shared__ float numerator[blockSize];
//	//	__shared__ float denominator[blockSize];
//	//	numerator[tid] = 0.0f;
//	//	denominator[tid] = 0.0f;
//	//	if(around.x>=0 && around.x<width && around.y>=0 && around.y<height){
//	//		if(abs(around.x-x)<=(window_size/2) && abs(around.y-y)<=(window_size/2)  
//	//							&& depth_device[around.y*width+around.x] > 50.0f){
//	//				float color_diff = powf((float)(color_image.data[(y*width+x)*3+0])-(float)(color_image.data[(around.y*width+around.x)*3+0]), 2) +
//	//									powf((float)(color_image.data[(y*width+x)*3+1])-(float)(color_image.data[(around.y*width+around.x)*3+1]), 2) +
//	//										powf((float)(color_image.data[(y*width+x)*3+2])-(float)(color_image.data[(around.y*width+around.x)*3+2]), 2); 
//	//				float color_filter = expf(-color_diff/(2*powf(color_sigma, 2.0f)));
//	//				//depth filter
//	//				float depth_filter = 1.0f;
//	//				float depth_diff = powf(depth_device[around.y*width+around.x] - w_average[0], 2.0f);
//	//				depth_filter = expf(-depth_diff/(2.0f*powf(depth_sigma, 2.0f)));
//	//				//calculate filter
//	//				//int i = threadIdx.y-blockDim.y/2;
//	//				//int j = threadIdx.x-blockDim.x/2;
//	//				numerator[tid] += depth_device[around.y*width+around.x] * spatial_filter_device[((around.y-y)+window_size/2)*window_size+((around.x-x)+window_size/2)] * color_filter * depth_filter;
//	//				denominator[tid] += spatial_filter_device[((around.y-y)+window_size/2)*window_size+((around.x-x)+window_size/2)] * color_filter * depth_filter;
//	//			}
//	//		}
//	//	__syncthreads();
//	//	if(blockSize >= 1024){
//	//		if(tid < 512){
//	//			numerator[tid] += numerator[tid+512];
//	//			denominator[tid] += denominator[tid+512];
//	//		}
//	//		__syncthreads();
//	//	}
//	//	if(blockSize >= 512){
//	//		if(tid < 256){
//	//			numerator[tid] += numerator[tid+256];
//	//			denominator[tid] += denominator[tid+256];
//	//		}
//	//			__syncthreads();
//	//	}
//	//	if(blockSize >= 256){
//	//		if(tid < 128){
//	//			numerator[tid] += numerator[tid+128];
//	//			denominator[tid] += denominator[tid+128];
//	//		}
//	//			__syncthreads();
//	//	}
//	//	if(blockSize >= 128){
//	//		if(tid < 64){
//	//			numerator[tid] += numerator[tid+64];
//	//			denominator[tid] += denominator[tid+64];
//	//		}
//	//				__syncthreads();
//	//	}
//	//	if(tid < 32){
//	//		if(blockSize >= 64){
//	//			numerator[tid] += numerator[tid+32];
//	//			denominator[tid] += denominator[tid+32];
//	//		}
//	//		if(blockSize >= 32){
//	//			numerator[tid] += numerator[tid+16];
//	//			denominator[tid] += denominator[tid+16];
//	//			
//	//		}
//	//		if(blockSize >= 16){
//	//			numerator[tid] += numerator[tid+8];
//	//			denominator[tid] += denominator[tid+8];
//	//			
//	//		}
//	//		if(blockSize >= 8){
//	//			numerator[tid] += numerator[tid+4];
//	//			denominator[tid] += denominator[tid+4];
//	//			
//	//		}
//	//		if(blockSize >= 4){
//	//			numerator[tid] += numerator[tid+2];
//	//			denominator[tid] += denominator[tid+2];
//	//			
//	//		}
//	//		if(blockSize >= 2){
//	//			numerator[tid] += numerator[tid+1];
//	//			denominator[tid] += denominator[tid+1];
//	//			
//	//		}
//	//	}
//	//	if(tid == 0){
//	//		if(denominator[0] == 0.0f)
//	//			filtered_device[y*width+x] = 0.0f;
//	//		else
//	//			filtered_device[y*width+x] = numerator[0]/denominator[0];
//	//	}
//	//}
//}

void JointBilateralFilter::Process(float* depth_device, cv::gpu::GpuMat color_image){
	
	cv::gpu::bilateralFilter(color_image, smooth_Device, 11, 50.0f, 30.0f);
	smooth_Device.download(smooth_Host);
	cv::imshow("smooth", smooth_Host);
	////joint bilateral filtering
	joint_bilateral_filtering<<<dim3(Width/32, Height/24), dim3(32, 24)>>>
		(Width, Height, depth_device, smooth_Device, SpatialFilter_Device, Filtered_Device, WindowSize, ColorSigma, DepthSigma);

	//joint_bilateral_filtering<8*8><<<dim3(Width, Height), dim3(8, 8)>>>
	//	(Width, Height, depth_device, color_image, SpatialFilter_Device, Filtered_Device, WindowSize, ColorSigma, DepthSigma);
	//cudaMemcpy(SpatialFilter_Host, SpatialFilter_Device, sizeof(float)*WindowSize*WindowSize, cudaMemcpyHostToDevice);
	//for(int i=0; i<WindowSize; i++){
	//	for(int j=0; j< WindowSize; j++){
	//		std::cout << SpatialFilter_Host[i*WindowSize+j]<<std::endl;
	//	}
	//}
	//cudaMemcpy(Filtered_Host, Filtered_Device, sizeof(float)*Width*Height, cudaMemcpyDeviceToHost);
	//for(int y = 0; y < Height; y++){
	//	for(int x = 0; x < Width; x++){
	//		std::cout <<Filtered_Host[y*Width+x]<<std::endl;
	//	}
	//}
}
