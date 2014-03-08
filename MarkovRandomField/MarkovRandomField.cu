#include "MarkovRandomField.h"

//computeBilateralFiltering
__global__ void markov_random_field(
	int width, int height,
	float* depth_device,
	cv::gpu::GpuMat color_image,
	float* filtered_device,
	int window_size,
	float color_sigma,
	float smooth_sigma){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		//calculate MRF
		float numerator = depth_device[y*width+x], denominator = 1.0f;
		for(int i = - window_size/2; i <= window_size/2; i++){		// y
			for(int j = -window_size/2; j <= window_size/2; j++){		// x
				int xj = x+j, yi = y+i;
				if(xj >= 0 && xj < width && yi >= 0 && yi < height && depth_device[yi*width+xj] > 50.0f){
					//calculate color diff
					float color_diff = powf((float)(color_image.data[(y*width+x)*3+0])-(float)(color_image.data[(yi*width+xj)*3+0]), 2) +
										powf((float)(color_image.data[(y*width+x)*3+1])-(float)(color_image.data[(yi*width+xj)*3+1]), 2) +
											powf((float)(color_image.data[(y*width+x)*3+2])-(float)(color_image.data[(yi*width+xj)*3+2]), 2); 
					float color_filter = 0.0f;
					if(color_sigma != 0.0f)
					color_filter = expf(-color_sigma*color_diff);
					//calculate filter
					float filter = smooth_sigma;
					filter *= color_filter;
					numerator += depth_device[yi*width+xj]*filter; 
					denominator += filter;
				}
			}
		}
		if(denominator == 0.0f)
			filtered_device[y*width+x] = 0.0f;
		else
			filtered_device[y*width+x] = numerator/denominator;

		////calculate weighted average
		//float w_average = 0.0f;
		//float weight = 0.0f;
		//for(int i = - window_size/2; i <= window_size/2; i++){		// y
		//	for(int j = -window_size/2; j <= window_size/2; j++){		// x
		//		int xj = x+j, yi = y+i;
		//		if(xj >= 0 && xj < width && yi >= 0 && yi < height && depth_device[yi*width+xj] > 50.0f){
		//			float color_diff = powf((float)(color_image.data[(y*width+x)*3+0])-(float)(color_image.data[(yi*width+xj)*3+0]), 2) +
		//								powf((float)(color_image.data[(y*width+x)*3+1])-(float)(color_image.data[(yi*width+xj)*3+1]), 2) +
		//									powf((float)(color_image.data[(y*width+x)*3+2])-(float)(color_image.data[(yi*width+xj)*3+2]), 2); 
		//			float color_filter = 0.0f;
		//			if(color_sigma != 0.0f)
		//			color_filter = expf(-color_diff/(2*powf(color_sigma, 2.0f)));
		//			//calculate filter
		//			float filter = 1.0f;
		//			if(color_filter != 0.0f)
		//				filter *= color_filter;
		//			w_average += depth_device[yi*width+xj]*filter;
		//			weight += filter;
		//		}
		//	}
		//}
		//if(weight > 0.0f){
		//	w_average /= weight;
		//
		////filtering
		//float numerator = 0.0f, denominator = 0.0f;
		////if(depth_device[y*width+x] > 50.0f){
		//for(int i = - window_size/2; i <= window_size/2; i++){		// y
		//	for(int j = -window_size/2; j <= window_size/2; j++){		// x
		//		int xj = x+j, yi = y+i;
		//		if(xj >= 0 && xj < width && yi >= 0 && yi < height && depth_device[yi*width+xj] > 50.0f){
		//			//color filter
		//			float color_diff = powf((float)(color_image.data[(y*width+x)*3+0])-(float)(color_image.data[(yi*width+xj)*3+0]), 2) +
		//								powf((float)(color_image.data[(y*width+x)*3+1])-(float)(color_image.data[(yi*width+xj)*3+1]), 2) +
		//									powf((float)(color_image.data[(y*width+x)*3+2])-(float)(color_image.data[(yi*width+xj)*3+2]), 2); 
		//			float color_filter = 0.0f;
		//			if(color_sigma != 0.0f)
		//			color_filter = expf(-color_diff/(2*powf(color_sigma, 2.0f)));
		//			//depth filter
		//			float depth_diff = powf(depth_device[yi*width+xj] - w_average, 2.0f);
		//			float depth_filter;
		//			if(depth_sigma!= 0.0f)
		//			depth_filter = expf(-depth_diff/(2.0f*powf(depth_sigma, 2.0f)));
		//			//calculate filter
		//			float filter = 1.0f;
		//			if(spatial_filter_device[(i+window_size/2)*window_size+(j+window_size/2)] != 0.0f)
		//				filter *= spatial_filter_device[(i+window_size/2)*window_size+(j+window_size/2)];
		//			if(color_filter != 0.0f)
		//				filter *= color_filter;
		//			if(depth_filter != 0.0f)
		//				filter *= depth_filter;
		//			numerator += depth_device[yi*width+xj]*filter; 
		//			denominator += filter;
		//		}
		//	}
		//}
		////}
		//if(denominator == 0.0f)
		//	filtered_device[y*width+x] = 0.0f;
		//else
		//	filtered_device[y*width+x] = numerator/denominator;
		//}
		//else{
		//	filtered_device[y*width+x] = 0.0f;
		//}
}

void MarkovRandomField::Process(float* depth_device, cv::gpu::GpuMat color_image){
	
	//cv::gpu::bilateralFilter(color_image, smooth_Device, 11, 50.0f, 30.0f);
	//smooth_Device.download(smooth_Host);
	//cv::imshow("smooth", smooth_Host);
	////joint bilateral filtering
	markov_random_field<<<dim3(Width/32, Height/24), dim3(32, 24)>>>
		(Width, Height, depth_device, color_image, Filtered_Device, WindowSize, ColorSigma, SmoothSigma);
}
