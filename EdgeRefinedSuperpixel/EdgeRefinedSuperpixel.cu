#include "EdgeRefinedSuperpixel.h"


__global__ void edge_refining(
	int width,
	int height,
	int* color_labels,
	float* depth,
	int* refined_labels,
	float* refined_depth,
	int window_size){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		//horizontal scan
		if(x+1<width){
			if(/*fabs(depth[y*width+x+1]-depth[y*width+x]) > 30.0f &&*/
				refined_labels[y*width+x] != refined_labels[y*width+x+1]){
					int current_color_label = color_labels[y*width+x];
					int target_color_label = current_color_label;
					int2 target_pix;
					target_pix.x = x;
					target_pix.y = y;
					int distance = 0;
					while((x-distance>=0 || x+distance<width) && 
							target_color_label == current_color_label &&
								distance <= window_size/2){
							target_pix.x = x-distance;
							if(target_pix.x >= 0)
								target_color_label = color_labels[y*width+target_pix.x];
							if(target_color_label != current_color_label){
								int refined_depth_label = refined_labels[y*width+x+1];
								for(int i=target_pix.x+1; i<=x; i++){
									refined_labels[y*width+i] = refined_depth_label;
									//refined_labels[y*width+i] = -100;
									if(fabs(refined_depth[y*width+i]-refined_depth[y*width+i+1])>refined_depth[y*width+i]*0.01f)
									refined_depth[y*width+i] = 0.0f;
								}
								break;
							}
							target_pix.x = x+distance;
							if(target_pix.x < width)
								target_color_label = color_labels[y*width+target_pix.x];
							if(target_color_label != current_color_label){
								int refined_depth_label = refined_labels[y*width+x];
								for(int i=x+1; i<=target_pix.x-1; i++){
									refined_labels[y*width+i] = refined_depth_label;
									//refined_labels[y*width+i] = -100;
									if(fabs(refined_depth[y*width+i]-refined_depth[y*width+i-1])>refined_depth[y*width+i]*0.01f)
									refined_depth[y*width+i] = 0.0f;
								}
								break;
							}
							distance++;
					}
			}
		}
		__syncthreads();
		//vertical scan
		if(y+1<height){
			if(/*fabs(depth[(y+1)*width+x]-depth[y*width+x]) > 30.0f &&*/
				refined_labels[y*width+x] != refined_labels[(y+1)*width+x]){
					int current_color_label = color_labels[y*width+x];
					int target_color_label = current_color_label;
					int2 target_pix;
					target_pix.x = x;
					target_pix.y = y;
					int distance = 0;
					while((y-distance>=0 || y+distance<height) && 
							target_color_label == current_color_label &&
								distance <= window_size/2){
							target_pix.y = y-distance;
							if(target_pix.y >= 0)
								target_color_label = color_labels[target_pix.y*width+x];
							if(target_color_label != current_color_label){
								int refined_depth_label = refined_labels[(y+1)*width+x];
								for(int i=target_pix.y+1; i<=y; i++){
									refined_labels[i*width+x] = refined_depth_label;
									//refined_labels[i*width+x] = -100;
									if(fabs(refined_depth[i*width+x]-refined_depth[(i+1)*width+x])>refined_depth[i*width+x]*0.01f)
									refined_depth[i*width+x] = 0.0f;
								}
								break;
							}
							target_pix.y = y+distance;
							if(target_pix.y < height)
								target_color_label = color_labels[target_pix.y*width+x];
							if(target_color_label != current_color_label){
								int refined_depth_label = refined_labels[y*width+x];
								for(int i=y+1; i<=target_pix.y-1; i++){
									refined_labels[i*width+x] = refined_depth_label;
									//refined_labels[i*width+x] = -100;
									if(fabs(refined_depth[i*width+x]-refined_depth[(i-1)*width+x])>refined_depth[i*width+x]*0.01f)
									refined_depth[i*width+x] = 0.0f;
								}
								break;
							}
							distance++;
					}
			}
		}
}
//computeBilateralFiltering
__global__ void depthmap_enhancement(
	int width, int height,
	float* refined_depth,
	cv::gpu::GpuMat color_image,
	int* refined_labels,
	float* spatial_filter_device,
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
				if(xj >= 0 && xj < width && yi >= 0 && yi < height && 
					refined_depth[yi*width+xj] > 50.0f /*&& refined_labels[y*width+x] == refined_labels[yi*width+xj]*/){
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
					w_average += refined_depth[yi*width+xj]*filter;
					weight += filter;
				}
			}
		}
		if(weight > 0.0f){
			w_average /= weight;
		//calculate deviation
		int count = 0;
		float deviation = 0.0f;
		for(int i = - window_size/2; i <= window_size/2; i++){		// y
			for(int j = -window_size/2; j <= window_size/2; j++){		// x
				int xj = x+j, yi = y+i;
				if(xj >= 0 && xj < width && yi >= 0 && yi < height && 
					refined_depth[yi*width+xj] > 50.0f && refined_labels[y*width+x] == refined_labels[yi*width+xj]){
					deviation += fabs(refined_depth[yi*width+xj]-w_average);
					count++;
				}
			}
		}
		if(count != 0)
			deviation /= (float)count;
		//filtering
		float numerator = 0.0f, denominator = 0.0f;
		//if(depth_device[y*width+x] > 50.0f){
		for(int i = - window_size/2; i <= window_size/2; i++){		// y
			for(int j = -window_size/2; j <= window_size/2; j++){		// x
				int xj = x+j, yi = y+i;
				if(xj >= 0 && xj < width && yi >= 0 && yi < height &&
					refined_depth[yi*width+xj] > 50.0f/* && refined_labels[y*width+x] == refined_labels[yi*width+xj]*/){
					//color filter
					float color_diff = powf((float)(color_image.data[(y*width+x)*3+0])-(float)(color_image.data[(yi*width+xj)*3+0]), 2) +
										powf((float)(color_image.data[(y*width+x)*3+1])-(float)(color_image.data[(yi*width+xj)*3+1]), 2) +
											powf((float)(color_image.data[(y*width+x)*3+2])-(float)(color_image.data[(yi*width+xj)*3+2]), 2); 
					float color_filter = 0.0f;
					if(color_sigma != 0.0f){
					float adaptive_sigma = 5.0*deviation/pow(w_average, 2.0f);
					if(adaptive_sigma > color_sigma*0.3f)
						color_sigma = adaptive_sigma;
					else
						color_sigma *= 0.3f;
					color_filter = expf(-color_diff/(2*powf(color_sigma, 2.0f)));
					}
					//depth filter
					float depth_diff = powf(refined_depth[yi*width+xj] - w_average, 2.0f);
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
					numerator += refined_depth[yi*width+xj]*filter; 
					denominator += filter;
				}
			}
		}
		
		if(denominator == 0.0f)
			refined_depth[y*width+x] = 0.0f;
		else
			refined_depth[y*width+x] = numerator/denominator;
		}
		else{
			refined_depth[y*width+x] = 0.0f;
		}
}


void EdgeRefinedSuperpixel::EdgeRefining(int* color_label_device, int* depth_label_device, float* depth_device, cv::gpu::GpuMat color_image){
	//copy data
	cudaMemcpy(refinedLabels_Device, depth_label_device, sizeof(int)*Width*Height, cudaMemcpyDeviceToDevice);
	cudaMemcpy(refinedDepth_Device, depth_device, sizeof(float)*Width*Height, cudaMemcpyDeviceToDevice);
	//edge refinig
	edge_refining<<<dim3(Width/32, Height/24), dim3(32, 24)>>>
		(Width, Height, color_label_device, depth_device, refinedLabels_Device, refinedDepth_Device, WindowSize);
	//for(int n=0; n<3; n++){
	//edge_refining<<<dim3(Width/32, Height/24), dim3(32, 24)>>>
	//	(Width, Height, color_label_device, refinedLabels_Device, depth_device, refinedLabels_Device, refinedDepth_Device, WindowSize);
	//}
	//depthmap enhancement
	depthmap_enhancement<<<dim3(Width/32, Height/24), dim3(32, 24)>>>
		(Width, Height, refinedDepth_Device, color_image, refinedLabels_Device, SpatialFilter_Device, WindowSize, ColorSigma, DepthSigma);
	cudaMemcpy(refinedLabels_Host, refinedLabels_Device, sizeof(int)*Width*Height, cudaMemcpyDeviceToHost);
	cudaMemcpy(refinedDepth_Host, refinedDepth_Device, sizeof(float)*Width*Height, cudaMemcpyDeviceToHost);

	//cudaMemcpy(colorLabels_Host, color_label_device, sizeof(int)*Width*Height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(depthLabels_Host, depth_label_device, sizeof(int)*Width*Height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(Depth_Host, depth_device, sizeof(float)*Width*Height, cudaMemcpyDeviceToHost);
	//for(int y = 0; y < Height; y++){
	//	for(int x = 0; x < Width; x++){
	//		//horizontal scan
	//		if(x+1<Width){
	//			if(/*fabs(Depth_Host[y*Width+x+1]-Depth_Host[y*Width+x]) > 30.0f &&*/
	//				depthLabels_Host[y*Width+x] != depthLabels_Host[y*Width+x+1]){
	//					int current_color_label = colorLabels_Host[y*Width+x];
	//					int target_color_label = current_color_label;
	//					int2 target_pix;
	//					target_pix.x = x;
	//					target_pix.y = y;
	//					int distance = 0;
	//					while((x-distance>=0 || x+distance<Width) && 
	//						target_color_label == current_color_label &&
	//						distance <= 15){
	//							target_pix.x = x-distance;
	//							if(target_pix.x >= 0)
	//								target_color_label = colorLabels_Host[y*Width+target_pix.x];
	//							if(target_color_label != current_color_label){
	//								int refined_depth_label = depthLabels_Host[y*Width+x+1];
	//								for(int i=target_pix.x+1; i<=x; i++){
	//									refinedLabels_Host[y*Width+i] = refined_depth_label;
	//									//refinedLabels_Host[y*Width+i] = -100;
	//									refinedDepth_Host[y*Width+i] = 0.0f;
	//								}
	//								break;
	//							}
	//							target_pix.x = x+distance;
	//							if(target_pix.x < Width)
	//								target_color_label = colorLabels_Host[y*Width+target_pix.x];
	//							if(target_color_label != current_color_label){
	//								int refined_depth_label = depthLabels_Host[y*Width+x];
	//								for(int i=x+1; i<=target_pix.x-1; i++){
	//									refinedLabels_Host[y*Width+i] = refined_depth_label;
	//									//refinedLabels_Host[y*Width+i] = -100;
	//									refinedDepth_Host[y*Width+i] = 0.0f;
	//								}
	//								break;
	//							}
	//							distance++;
	//					}
	//			}
	//		}
	//	}
	//}
}