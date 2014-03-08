#include "SuperpixelSegmentation.h"

__global__ void initLD(
	SuperpixelSegmentation::label_distance* pixel_ld,
	int width,
	int height,
	int2 cluster_num,
	int2 window_size){
		int x = blockIdx.x*blockDim.x+threadIdx.x;
		int y = blockIdx.y*blockDim.y+threadIdx.y;
		//current label
		pixel_ld[y*width+x].l = (y/window_size.y)*cluster_num.x + (x/window_size.x);
		pixel_ld[y*width+x].d = 999999.9f;
}

template<int blockSize>
__global__ void sampleInitialClusters_(
	cv::gpu::GpuMat color_input,
	SuperpixelSegmentation::superpixel*	mean,
	int width,
	int height,
	int2 window_size){
		__shared__ float gradient[blockSize];
		__shared__ int	 arounds_x[blockSize];
		__shared__ int	 arounds_y[blockSize];
		//center of cluster
		int2 center;
		center.x = blockIdx.x*window_size.x + window_size.x/2;
		center.y = blockIdx.y*window_size.y + window_size.y/2;
		//around center
		int2 around;
		around.x = center.x + threadIdx.x - blockDim.x/2;
		around.y = center.y + threadIdx.y - blockDim.y/2;
		//thread id
		int tid = threadIdx.y*blockDim.x+threadIdx.x;
		//compute gradient
		float sumG = 0.0f;
		int count = 0;
		float g;
		for(int yy = -5; yy <= 5; yy++){
			for(int xx = -5; xx <= 5; xx++){
				int lx = xx + around.x;
				lx = lx > 0 ? lx : 0;
				lx = lx < width ? lx : width - 1;

				int ly = yy + around.y;
				ly = ly > 0 ? ly : 0;
				ly = ly < height ? ly : height - 1;

				g = sqrtf(pow((float)color_input.data[(around.y*width+around.x)*3]-(float)color_input.data[(yy*width+xx)*3], 2) +
					pow((float)color_input.data[(around.y*width+around.x)*3+1]-(float)color_input.data[(yy*width+xx)*3+1], 2) +
					pow((float)color_input.data[(around.y*width+around.x)*3+2]-(float)color_input.data[(yy*width+xx)*3+2], 2));
				count += g > 0.0 ? 1 : 0;
				sumG += g;
			}
		}
		//全点についてgradientを計算
		gradient[tid] = sumG/(float)count;
		arounds_x[tid] = around.x;
		arounds_y[tid] = around.y;
		__syncthreads();

		//compute min value
		if(blockSize >= 1024){
			if(tid < 512){
				if(gradient[tid] > gradient[tid+512]){
					gradient[tid] = gradient[tid+512];
					arounds_x[tid] = arounds_x[tid+512];
					arounds_y[tid] = arounds_y[tid+512];
					__syncthreads();
				}
			}
		}
		if(blockSize >= 512){
			if(tid < 256){
				if(gradient[tid] > gradient[tid+256]){
					gradient[tid] = gradient[tid+256];
					arounds_x[tid] = arounds_x[tid+256];
					arounds_y[tid] = arounds_y[tid+256];
					__syncthreads();
				}
			}
		}
		if(blockSize >= 256){
			if(tid < 128){
				if(gradient[tid] > gradient[tid+128]){
					gradient[tid] = gradient[tid+128];
					arounds_x[tid] = arounds_x[tid+128];
					arounds_y[tid] = arounds_y[tid+128];
					__syncthreads();
				}
			}
		}
		if(blockSize >= 128){
			if(tid < 64){
				if(gradient[tid] > gradient[tid+64]){
					gradient[tid] = gradient[tid+64];
					arounds_x[tid] = arounds_x[tid+64];
					arounds_y[tid] = arounds_y[tid+64];
					__syncthreads();
				}
			}
		}
		if(tid < 32){
			if(blockSize >= 64){
				if(gradient[tid] > gradient[tid+32]){
					gradient[tid] = gradient[tid+32];
					arounds_x[tid] = arounds_x[tid+32];
					arounds_y[tid] = arounds_y[tid+32];
				}
			}
			if(blockSize >= 32){
				if(gradient[tid] > gradient[tid+16]){
					gradient[tid] = gradient[tid+16];
					arounds_x[tid] = arounds_x[tid+16];
					arounds_y[tid] = arounds_y[tid+16];
				}
			}
			if(blockSize >= 16){
				if(gradient[tid] > gradient[tid+8]){
					gradient[tid] = gradient[tid+8];
					arounds_x[tid] = arounds_x[tid+8];
					arounds_y[tid] = arounds_y[tid+8];
				}
			}
			if(blockSize >= 8){
				if(gradient[tid] > gradient[tid+4]){
					gradient[tid] = gradient[tid+4];
					arounds_x[tid] = arounds_x[tid+4];
					arounds_y[tid] = arounds_y[tid+4];
				}
			}
			if(blockSize >= 4){
				if(gradient[tid] > gradient[tid+2]){
					gradient[tid] = gradient[tid+2];
					arounds_x[tid] = arounds_x[tid+2];
					arounds_y[tid] = arounds_y[tid+2];
				}
			}
			if(blockSize >= 2){
				if(gradient[tid] > gradient[tid+1]){
					gradient[tid] = gradient[tid+1];
					arounds_x[tid] = arounds_x[tid+1];
					arounds_y[tid] = arounds_y[tid+1];
				}
			}
		}
		//store center point
		if(tid == 0){
			int2 smooth;
			smooth.x = arounds_x[0];
			smooth.y = arounds_y[0];
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].x = (unsigned short)smooth.x;
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].y = (unsigned short)smooth.y;
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].r = color_input.data[(smooth.y*width+smooth.x)*3];
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].g = color_input.data[(smooth.y*width+smooth.x)*3+1];
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].b = color_input.data[(smooth.y*width+smooth.x)*3]+2;
		}

}

template<int blockSize>
__global__ void calculate_LD(
	cv::gpu::GpuMat color_input,
	SuperpixelSegmentation::label_distance* pixel_ld,
	SuperpixelSegmentation::superpixel* mean,
	int* labels,
	int2 window_size,
	int width,
	int height,
	float color_sigma,
	float spatial_sigma,
	int2 cluster_num){
		__shared__ int label_shared[blockSize];
		__shared__ float distance_shared[blockSize];
		//thread id
		int tid = threadIdx.y*blockDim.x+threadIdx.x;	
		//current pixel
		int x = blockIdx.x;
		int y = blockIdx.y;
		//current cluster pos
		int2 current_cluster;
		current_cluster.x = pixel_ld[y*width+x].l%cluster_num.x;
		current_cluster.y = pixel_ld[y*width+x].l/cluster_num.x;
		//assign threads around each cluster
		int2 ref_cluster;
		ref_cluster.x = current_cluster.x-blockDim.x/2+threadIdx.x;
		ref_cluster.y = current_cluster.y-blockDim.y/2+threadIdx.y;
		//ref cluster id
		int ref_cluster_id = ref_cluster.y*cluster_num.x+ref_cluster.x;
		if(ref_cluster.x>=0 && ref_cluster.x<cluster_num.x && 
			ref_cluster.y>=0 && ref_cluster.y<cluster_num.y){
				int2 ref_center;
				ref_center.x = (int)mean[ref_cluster_id].x;
				ref_center.y = (int)mean[ref_cluster_id].y;
				//calculate distance
				float color_distance = pow((float)color_input.data[(y*width+x)*3]-(float)mean[ref_cluster_id].r, 2) +
					pow((float)color_input.data[(y*width+x)*3+1]-(float)mean[ref_cluster_id].g, 2) +
					pow((float)color_input.data[(y*width+x)*3+2]-(float)mean[ref_cluster_id].b, 2);
				//calculate pixel distance
				float pixel_distance = sqrtf(pow((float)(x-ref_center.x), 2) + pow((float)(y-ref_center.y), 2))*pow((float)(window_size.x+window_size.y)/2.0f, 2);
				//set current ld
				//distance_shared[tid] = sqrt(pow(color_distance, 2) + pow(pixel_distance / entropy, 2) * pow((float)(window_size.x+window_size.y)/2.0f, 2));
				//distance_shared[tid] = pow(color_distance, 2)/spatial_sigma+pow(pixel_distance, 2) * pow((float)(window_size.x+window_size.y)/2.0f, 2)/color_sigma;
				distance_shared[tid] = color_distance*color_sigma/(spatial_sigma+color_sigma) + pixel_distance*spatial_sigma/(spatial_sigma+color_sigma);
				label_shared[tid] = ref_cluster.y*cluster_num.x+ref_cluster.x;		
		}
		else{
			distance_shared[tid] = pixel_ld[y*width+x].d;
			label_shared[tid] = pixel_ld[y*width+x].l;
		}
		__syncthreads();

		//assign cluster label
		if(blockSize >= 1024){
			if(tid < 512){
				if(distance_shared[tid] > distance_shared[tid+512]){
					label_shared[tid] = label_shared[tid+512];
					distance_shared[tid] = distance_shared[tid+512];
				}
				__syncthreads();
			}
		}
		if(blockSize >= 512){
			if(tid < 256){
				if(distance_shared[tid]> distance_shared[tid+256]){
					label_shared[tid] = label_shared[tid+256];
					distance_shared[tid] = distance_shared[tid+256];
				}
				__syncthreads();
			}
		}
		if(blockSize >= 256){
			if(tid < 128){
				if(distance_shared[tid] > distance_shared[tid+128]){
					label_shared[tid] = label_shared[tid+128];
					distance_shared[tid] = distance_shared[tid+128];
				}
				__syncthreads();
			}
		}
		if(blockSize >= 128){
			if(tid < 64){
				if(distance_shared[tid] > distance_shared[tid+64]){
					label_shared[tid] = label_shared[tid+64];
					distance_shared[tid] = distance_shared[tid+64];
				}
				__syncthreads();
			}
		}
		if(tid < 32){
			if(blockSize >= 64){
				if(distance_shared[tid] > distance_shared[tid+32]){
					label_shared[tid] = label_shared[tid+32];
					distance_shared[tid] = distance_shared[tid+32];
				}
			}
			if(blockSize >= 32){
				if(distance_shared[tid] > distance_shared[tid+16]){
					label_shared[tid] = label_shared[tid+16];
					distance_shared[tid] = distance_shared[tid+16];
				}
			}
			if(blockSize >= 16){
				if(distance_shared[tid] > distance_shared[tid+8]){
					label_shared[tid] = label_shared[tid+8];
					distance_shared[tid] = distance_shared[tid+8];
				}
			}
			if(blockSize >= 8){
				if(distance_shared[tid] > distance_shared[tid+4]){
					label_shared[tid] = label_shared[tid+4];
					distance_shared[tid] = distance_shared[tid+4];
				}
			}
			if(blockSize >= 4){
				if(distance_shared[tid] > distance_shared[tid+2]){
					label_shared[tid] = label_shared[tid+2];
					distance_shared[tid] = distance_shared[tid+2];
				}
			}
			if(blockSize >= 2){
				if(distance_shared[tid] > distance_shared[tid+1]){
					label_shared[tid] = label_shared[tid+1];
					distance_shared[tid] = distance_shared[tid+1];
				}
			}
		}
		//store center point
		if(tid == 0){
			pixel_ld[y*width+x].l = label_shared[0];
			pixel_ld[y*width+x].d = distance_shared[0];
			labels[y*width+x] = pixel_ld[y*width+x].l;
		}
}

template<int blockSize>
__global__ void analyzeClusters(
	cv::gpu::GpuMat color_input,
	SuperpixelSegmentation::label_distance* pixel_ld,
	SuperpixelSegmentation::superpixel* mean,
	int2 window_size,
	int2 cluster_num,
	int width,
	int height){
		//4*6=24 Byte 16384/24 = 26*26 threads
		__shared__ int r_shared[blockSize];
		__shared__ int g_shared[blockSize];
		__shared__ int b_shared[blockSize];
		__shared__ int x_shared[blockSize];
		__shared__ int y_shared[blockSize];
		__shared__ int size_shared[blockSize];
		//thread id
		int tid = threadIdx.y*blockDim.x+threadIdx.x;
		r_shared[tid] = 0;
		g_shared[tid] = 0;
		b_shared[tid] = 0;
		x_shared[tid] = 0;
		y_shared[tid] = 0;
		size_shared[tid] = 0;
		//current cluster
		int2 cluster_pos;
		cluster_pos.x = blockIdx.x;
		cluster_pos.y = blockIdx.y;
		int cluster_id = cluster_pos.y*cluster_num.x+cluster_pos.x;
		//assign threads around cluster
		int2 arounds;
		int2 ref_pixels;
		ref_pixels.x = window_size.x*4/blockDim.x+1;
		ref_pixels.y = window_size.y*4/blockDim.y+1;
		for(int yy=0; yy<ref_pixels.y; yy++){
			for(int xx=0; xx<ref_pixels.x; xx++){
				arounds.x = mean[cluster_id].x+(threadIdx.x-blockDim.x/2)*ref_pixels.x+xx;
				arounds.y = mean[cluster_id].y+(threadIdx.y-blockDim.y/2)*ref_pixels.y+yy;
				if(arounds.x>=0 && arounds.x<width && arounds.y>=0 && arounds.y<height){
					int around_id = pixel_ld[arounds.y*width+arounds.x].l;
					if(around_id == cluster_id){
						int r = (int)color_input.data[(arounds.y*width+arounds.x)*3];
						int g = (int)color_input.data[(arounds.y*width+arounds.x)*3+1];
						int b = (int)color_input.data[(arounds.y*width+arounds.x)*3+2];
						r_shared[tid] += r>255 ? 255:r;
						g_shared[tid] += g>255 ? 255:g;
						b_shared[tid] += b>255 ? 255:b;
						x_shared[tid] += arounds.x;
						y_shared[tid] += arounds.y;
						size_shared[tid] += 1;
					}
				}
			}
		}
		////assign threads around cluster
		//int2 arounds;
		//arounds.x = mean[cluster_id].x+(threadIdx.x-blockDim.x/2);
		//arounds.y = mean[cluster_id].y+(threadIdx.y-blockDim.y/2);
		//
		//	if(arounds.x>=0 && arounds.x<width && arounds.y>=0 && arounds.y<height){
		//			int around_id = pixel_ld[arounds.y*width+arounds.x].l;
		//			if(around_id == cluster_id){
		//				int r = (int)color_input.data[(arounds.y*width+arounds.x)*3];
		//				int g = (int)color_input.data[(arounds.y*width+arounds.x)*3+1];
		//				int b = (int)color_input.data[(arounds.y*width+arounds.x)*3+2];
		//				r_shared[tid] += r>255 ? 255:r;
		//				g_shared[tid] += g>255 ? 255:g;
		//				b_shared[tid] += b>255 ? 255:b;
		//				x_shared[tid] += arounds.x;
		//				y_shared[tid] += arounds.y;
		//				size_shared[tid] += 1;
		//			}
		//		}
		__syncthreads();
		//assign cluster label
		if(blockSize >= 1024){
			if(tid < 512){
				r_shared[tid] += r_shared[tid+512];
				g_shared[tid] += g_shared[tid+512];
				b_shared[tid] += b_shared[tid+512];
				x_shared[tid] += x_shared[tid+512];
				y_shared[tid] += y_shared[tid+512];
				size_shared[tid] += size_shared[tid+512];
			}
			__syncthreads();
		}
		if(blockSize >= 512){
			if(tid < 256){
				r_shared[tid] += r_shared[tid+256];
				g_shared[tid] += g_shared[tid+256];
				b_shared[tid] += b_shared[tid+256];
				x_shared[tid] += x_shared[tid+256];
				y_shared[tid] += y_shared[tid+256];
				size_shared[tid] += size_shared[tid+256];
			}
			__syncthreads();
		}
		if(blockSize >= 256){
			if(tid < 128){
				r_shared[tid] += r_shared[tid+128];
				g_shared[tid] += g_shared[tid+128];
				b_shared[tid] += b_shared[tid+128];
				x_shared[tid] += x_shared[tid+128];
				y_shared[tid] += y_shared[tid+128];
				size_shared[tid] += size_shared[tid+128];
			}
				__syncthreads();

		}
		if(blockSize >= 128){
			if(tid < 64){
				r_shared[tid] += r_shared[tid+64];
				g_shared[tid] += g_shared[tid+64];
				b_shared[tid] += b_shared[tid+64];
				x_shared[tid] += x_shared[tid+64];
				y_shared[tid] += y_shared[tid+64];
				size_shared[tid] += size_shared[tid+64];
			}
				__syncthreads();

		}
		//calculate average
		if(tid < 32){
			if(blockSize >= 64){
				r_shared[tid] += r_shared[tid+32];
				g_shared[tid] += g_shared[tid+32];
				b_shared[tid] += b_shared[tid+32];
				x_shared[tid] += x_shared[tid+32];
				y_shared[tid] += y_shared[tid+32];
				size_shared[tid] += size_shared[tid+32];
			}
			if(blockSize >= 32){
				r_shared[tid] += r_shared[tid+16];
				g_shared[tid] += g_shared[tid+16];
				b_shared[tid] += b_shared[tid+16];
				x_shared[tid] += x_shared[tid+16];
				y_shared[tid] += y_shared[tid+16];
				size_shared[tid] += size_shared[tid+16];
			}
			if(blockSize >= 16){
				r_shared[tid] += r_shared[tid+8];
				g_shared[tid] += g_shared[tid+8];
				b_shared[tid] += b_shared[tid+8];
				x_shared[tid] += x_shared[tid+8];
				y_shared[tid] += y_shared[tid+8];
				size_shared[tid] += size_shared[tid+8];
			}
			if(blockSize >= 8){
				r_shared[tid] += r_shared[tid+4];
				g_shared[tid] += g_shared[tid+4];
				b_shared[tid] += b_shared[tid+4];
				x_shared[tid] += x_shared[tid+4];
				y_shared[tid] += y_shared[tid+4];
				size_shared[tid] += size_shared[tid+4];
			}
			if(blockSize >= 4){
				r_shared[tid] += r_shared[tid+2];
				g_shared[tid] += g_shared[tid+2];
				b_shared[tid] += b_shared[tid+2];
				x_shared[tid] += x_shared[tid+2];
				y_shared[tid] += y_shared[tid+2];
				size_shared[tid] += size_shared[tid+2];
			}
			if(blockSize >= 2){
				r_shared[tid] += r_shared[tid+1];
				g_shared[tid] += g_shared[tid+1];
				b_shared[tid] += b_shared[tid+1];
				x_shared[tid] += x_shared[tid+1];
				y_shared[tid] += y_shared[tid+1];
				size_shared[tid] += size_shared[tid+1];
			}
		}
		//store center point
		if(tid == 0){
			if(size_shared[0] != 0){
				int r = r_shared[0]/size_shared[0]>255 ? 255:r_shared[0]/size_shared[0];
				int g = g_shared[0]/size_shared[0]>255 ? 255:g_shared[0]/size_shared[0];
				int b = b_shared[0]/size_shared[0]>255 ? 255:b_shared[0]/size_shared[0];
				r = r<0 ? 0:r;
				g = g<0 ? 0:g;
				b = b<0 ? 0:b;
				mean[cluster_id].r = (unsigned char)(r);
				mean[cluster_id].g = (unsigned char)(g);
				mean[cluster_id].b = (unsigned char)(b);
				mean[cluster_id].x = (int)(x_shared[0]/size_shared[0]);
				mean[cluster_id].y = (int)(y_shared[0]/size_shared[0]);
				mean[cluster_id].size = size_shared[0];
			}
		}

}
void SuperpixelSegmentation::Process(cv::gpu::GpuMat color_image, float color_sigma, float spatial_sigma, int iteration){
	//init label distance
	initLD<<<dim3(width/32, height/32), dim3(32, 32)>>>
		(LD_Device, width, height, ClusterNum, Window_Size);
	//sample clusters, move centers
	sampleInitialClusters_<16*16><<<dim3(ClusterNum.x, ClusterNum.y), dim3(16, 16)>>>
		(color_image, meanData_Device, width, height, Window_Size);
	for(int i = 0; i < iteration; i++){
		//Set cluster IDs	
		calculate_LD<4*4><<<dim3(width, height), dim3(4, 4)>>>
			(color_image, LD_Device, meanData_Device, Labels_Device, Window_Size, width, height, color_sigma, spatial_sigma, ClusterNum);
		analyzeClusters<16*16><<<dim3(ClusterNum.x, ClusterNum.y), dim3(16, 16)>>>
			(color_image, LD_Device, meanData_Device, Window_Size, ClusterNum, width, height);
	}
	cudaMemcpy(Labels_Host, Labels_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//__global__ void computeGrd_Kernel(
//	cv::gpu::GpuMat color_input,
//	float* grd,
//	int width,
//	int height){
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	//SuperpixelSegmentation::rgb* center = &in[x + y * width];
//	int lx;
//	int ly;
//	float sumG = 0.0;
//	int count = 0;
//	float g;
//	//gradient は中心点の3*3で計算
//	for(int yy = -5; yy <= 5; yy++){
//		for(int xx = -5; xx <= 5; xx++){
//			lx = xx + x;
//			lx = lx > 0 ? lx : 0;
//			lx = lx < width ? lx : width - 1;
//
//			ly = yy + y;
//			ly = ly > 0 ? ly : 0;
//			ly = ly < height ? ly : height - 1;
//
//			g = sqrtf(pow((float)color_input.data[(y*width+x)*3]-(float)color_input.data[(yy*width+xx)*3], 2) +
//						pow((float)color_input.data[(y*width+x)*3+1]-(float)color_input.data[(yy*width+xx)*3+1], 2) +
//							pow((float)color_input.data[(y*width+x)*3+2]-(float)color_input.data[(yy*width+xx)*3+2], 2));
//			count += g > 0.0 ? 1 : 0;
//			sumG += g;
//		}
//	}
//	//全点についてgradientを計算
//	grd[x + y * width] = sumG / count;
//}
//template<int blockSize>
//__global__ void compute_Gradient_(
//	cv::gpu::GpuMat color_input,
//	float* grd,
//	int width,
//	int height){
//	__shared__ float gradient[blockSize];
//	__shared__ int count[blockSize];
//	//blockId = center_position
//	int2 center, current;
//	center.x = blockIdx.x;
//	center.y = blockIdx.y;
//	//current position
//	current.x = blockIdx.x + threadIdx.x - blockDim.x/2;
//	current.y = blockIdx.y + threadIdx.y - blockDim.y/2;
//	//thread id
//	unsigned int tid = threadIdx.y*blockDim.x+threadIdx.x;
//	if(current.x >=0 && current.x < width && current.y >= 0 && current.y < height){
//		gradient[tid] = sqrtf(pow((float)color_input.data[(current.y*width+current.x)*3]-(float)color_input.data[(center.y*width+center.x)*3], 2) +
//						pow((float)color_input.data[(current.y*width+current.x)*3+1]-(float)color_input.data[(center.y*width+center.x)*3+1], 2) +
//							pow((float)color_input.data[(current.y*width+current.x)*3+2]-(float)color_input.data[(center.y*width+center.x)*3+2], 2));
//		count[tid] = 1;
//	}
//	else{
//		gradient[tid] = 0.0f;
//		count[tid] = 0;
//	}
//	__syncthreads();
//	//compute diffence
//	if(blockSize >= 1024){
//		if(threadIdx.y*blockDim.x+threadIdx.x < 512){
//			gradient[tid] += gradient[tid+512];
//			count[tid] += count[tid+512];
//			__syncthreads();
//		}
//	}
//	if(blockSize >= 512){
//		if(threadIdx.y*blockDim.x+threadIdx.x < 256){
//			gradient[tid] += gradient[tid+256];
//			count[tid] += count[tid+256];
//			__syncthreads();
//		}
//	}
//    if(blockSize >= 256){
//		if(threadIdx.y*blockDim.x+threadIdx.x < 128){
//			gradient[tid] += gradient[tid+128];
//			count[tid] += count[tid+128];
//			__syncthreads();
//		}
//	}
//	if(blockSize >= 128){
//		if(threadIdx.y*blockDim.x+threadIdx.x < 64){
//			gradient[tid] += gradient[tid+64];
//			count[tid] += count[tid+64];
//			__syncthreads();
//		}
//	}
//	if(tid < 32){
//		if(blockSize >= 64){
//			gradient[tid] += gradient[tid+32];
//			count[tid] += count[tid+32];
//		}
//		if(blockSize >= 32){
//			gradient[tid] += gradient[tid+16];
//			count[tid] += count[tid+16];
//		}
//		if(blockSize >= 16){
//			gradient[tid] += gradient[tid+8];
//			count[tid] += count[tid+8];
//		}
//		if(blockSize >= 8){
//			gradient[tid] += gradient[tid+4];
//			count[tid] += count[tid+4];
//		}
//		if(blockSize >= 4){
//			gradient[tid] += gradient[tid+2];
//			count[tid] += count[tid+2];
//		}
//		if(blockSize >= 2){
//			gradient[tid] += gradient[tid+1];
//			count[tid] += count[tid+1];
//		}
//	}
//	if(tid == 0)
//		grd[blockIdx.y*width+blockIdx.x] = gradient[0]/(float)count[0];
//}
////void SuperpixelSegmentation::computeGrd(){
////	computeGrd_Kernel<<<dim3(width / 32, height / 32), dim3(32, 32)>>>
////		(ColorInput_Device, grd, width, height);
////	//compute_Gradient_<8*8><<<dim3(width, height), dim3(8, 8)>>>
////	//	(ColorInput_Device, grd, width, height);
////}
//
//__global__ void sampleInitialClusters_Kernel(
//	SuperpixelSegmentation::superpixel*	mean,
//	float* grd,
//	int width,
//	int height,
//	int2 window_size){
//
//	int index_y = width / window_size.x;	
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	int x = (i % index_y) * window_size.x;
//	int y = (i / index_y) * window_size.y;	
//
//	x = x >= width ? width - 1 : x;
//	y = y >= height ? height - 1 : y;
//
//	int lx;
//	int ly;
//	float grd_c = grd[x + y * width];
//	int cx = x;
//	int cy = y;
//	//正規的に分布したseedの周辺3*3のなかでgradientが最小のところにseedを置く
//	for(int i = -1; i <= 1; i++){
//		for(int j = -1; j <= 1; j++){
//			lx = j + x;
//			lx = lx > 0 ? lx : 0;
//			lx = lx < width ? lx : width - 1;
//
//			ly = i + y;
//			ly = ly > 0 ? ly : 0;
//			ly = ly < height ? ly : height - 1;
//			if(grd[lx + ly * width] < grd_c){
//				cx = lx;
//				cy = ly;
//				grd_c = grd[lx + ly * width];
//			}						
//		}
//	}	
//	mean[i].x = cx;
//	mean[i].y = cy;
//}
//
//void SuperpixelSegmentation::sampleInitialClusters(){
//	///////////////////////////k/20がint かつ　32*32=1024を超えない
//	//dim3 gridSize(20);
//	//dim3 blockSize(Cluster_Num / 20);
//	//cluster center sampling
//	//sampleInitialClusters_Kernel<<<gridSize, blockSize>>>
//	//	(meanData_Device, grd, width, height, Window_Size);
//	
//}

//__global__ void set_ld_Kernel(
//	cv::gpu::GpuMat color_input,
//	SuperpixelSegmentation::label_distance* pixel_ld,
//	SuperpixelSegmentation::superpixel* mean,
//	int2 window_size,
//	int width,
//	int height,
//	float entropy){
//		//64 threads per each cluster
//		int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//		//distanceとlabel
//		SuperpixelSegmentation::label_distance current_ld;
//		int dy_min, dy_max, dx_min, dx_max;
//		int x = mean[i/64].x;
//		int y = mean[i/64].y;
//		int lx;
//		int ly;
//		//cluster id
//		int cluster = i / 64;
//		//i%64 各クラスタの何番目のスレッドであるか
//		//
//		switch((i % 64) % 8){
//		case 0:
//			dx_min = x - 2 * window_size.x;
//			dx_max = x - 1.5 * window_size.x;
//			break;
//		case 1:
//			dx_min = x - 1.5 * window_size.x + 1;
//			dx_max = x - window_size.x;
//			break;
//		case 2:
//			dx_min = x - window_size.x + 1;
//			dx_max = x - 0.5 * window_size.x;
//			break;
//		case 3:
//			dx_min = x - 0.5 * window_size.x + 1;
//			dx_max = x;
//			break;
//		case 4:
//			dx_min = x + 1;
//			dx_max = x + 0.5 * window_size.x;
//			break;
//		case 5:
//			dx_min = x + 0.5 * window_size.x + 1;
//			dx_max = x + window_size.x;
//			break;
//		case 6:
//			dx_min = x + window_size.x + 1;
//			dx_max = x + 1.5 * window_size.x;
//			break;
//		case 7:
//			dx_min = x + 1.5 * window_size.x + 1;
//			dx_max = x + 2 * window_size.x;
//			break;
//
//		default:
//			dx_min = 0;
//			dx_max = 0;
//			break;
//		}
//
//		switch((i % 64) / 8){
//		case 0:
//			dy_min = y - 2 * window_size.y;
//			dy_max = y - 1.5 * window_size.y;
//			break;	 
//		case 1:		 
//			dy_min = y - 1.5 * window_size.y + 1;
//			dy_max = y - window_size.y;
//			break;	
//		case 2:		
//			dy_min = y - window_size.y + 1;
//			dy_max = y - 0.5 * window_size.y;
//			break;	 
//		case 3:		 
//			dy_min = y - 0.5 * window_size.y + 1;
//			dy_max = y;
//			break;	 
//		case 4:		 
//			dy_min = y + 1;
//			dy_max = y + 0.5 * window_size.y;
//			break;	 
//		case 5:		 
//			dy_min = y + 0.5 * window_size.y + 1;
//			dy_max = y + window_size.y;
//			break;	 
//		case 6:		 
//			dy_min = y + window_size.y + 1;
//			dy_max = y + 1.5 * window_size.y;
//			break;	
//		case 7:		
//			dy_min = y + 1.5 * window_size.y + 1;
//			dy_max = y + 2 * window_size.y;
//			break;
//
//		default:
//			dy_min = 0;
//			dy_max = 0;
//			break;
//		}
//
//
//		for(int dy = dy_min; dy <= dy_max; dy++){	
//			ly = dy;
//			ly = ly < 0 ? 0 : ly;
//			ly = ly >= height ? height - 1 : ly;
//
//			for(int dx = dx_min; dx <= dx_max; dx++){				
//				lx = dx;
//				lx = lx < 0 ? 0 : lx;
//				lx = lx >= width ? width - 1 : lx;
//
//				//ref.color = pixel_in[lx + ly * width];
//				//ref.x = lx;
//				//ref.y = ly;
//				float ave_size = (float)(window_size.x+window_size.y)/2.0f;
//				//calculate color distance
//				float color_distance = sqrtf(pow((float)color_input.data[(y*width+x)*3]-(float)color_input.data[(ly*width+lx)*3], 2) +
//					pow((float)color_input.data[(y*width+x)*3+1]-(float)color_input.data[(ly*width+lx)*3+1], 2) +
//					pow((float)color_input.data[(y*width+x)*3+2]-(float)color_input.data[(ly*width+lx)*3+2], 2));
//				float pixel_distance = sqrtf(pow((float)(x-lx), 2) + pow((float)(y-ly), 2));
//
//				current_ld.d = sqrt(pow(color_distance, 2) + pow(pixel_distance / entropy, 2) * pow((float)(window_size.x+window_size.y)/2.0f, 2));
//				current_ld.l = cluster;
//
//				//hack for atomic instructions of label_distance
//				atomicMin((double*)&pixel_ld[lx + ly * width], (double*)&current_ld);
//			}
//		}	
//}
//
//void SuperpixelSegmentation::set_ld(){
//	//////////////////////////////////////////////////////////////////////////(k/16)*64が32*32を超えない
//	dim3 gridSize(192);
//	dim3 blockSize((ClusterNum.x*ClusterNum.y / 192) * 64);
//
//	set_ld_Kernel<<<gridSize, blockSize>>>(
//		ColorInput_Device, LD_Device, meanData_Device,	Window_Size, width, height, Entropy);
//
//	//compute_LD<<<dim3(((int)(Window_Size.x*2/8)+1)*2,((int)(Window_Size.x*2/8)+1)*2, Cluster_Num), dim3(8, 8)>>>(
//	//	ColorInput_Device, LD_Device, meanData_Device,	Window_Size, width, height, Entropy);
//
//}

//__global__ void compute_LD(
//	cv::gpu::GpuMat color_input,
//	SuperpixelSegmentation::label_distance* pixel_ld,
//	SuperpixelSegmentation::superpixel* mean,
//	int2 window_size,
//	int width,
//	int height,
//	float entropy,
//	int2 cluster_num,
//	int2 block_cluster){
//		//cluster id 
//		int2 cluster_pos;
//		cluster_pos.x = blockIdx.x/block_cluster.x;
//		cluster_pos.y = blockIdx.y/block_cluster.y;
//		int cluster_id = cluster_pos.y*cluster_num.x+cluster_pos.x;
//		//block id in each cluster
//		int2 block_pos;
//		block_pos.x = blockIdx.x%block_cluster.x;
//		block_pos.y = blockIdx.y%block_cluster.y;
//		//current position
//		int x = mean[cluster_id].x + (block_pos.x-block_cluster.x/2)*blockDim.x + threadIdx.x;
//		int y = mean[cluster_id].y + (block_pos.y-block_cluster.y/2)*blockDim.y + threadIdx.y;
//		if(x >= 0 && x < width && y >= 0 && y < height){
//			//calculate color distance
//			float color_distance = sqrtf(pow((float)color_input.data[(y*width+x)*3]-(float)color_input.data[(mean[cluster_id].y*width+mean[cluster_id].x)*3], 2) +
//				pow((float)color_input.data[(y*width+x)*3+1]-(float)color_input.data[(mean[cluster_id].y*width+mean[cluster_id].x)*3+1], 2) +
//				pow((float)color_input.data[(y*width+x)*3+2]-(float)color_input.data[(mean[cluster_id].y*width+mean[cluster_id].x)*3+2], 2));
//
//			//calculate pixel distance
//			float pixel_distance = sqrtf(pow((float)(x-mean[cluster_id].x), 2) + pow((float)(y-mean[cluster_id].y), 2));
//			//set current ld
//			SuperpixelSegmentation::label_distance current_ld;
//			current_ld.d = sqrt(pow(color_distance, 2) + pow(pixel_distance / entropy, 2) * pow((float)(window_size.x+window_size.y)/2.0f, 2));
//			current_ld.l = cluster_id;
//			//hack for atomic instructions of label_distance
//			atomicMin((double*)&pixel_ld[x + y * width], (double*)&current_ld);
//
//		}
//}
//__global__ void aggregateClusters_Kernel(
//	cv::gpu::GpuMat color_input,
//	SuperpixelSegmentation::label_distance* pixel_ld,
//	SuperpixelSegmentation::superpixel*	mean_data,
//	int width,
//	int height){
//		int x = blockIdx.x * blockDim.x + threadIdx.x;
//		int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//		atomicAdd(&mean_data[pixel_ld[y*width+x].l].r, (int)(color_input.data[(y*width+x)*3]));
//		atomicAdd(&mean_data[pixel_ld[y*width+x].l].g, (int)(color_input.data[(y*width+x)*3+1]));
//		atomicAdd(&mean_data[pixel_ld[y*width+x].l].b, (int)(color_input.data[(y*width+x)*3+2]));
//		atomicAdd(&mean_data[pixel_ld[y*width+x].l].x, x);
//		atomicAdd(&mean_data[pixel_ld[y*width+x].l].y, y);
//		atomicAdd(&mean_data[pixel_ld[y*width+x].l].size, 1);
//
//}							 
//
//
//__global__ void getNewCenters_Kernel(SuperpixelSegmentation::superpixel* mean_data){
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//	mean_data[i].r /= mean_data[i].size;
//	mean_data[i].g /= mean_data[i].size;
//	mean_data[i].b /= mean_data[i].size;
//	mean_data[i].x /= mean_data[i].size;
//	mean_data[i].y /= mean_data[i].size;
//
//}
//void SuperpixelSegmentation::aggregateClusters(){
//	aggregateClusters_Kernel<<<dim3(width / 32, height / 32), dim3(32, 32)>>>
//		(ColorInput_Device, LD_Device, meanData_Device, width, height);
//	//////////////////////////////////////////////////////////////////////////////////
//	dim3 gridSize(20);
//	dim3 blockSize(ClusterNum.x*ClusterNum.y / 20);
//
//	getNewCenters_Kernel<<<gridSize, blockSize>>>
//		(meanData_Device);	
//}

//__global__ void postProcess_Kernel(
//	SuperpixelSegmentation::label_distance* ld, 
//	int* labels,
//	int width, 
//	int height){
//		int x = blockIdx.x * blockDim.x + threadIdx.x;
//		int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//		SuperpixelSegmentation::label_distance* ld_ref = &ld[x + y * width];
//		SuperpixelSegmentation::label_distance* ld_neighbor;
//		//Clusterに入らなかったpixelを一番近くのclusterに配置
//		int dy;
//		int dx;
//		bool joint = false;
//		int label_c;
//
//		for(int i = -1; i <= 1 && !joint; i++){
//			dy = y + i;
//			dy = dy < 0 ? 0 : dy;
//			dy = dy >= height ? height - 1 : dy;
//			for(int j = -1; j <= 1 && !joint; j++){
//				dx = x + j;
//				dx = dx < 0 ? 0 : dx;
//				dx = dx >= width ? width - 1 : dx;
//				if(i == 0 && j == 0)
//					continue;
//
//				ld_neighbor = &ld[dx + dy * width];
//				joint = ld_ref->l == ld_neighbor->l;
//				label_c = ld_neighbor->l;
//			}
//		}
//
//		if(!joint)
//			ld_ref->l = label_c;
//
//		labels[x + y * width] = label_c;
//}
//
//
//
//void SuperpixelSegmentation::postProcess(){
//	postProcess_Kernel<<<dim3(width / 32, height / 32), dim3(32, 32)>>>
//		(LD_Device, Labels_Device, width, height);	
//}
//template<int blockSize>
//__global__ void calculate_LD(
//	cv::gpu::GpuMat color_input,
//	SuperpixelSegmentation::label_distance* pixel_ld,
//	SuperpixelSegmentation::superpixel* mean,
//	int* labels,
//	int2 window_size,
//	int width,
//	int height,
//	float color_sigma,
//	float spatial_sigma,
//	int2 cluster_num){
//		__shared__ int label_shared[blockSize];
//		__shared__ float distance_shared[blockSize];
//		//current pixel
//		int x = blockIdx.x;
//		int y = blockIdx.y;
//		//current cluster pos
//		int current_id = pixel_ld[y*width+x].l;
//		//thread id
//		int tid = threadIdx.y*blockDim.x+threadIdx.x;	
//		label_shared[tid] = pixel_ld[y*width+x].l;
//		distance_shared[tid] = pixel_ld[y*width+x].d;
//		//assign threads around cluster
//		int2 arounds;
//		int2 ref_pixels;
//		ref_pixels.x = window_size.x*2/blockDim.x+1;
//		ref_pixels.y = window_size.y*2/blockDim.y+1;
//		for(int yy=0; yy<ref_pixels.y; yy++){
//			for(int xx=0; xx<ref_pixels.x; xx++){
//				arounds.x = x+(threadIdx.x-blockDim.x/2)*ref_pixels.x+xx;
//				arounds.y = y+(threadIdx.y-blockDim.y/2)*ref_pixels.y+yy;
//				if(arounds.x>=0 && arounds.x<width && arounds.y>=0 && arounds.y<height){
//					//around id
//					int around_id = pixel_ld[arounds.y*width+arounds.x].l;
//					//color distance
//					float color_distance = (color_sigma/(color_sigma+spatial_sigma))*(pow((float)color_input.data[(y*width+x)*3]-(float)mean[around_id].r, 2.0f) +
//						pow((float)color_input.data[(y*width+x)*3+1]-(float)mean[around_id].g, 2.0f) +
//						pow((float)color_input.data[(y*width+x)*3+2]-(float)mean[around_id].b, 2.0f))/*/255.0f*/;
//					//spatial distance
//					float spatial_distance = (spatial_sigma/(color_sigma+spatial_sigma))*(pow((float)(x-mean[around_id].x), 2.0f) + pow((float)(y-mean[around_id].y), 2.0f))/pow((float)(window_size.x+window_size.y)/2.0f, 2);
//					if(distance_shared[tid] > color_distance+spatial_distance){
//						distance_shared[tid] = color_distance+spatial_distance;
//						label_shared[tid] = around_id;
//					}
//				}
//			}
//		}
//		//__shared__ int label_shared[blockSize];
////__shared__ float distance_shared[blockSize];
//////thread id
////int tid = threadIdx.y*blockDim.x+threadIdx.x;	
//////current pixel
////int x = blockIdx.x;
////int y = blockIdx.y;
//////current cluster pos
////int2 current_cluster;
////current_cluster.x = pixel_ld[y*width+x].l%cluster_num.x;
////current_cluster.y = pixel_ld[y*width+x].l/cluster_num.x;
//////assign threads around each cluster
////int2 ref_cluster;
////ref_cluster.x = current_cluster.x-blockDim.x/2+threadIdx.x;
////ref_cluster.y = current_cluster.y-blockDim.y/2+threadIdx.y;
//////ref cluster id
////int ref_cluster_id = ref_cluster.y*cluster_num.x+ref_cluster.x;
////if(ref_cluster.x>=0 && ref_cluster.x<cluster_num.x && 
////	ref_cluster.y>=0 && ref_cluster.y<cluster_num.y){
////		int2 ref_center;
////		ref_center.x = (int)mean[ref_cluster_id].x;
////		ref_center.y = (int)mean[ref_cluster_id].y;
////		//calculate distance
////		float color_distance = sqrtf(pow((float)color_input.data[(y*width+x)*3]-(float)mean[ref_cluster_id].r, 2) +
////			pow((float)color_input.data[(y*width+x)*3+1]-(float)mean[ref_cluster_id].g, 2) +
////			pow((float)color_input.data[(y*width+x)*3+2]-(float)mean[ref_cluster_id].b, 2));
////		//calculate pixel distance
////		float pixel_distance = sqrtf(pow((float)(x-ref_center.x), 2) + pow((float)(y-ref_center.y), 2));
////		//set current ld
////		//distance_shared[tid] = sqrt(pow(color_distance, 2) + pow(pixel_distance / entropy, 2) * pow((float)(window_size.x+window_size.y)/2.0f, 2));
////		distance_shared[tid] = pow(color_distance, 2)/spatial_sigma+pow(pixel_distance, 2) * pow((float)(window_size.x+window_size.y)/2.0f, 2)/color_sigma;
////		label_shared[tid] = ref_cluster.y*cluster_num.x+ref_cluster.x;		
////}
////else{
////	distance_shared[tid] = pixel_ld[y*width+x].d;
////	label_shared[tid] = pixel_ld[y*width+x].l;
////}
//		__syncthreads();
//
//		//assign cluster label
//		if(blockSize >= 1024){
//			if(tid < 512){
//				if(distance_shared[tid] > distance_shared[tid+512]){
//					label_shared[tid] = label_shared[tid+512];
//					distance_shared[tid] = distance_shared[tid+512];
//				}
//				__syncthreads();
//			}
//		}
//		if(blockSize >= 512){
//			if(tid < 256){
//				if(distance_shared[tid]> distance_shared[tid+256]){
//					label_shared[tid] = label_shared[tid+256];
//					distance_shared[tid] = distance_shared[tid+256];
//				}
//				__syncthreads();
//			}
//		}
//		if(blockSize >= 256){
//			if(tid < 128){
//				if(distance_shared[tid] > distance_shared[tid+128]){
//					label_shared[tid] = label_shared[tid+128];
//					distance_shared[tid] = distance_shared[tid+128];
//				}
//				__syncthreads();
//			}
//		}
//		if(blockSize >= 128){
//			if(tid < 64){
//				if(distance_shared[tid] > distance_shared[tid+64]){
//					label_shared[tid] = label_shared[tid+64];
//					distance_shared[tid] = distance_shared[tid+64];
//				}
//				__syncthreads();
//			}
//		}
//		if(tid < 32){
//			if(blockSize >= 64){
//				if(distance_shared[tid] > distance_shared[tid+32]){
//					label_shared[tid] = label_shared[tid+32];
//					distance_shared[tid] = distance_shared[tid+32];
//				}
//			}
//			if(blockSize >= 32){
//				if(distance_shared[tid] > distance_shared[tid+16]){
//					label_shared[tid] = label_shared[tid+16];
//					distance_shared[tid] = distance_shared[tid+16];
//				}
//			}
//			if(blockSize >= 16){
//				if(distance_shared[tid] > distance_shared[tid+8]){
//					label_shared[tid] = label_shared[tid+8];
//					distance_shared[tid] = distance_shared[tid+8];
//				}
//			}
//			if(blockSize >= 8){
//				if(distance_shared[tid] > distance_shared[tid+4]){
//					label_shared[tid] = label_shared[tid+4];
//					distance_shared[tid] = distance_shared[tid+4];
//				}
//			}
//			if(blockSize >= 4){
//				if(distance_shared[tid] > distance_shared[tid+2]){
//					label_shared[tid] = label_shared[tid+2];
//					distance_shared[tid] = distance_shared[tid+2];
//				}
//			}
//			if(blockSize >= 2){
//				if(distance_shared[tid] > distance_shared[tid+1]){
//					label_shared[tid] = label_shared[tid+1];
//					distance_shared[tid] = distance_shared[tid+1];
//				}
//			}
//		}
//		//store center point
//		if(tid == 0){
//			pixel_ld[y*width+x].l = label_shared[0];
//			pixel_ld[y*width+x].d = distance_shared[0];
//			labels[y*width+x] = pixel_ld[y*width+x].l;
//		}
//}


////////////////////////////////////////////////////////////////アルゴリズムが違う
//template<int blockSize>
//__global__ void calculate_LD(
//	cv::gpu::GpuMat color_input,
//	SuperpixelSegmentation::label_distance* pixel_ld,
//	SuperpixelSegmentation::superpixel* mean,
//	int* labels,
//	int2 window_size,
//	int width,
//	int height,
//	float color_sigma,
//	float spatial_sigma,
//	int2 cluster_num){
//		__shared__ int label_shared[blockSize];
//		__shared__ float distance_shared[blockSize];
//		//thread id
//		int tid = threadIdx.y*blockDim.x+threadIdx.x;	
//		//current pixel
//		int x = blockIdx.x;
//		int y = blockIdx.y;
//		//calculate search size
//		int2 search;
//		search.x = window_size.x*4/blockDim.x+1;
//		search.y = window_size.y*4/blockDim.y+1;
//		//ref pixel
//		int2 ref_pixel;
//		ref_pixel.x = x+(threadIdx.x-blockDim.x)/2*search.x;
//		ref_pixel.y = y+(threadIdx.y-blockDim.y)/2*search.y;
//		//min distance and label
//		SuperpixelSegmentation::label_distance min_ld = pixel_ld[y*width+x];
//		for(int yy=0; yy<search.y; yy++){
//			for(int xx=0; xx<search.x; xx++){
//				if(ref_pixel.x+xx>=0 && ref_pixel.x+xx<cluster_num.x && 
//					ref_pixel.y+yy>=0 && ref_pixel.y+yy<cluster_num.y){
//						int ref_cluster_id = pixel_ld[ref_pixel.y*width+ref_pixel.x].l;
//						//calculate distance
//						float color_distance = sqrtf(pow((float)color_input.data[(y*width+x)*3]-(float)mean[ref_cluster_id].r, 2) +
//							pow((float)color_input.data[(y*width+x)*3+1]-(float)mean[ref_cluster_id].g, 2) +
//							pow((float)color_input.data[(y*width+x)*3+2]-(float)mean[ref_cluster_id].b, 2));
//						//calculate pixel distance
//						float pixel_distance = sqrtf(pow((float)(x-mean[ref_cluster_id].x), 2) + pow((float)(y-mean[ref_cluster_id].y), 2));
//						//distance
//						float distance =  pow(color_distance, 2)/spatial_sigma+pow(pixel_distance, 2) * pow((float)(window_size.x+window_size.y)/2.0f, 2)/color_sigma;
//						if(min_ld.d > distance){
//							//set current ld
//							min_ld.l = ref_cluster_id;
//							min_ld.d = distance;
//						}
//				}
//			}
//		}
//		label_shared[tid]=min_ld.l;
//		distance_shared[tid]=min_ld.d;
//		__syncthreads();
//
//		//assign cluster label
//		if(blockSize >= 1024){
//			if(tid < 512){
//				if(distance_shared[tid] > distance_shared[tid+512]){
//					label_shared[tid] = label_shared[tid+512];
//					distance_shared[tid] = distance_shared[tid+512];
//				}
//				__syncthreads();
//			}
//		}
//		if(blockSize >= 512){
//			if(tid < 256){
//				if(distance_shared[tid]> distance_shared[tid+256]){
//					label_shared[tid] = label_shared[tid+256];
//					distance_shared[tid] = distance_shared[tid+256];
//				}
//				__syncthreads();
//			}
//		}
//		if(blockSize >= 256){
//			if(tid < 128){
//				if(distance_shared[tid] > distance_shared[tid+128]){
//					label_shared[tid] = label_shared[tid+128];
//					distance_shared[tid] = distance_shared[tid+128];
//				}
//				__syncthreads();
//			}
//		}
//		if(blockSize >= 128){
//			if(tid < 64){
//				if(distance_shared[tid] > distance_shared[tid+64]){
//					label_shared[tid] = label_shared[tid+64];
//					distance_shared[tid] = distance_shared[tid+64];
//				}
//				__syncthreads();
//			}
//		}
//		if(tid < 32){
//			if(blockSize >= 64){
//				if(distance_shared[tid] > distance_shared[tid+32]){
//					label_shared[tid] = label_shared[tid+32];
//					distance_shared[tid] = distance_shared[tid+32];
//				}
//			}
//			if(blockSize >= 32){
//				if(distance_shared[tid] > distance_shared[tid+16]){
//					label_shared[tid] = label_shared[tid+16];
//					distance_shared[tid] = distance_shared[tid+16];
//				}
//			}
//			if(blockSize >= 16){
//				if(distance_shared[tid] > distance_shared[tid+8]){
//					label_shared[tid] = label_shared[tid+8];
//					distance_shared[tid] = distance_shared[tid+8];
//				}
//			}
//			if(blockSize >= 8){
//				if(distance_shared[tid] > distance_shared[tid+4]){
//					label_shared[tid] = label_shared[tid+4];
//					distance_shared[tid] = distance_shared[tid+4];
//				}
//			}
//			if(blockSize >= 4){
//				if(distance_shared[tid] > distance_shared[tid+2]){
//					label_shared[tid] = label_shared[tid+2];
//					distance_shared[tid] = distance_shared[tid+2];
//				}
//			}
//			if(blockSize >= 2){
//				if(distance_shared[tid] > distance_shared[tid+1]){
//					label_shared[tid] = label_shared[tid+1];
//					distance_shared[tid] = distance_shared[tid+1];
//				}
//			}
//		}
//		//store center point
//		if(tid == 0){
//			pixel_ld[y*width+x].l = label_shared[0];
//			pixel_ld[y*width+x].d = distance_shared[0];
//			labels[y*width+x] = pixel_ld[y*width+x].l;
//		}
//}

//template<int blockSize>
//__global__ void analyzeClusters(
//	cv::gpu::GpuMat color_input,
//	SuperpixelSegmentation::label_distance* pixel_ld,
//	SuperpixelSegmentation::superpixel* mean,
//	int2 blocks_cluster,
//	int2 cluster_num,
//	int width,
//	int height){
//		__shared__ int r[blockSize];
//		__shared__ int g[blockSize];
//		__shared__ int b[blockSize];
//		__shared__ int x[blockSize];
//		__shared__ int y[blockSize];
//		__shared__ int size[blockSize];
//		//thread id
//		int tid = threadIdx.y*blockDim.x+threadIdx.x;
//		//cluster position
//		int2 cluster_pos;
//		cluster_pos.x = blockIdx.x/blocks_cluster.x;
//		cluster_pos.y = blockIdx.y/blocks_cluster.y;
//		int cluster_id = cluster_pos.y*cluster_num.x+cluster_pos.x;
//		////prev cluster center
//		//int2 prev_center;
//		//prev_center.x = (int)mean[cluster_id].x;
//		//prev_center.y = (int)mean[cluster_id].y;
//		//current pixel
//		int2 current_pix;
//		current_pix.x = (int)mean[cluster_id].x-blocks_cluster.x/2*blockDim.x+threadIdx.x;
//		current_pix.y = (int)mean[cluster_id].y-blocks_cluster.y/2*blockDim.y+threadIdx.y;
//		if(current_pix.x>=0 && current_pix.x<width && current_pix.y>=0 && current_pix.y<height){ 
//			//current label
//			int current_label = pixel_ld[current_pix.y*width+current_pix.x].l;
//			//analyze
//			if(current_label == cluster_id){
//				//get each value
//				r[tid] = color_input.data[(current_pix.y*width+current_pix.x)*3];
//				g[tid] = color_input.data[(current_pix.y*width+current_pix.x)*3+1];
//				b[tid] = color_input.data[(current_pix.y*width+current_pix.x)*3+2];
//				x[tid] = current_pix.x;
//				y[tid] = current_pix.y;
//				size[tid] = 1;
//			}
//			else{
//				r[tid] = 0;
//				g[tid] = 0;
//				b[tid] = 0;
//				x[tid] = 0;
//				y[tid] = 0;
//				size[tid] = 0;
//			}
//		}
//		else{
//			r[tid] = 0;
//			g[tid] = 0;
//			b[tid] = 0;
//			x[tid] = 0;
//			y[tid] = 0;
//			size[tid] = 0;
//		}
//		__syncthreads();
//
//		//calculate average
//
//		if(blockSize >= 32){
//			r[tid] += r[tid+16];
//			g[tid] += g[tid+16];
//			b[tid] += b[tid+16];
//			x[tid] += x[tid+16];
//			y[tid] += y[tid+16];
//			size[tid] += size[tid+16];
//		}
//		if(blockSize >= 16){
//			r[tid] += r[tid+8];
//			g[tid] += g[tid+8];
//			b[tid] += b[tid+8];
//			x[tid] += x[tid+8];
//			y[tid] += y[tid+8];
//			size[tid] += size[tid+8];
//		}
//		if(blockSize >= 8){
//			r[tid] += r[tid+4];
//			g[tid] += g[tid+4];
//			b[tid] += b[tid+4];
//			x[tid] += x[tid+4];
//			y[tid] += y[tid+4];
//			size[tid] += size[tid+4];
//		}
//		if(blockSize >= 4){
//			r[tid] += r[tid+2];
//			g[tid] += g[tid+2];
//			b[tid] += b[tid+2];
//			x[tid] += x[tid+2];
//			y[tid] += y[tid+2];
//			size[tid] += size[tid+2];
//		}
//		if(blockSize >= 2){
//			r[tid] += r[tid+1];
//			g[tid] += g[tid+1];
//			b[tid] += b[tid+1];
//			x[tid] += x[tid+1];
//			y[tid] += y[tid+1];
//			size[tid] += size[tid+1];
//		}
//		//store center point
//		if(tid == 0){
//			mean[cluster_id].r = (unsigned char)(r[0]/size[0]);
//			mean[cluster_id].g = (unsigned char)(g[0]/size[0]);
//			mean[cluster_id].b = (unsigned char)(b[0]/size[0]);
//			mean[cluster_id].x = (unsigned short)(x[0]/size[0]);
//			mean[cluster_id].y = (unsigned short)(y[0]/size[0]);
//		}
//
//}
