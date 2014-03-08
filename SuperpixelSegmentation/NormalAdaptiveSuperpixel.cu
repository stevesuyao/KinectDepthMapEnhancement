#include "NormalAdaptiveSuperpixel.h"

__global__ void initLD_NASP(
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
__global__ void sampleInitialClusters_NASP(
	cv::gpu::GpuMat color_input,
	float3*		input_points,
	float3*		normals,
	SuperpixelSegmentation::superpixel*	mean,
	float3*		sp_centers,
	float3*		sp_normals,
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
				if(normals[around.y*width+around.x].x != -1.0f && 
						normals[around.y*width+around.x].y != -1.0f && 
							normals[around.y*width+around.x].z != -1.0f){
					if(normals[yy*width+xx].x != -1.0f && 
						normals[yy*width+xx].y != -1.0f && 
							normals[yy*width+xx].z != -1.0f){
								float normal_diff = abs(normals[around.y*width+around.x].x*normals[yy*width+xx].x +
															normals[around.y*width+around.x].y*normals[yy*width+xx].y +
																normals[around.y*width+around.x].z*normals[yy*width+xx].z);
								g *= (1.0f-normal_diff);
					}
				}
				count += g > 0.0 ? 1 : 0;
				sumG += g;
			}
		}
		//‘S“_‚É‚Â‚¢‚Ägradient‚ðŒvŽZ
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
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].x = (int)smooth.x;
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].y = (int)smooth.y;
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].r = color_input.data[(smooth.y*width+smooth.x)*3];
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].g = color_input.data[(smooth.y*width+smooth.x)*3+1];
			mean[blockIdx.y*(width/window_size.x)+blockIdx.x].b = color_input.data[(smooth.y*width+smooth.x)*3]+2;
			sp_centers[blockIdx.y*(width/window_size.x)+blockIdx.x].x = input_points[smooth.y*width+smooth.x].x;
			sp_centers[blockIdx.y*(width/window_size.x)+blockIdx.x].y = input_points[smooth.y*width+smooth.x].y;
			sp_centers[blockIdx.y*(width/window_size.x)+blockIdx.x].z = input_points[smooth.y*width+smooth.x].z;
			sp_normals[blockIdx.y*(width/window_size.x)+blockIdx.x].x = normals[smooth.y*width+smooth.x].x;
			sp_normals[blockIdx.y*(width/window_size.x)+blockIdx.x].y = normals[smooth.y*width+smooth.x].y;
			sp_normals[blockIdx.y*(width/window_size.x)+blockIdx.x].z = normals[smooth.y*width+smooth.x].z;
		}

}

template<int blockSize>
__global__ void calculateLD_NASP(
	cv::gpu::GpuMat color_input,
	float3*		input_points,
	float3*		normals,
	SuperpixelSegmentation::label_distance* pixel_ld,
	SuperpixelSegmentation::superpixel* mean,
	float3*		sp_centers,
	float3*		sp_normals,
	int* labels,
	int2 window_size,
	int width,
	int height,
	float color_sigma,
	float spatial_sigma,
	float depth_sigma,
	float normal_sigma,
	int2 cluster_num,
	cv::gpu::GpuMat intr){
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
				//calculate spatial distance
				float spatial_distance = sqrtf(pow((float)(x-mean[ref_cluster_id].x), 2) + pow((float)(y-mean[ref_cluster_id].y), 2)) * 
					pow((float)(window_size.x+window_size.y)/2.0f, 2);
				//calculate normal distance
				float normal_distance;
				//calculate depth adaptive sigma
				//float da_spatial_sigma = spatial_sigma;
				float da_normal_sigma = normal_sigma;
				float depth_distance = 0.0f;
				if(input_points[y*width+x].z > 50.0f &&  sp_centers[ref_cluster_id].z > 50.0f){
					//depth distance
					float diff = abs(input_points[y*width+x].z - sp_centers[ref_cluster_id].z );
					float focal = (intr.data[0]+intr.data[4])/2.0f;
					//spatial_distance += (diff/input_points[y*width+x].z)*focal*pow((float)(window_size.x+window_size.y)/2.0f, 2.0f);
					//depth_distance = (diff/input_points[y*width+x].z)*focal*pow((float)(window_size.x+window_size.y)/2.0f, 2.0f);
					depth_distance = diff;
					//spatial_distance += diff*pow((float)(window_size.x+window_size.y)/2.0f, 2.0f);
					//da_spatial_sigma *= focal/input_points[y*width+x].z;
					//normal distance
					if(normals[y*width+x].x != -1.0f || 
						normals[y*width+x].y != -1.0f || 
							normals[y*width+x].z != -1.0f){
							if(sp_normals[ref_cluster_id].x != -1.0f || 
								sp_normals[ref_cluster_id].y != -1.0f || 
									sp_normals[ref_cluster_id].z != -1.0f){
								float normal_diff = normals[y*width+x].x*sp_normals[ref_cluster_id].x +
															normals[y*width+x].y*normals[ref_cluster_id].y +
																normals[y*width+x].z*normals[ref_cluster_id].z;
								normal_diff = normal_diff < 0.0f ? 0.0f:normal_diff; 
								//normal_distance = 255.0f*(1.0f-normal_diff)*focal/input_points[y*width+x].z;
								normal_distance = pow(255.0f, 2.0f)*(1.0f-normal_diff);
								//da_normal_sigma *= focal/input_points[y*width+x].z;
							}
					}
				}
				//set current ld
				//float sum_sigma = spatial_sigma+color_sigma+normal_sigma;
				float sum_sigma = spatial_sigma+color_sigma+da_normal_sigma+depth_sigma;
				distance_shared[tid] = color_distance*pow(color_sigma/sum_sigma, 2.0f) + spatial_distance*pow(spatial_sigma/sum_sigma, 2.0f) + 
											depth_distance*pow(depth_sigma/sum_sigma, 2.0f) + normal_distance*pow(da_normal_sigma/sum_sigma, 2.0f);
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
			labels[y*width+x] = label_shared[0];
		}
		if(input_points[y*width+x].z < 50.0f && 
				(depth_sigma != 0.0f || normal_sigma != 0.0f)){
			pixel_ld[y*width+x].l = -1;
			pixel_ld[y*width+x].d = 0.0f;
			labels[y*width+x] = -1;
		}
}

template<int blockSize>
__global__ void analyzeClusters_NASP(
	cv::gpu::GpuMat color_input,
	float3*		input_points,
	float3*		normals,
	SuperpixelSegmentation::label_distance* pixel_ld,
	SuperpixelSegmentation::superpixel* mean,
	float3*		sp_centers,
	float3*		sp_normals,
	int2 window_size,
	int2 cluster_num,
	int width,
	int height,
	cv::gpu::GpuMat intr){
		//4*13=32 Byte 16384/52 = 17*17 threads
		__shared__ int r_shared[blockSize];
		__shared__ int g_shared[blockSize];
		__shared__ int b_shared[blockSize];
		__shared__ int x_shared[blockSize];
		__shared__ int y_shared[blockSize];
		__shared__ int size_shared[blockSize];
		//3d data
		__shared__ float xw_shared[blockSize];
		__shared__ float yw_shared[blockSize];
		__shared__ float zw_shared[blockSize];
		__shared__ int num_of_points[blockSize];
		//normal data
		__shared__ float nx_shared[blockSize];
		__shared__ float ny_shared[blockSize];
		__shared__ float nz_shared[blockSize];
		
		//thread id
		int tid = threadIdx.y*blockDim.x+threadIdx.x;
		r_shared[tid] = 0;
		g_shared[tid] = 0;
		b_shared[tid] = 0;
		x_shared[tid] = 0;
		y_shared[tid] = 0;
		size_shared[tid] = 0;
		//3d data
		xw_shared[tid] = 0.0f;
		yw_shared[tid] = 0.0f;
		zw_shared[tid] = 0.0f;
		num_of_points[tid] = 0;
		//normal data
		nx_shared[tid] = 0.0f;
		ny_shared[tid] = 0.0f;
		nz_shared[tid] = 0.0f;
		//current cluster
		int2 cluster_pos;
		cluster_pos.x = blockIdx.x;
		cluster_pos.y = blockIdx.y;
		int cluster_id = cluster_pos.y*cluster_num.x+cluster_pos.x;
		//assign threads around cluster
		int2 arounds;
		int2 ref_pixels;
		ref_pixels.x = window_size.x*2/blockDim.x+1;
		ref_pixels.y = window_size.y*2/blockDim.y+1;
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
						r = r>255 ? 255:r;
						g = g>255 ? 255:g;
						b = b>255 ? 255:b;
						r = r<0 ? 0:r;
						g = g<0 ? 0:g;
						b = b<0 ? 0:b;
						r_shared[tid] += r;
						g_shared[tid] += g;
						b_shared[tid] += b;
						x_shared[tid] += arounds.x;
						y_shared[tid] += arounds.y;
						size_shared[tid] += 1;
						if(input_points[arounds.y*width+arounds.x].z>50.0f){
							if(normals[arounds.y*width+arounds.x].x != -1.0f ||
								normals[arounds.y*width+arounds.x].y != -1.0f ||
								 normals[arounds.y*width+arounds.x].z != -1.0f ){
									 //3d points
									xw_shared[tid] += input_points[arounds.y*width+arounds.x].x;
									yw_shared[tid] += input_points[arounds.y*width+arounds.x].y;
									zw_shared[tid] += input_points[arounds.y*width+arounds.x].z;
									//normals
									nx_shared[tid] += normals[arounds.y*width+arounds.x].x;
									ny_shared[tid] += normals[arounds.y*width+arounds.x].y;
									nz_shared[tid] += normals[arounds.y*width+arounds.x].z;
									num_of_points[tid] += 1;
							}
						}
					}
				}
			}
		}
		__syncthreads();
		
		//calculate average
		if(blockSize >= 1024){
			if(tid < 512){
				r_shared[tid] += r_shared[tid+512];
				g_shared[tid] += g_shared[tid+512];
				b_shared[tid] += b_shared[tid+512];
				x_shared[tid] += x_shared[tid+512];
				y_shared[tid] += y_shared[tid+512];
				xw_shared[tid] += xw_shared[tid+512];
				yw_shared[tid] += yw_shared[tid+512];
				zw_shared[tid] += zw_shared[tid+512];
				nx_shared[tid] += nx_shared[tid+512];
				ny_shared[tid] += ny_shared[tid+512];
				nz_shared[tid] += nz_shared[tid+512];
				size_shared[tid] += size_shared[tid+512];
				num_of_points[tid] += num_of_points[tid+512];
				__syncthreads();
			}
		}
		if(blockSize >= 512){
			if(tid < 256){
				r_shared[tid] += r_shared[tid+256];
				g_shared[tid] += g_shared[tid+256];
				b_shared[tid] += b_shared[tid+256];
				x_shared[tid] += x_shared[tid+256];
				y_shared[tid] += y_shared[tid+256];
				xw_shared[tid] += xw_shared[tid+256];
				yw_shared[tid] += yw_shared[tid+256];
				zw_shared[tid] += zw_shared[tid+256];
				nx_shared[tid] += nx_shared[tid+256];
				ny_shared[tid] += ny_shared[tid+256];
				nz_shared[tid] += nz_shared[tid+256];
				size_shared[tid] += size_shared[tid+256];
				num_of_points[tid] += num_of_points[tid+256];
				__syncthreads();
			}
		}
		if(blockSize >= 256){
			if(tid < 128){
				r_shared[tid] += r_shared[tid+128];
				g_shared[tid] += g_shared[tid+128];
				b_shared[tid] += b_shared[tid+128];
				x_shared[tid] += x_shared[tid+128];
				y_shared[tid] += y_shared[tid+128];
				xw_shared[tid] += xw_shared[tid+128];
				yw_shared[tid] += yw_shared[tid+128];
				zw_shared[tid] += zw_shared[tid+128];
				nx_shared[tid] += nx_shared[tid+128];
				ny_shared[tid] += ny_shared[tid+128];
				nz_shared[tid] += nz_shared[tid+128];
				size_shared[tid] += size_shared[tid+128];
				num_of_points[tid] += num_of_points[tid+128];
				__syncthreads();
			}
		}
		if(blockSize >= 128){
			if(tid < 64){
				r_shared[tid] += r_shared[tid+64];
				g_shared[tid] += g_shared[tid+64];
				b_shared[tid] += b_shared[tid+64];
				x_shared[tid] += x_shared[tid+64];
				y_shared[tid] += y_shared[tid+64];
				xw_shared[tid] += xw_shared[tid+64];
				yw_shared[tid] += yw_shared[tid+64];
				zw_shared[tid] += zw_shared[tid+64];
				nx_shared[tid] += nx_shared[tid+64];
				ny_shared[tid] += ny_shared[tid+64];
				nz_shared[tid] += nz_shared[tid+64];
				size_shared[tid] += size_shared[tid+64];
				num_of_points[tid] += num_of_points[tid+64];
				__syncthreads();
			}
		}
		if(tid < 32){
			if(blockSize >= 64){
				r_shared[tid] += r_shared[tid+32];
				g_shared[tid] += g_shared[tid+32];
				b_shared[tid] += b_shared[tid+32];
				x_shared[tid] += x_shared[tid+32];
				y_shared[tid] += y_shared[tid+32];
				xw_shared[tid] += xw_shared[tid+32];
				yw_shared[tid] += yw_shared[tid+32];
				zw_shared[tid] += zw_shared[tid+32];
				nx_shared[tid] += nx_shared[tid+32];
				ny_shared[tid] += ny_shared[tid+32];
				nz_shared[tid] += nz_shared[tid+32];
				size_shared[tid] += size_shared[tid+32];
				num_of_points[tid] += num_of_points[tid+32];
			}
			if(blockSize >= 32){
				r_shared[tid] += r_shared[tid+16];
				g_shared[tid] += g_shared[tid+16];
				b_shared[tid] += b_shared[tid+16];
				x_shared[tid] += x_shared[tid+16];
				y_shared[tid] += y_shared[tid+16];
				xw_shared[tid] += xw_shared[tid+16];
				yw_shared[tid] += yw_shared[tid+16];
				zw_shared[tid] += zw_shared[tid+16];
				nx_shared[tid] += nx_shared[tid+16];
				ny_shared[tid] += ny_shared[tid+16];
				nz_shared[tid] += nz_shared[tid+16];
				size_shared[tid] += size_shared[tid+16];
				num_of_points[tid] += num_of_points[tid+16];
			}
			if(blockSize >= 16){
				r_shared[tid] += r_shared[tid+8];
				g_shared[tid] += g_shared[tid+8];
				b_shared[tid] += b_shared[tid+8];
				x_shared[tid] += x_shared[tid+8];
				y_shared[tid] += y_shared[tid+8];
				xw_shared[tid] += xw_shared[tid+8];
				yw_shared[tid] += yw_shared[tid+8];
				zw_shared[tid] += zw_shared[tid+8];
				nx_shared[tid] += nx_shared[tid+8];
				ny_shared[tid] += ny_shared[tid+8];
				nz_shared[tid] += nz_shared[tid+8];
				size_shared[tid] += size_shared[tid+8];
				num_of_points[tid] += num_of_points[tid+8];
			}
			if(blockSize >= 8){
				r_shared[tid] += r_shared[tid+4];
				g_shared[tid] += g_shared[tid+4];
				b_shared[tid] += b_shared[tid+4];
				x_shared[tid] += x_shared[tid+4];
				y_shared[tid] += y_shared[tid+4];
				xw_shared[tid] += xw_shared[tid+4];
				yw_shared[tid] += yw_shared[tid+4];
				zw_shared[tid] += zw_shared[tid+4];
				nx_shared[tid] += nx_shared[tid+4];
				ny_shared[tid] += ny_shared[tid+4];
				nz_shared[tid] += nz_shared[tid+4];
				size_shared[tid] += size_shared[tid+4];
				num_of_points[tid] += num_of_points[tid+4];
			}
			if(blockSize >= 4){
				r_shared[tid] += r_shared[tid+2];
				g_shared[tid] += g_shared[tid+2];
				b_shared[tid] += b_shared[tid+2];
				x_shared[tid] += x_shared[tid+2];
				y_shared[tid] += y_shared[tid+2];
				xw_shared[tid] += xw_shared[tid+2];
				yw_shared[tid] += yw_shared[tid+2];
				zw_shared[tid] += zw_shared[tid+2];
				nx_shared[tid] += nx_shared[tid+2];
				ny_shared[tid] += ny_shared[tid+2];
				nz_shared[tid] += nz_shared[tid+2];
				size_shared[tid] += size_shared[tid+2];
				num_of_points[tid] += num_of_points[tid+2];
			}
			if(blockSize >= 2){
				r_shared[tid] += r_shared[tid+1];
				g_shared[tid] += g_shared[tid+1];
				b_shared[tid] += b_shared[tid+1];
				x_shared[tid] += x_shared[tid+1];
				y_shared[tid] += y_shared[tid+1];
				xw_shared[tid] += xw_shared[tid+1];
				yw_shared[tid] += yw_shared[tid+1];
				zw_shared[tid] += zw_shared[tid+1];
				nx_shared[tid] += nx_shared[tid+1];
				ny_shared[tid] += ny_shared[tid+1];
				nz_shared[tid] += nz_shared[tid+1];
				size_shared[tid] += size_shared[tid+1];
				num_of_points[tid] += num_of_points[tid+1];
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
				int2 pixel;
				if(num_of_points[0] != 0){
					//centers
					sp_centers[cluster_id].x = xw_shared[0]/(float)num_of_points[0];
					sp_centers[cluster_id].y = yw_shared[0]/(float)num_of_points[0];
					sp_centers[cluster_id].z = zw_shared[0]/(float)num_of_points[0];
					//real to projective
					float2 norm;
					norm.x = sp_centers[cluster_id].x/sp_centers[cluster_id].z;
					norm.y = sp_centers[cluster_id].y/sp_centers[cluster_id].z;
					pixel.x = (int)(norm.x*intr.data[0] + intr.data[2]);
					pixel.y = (int)(intr.data[5] - norm.y*intr.data[4]);
					if(pixel.x<0 || pixel.x>=width || pixel.y<0 || pixel.y<=height){
						pixel.x = x_shared[0]/size_shared[0];
						pixel.y = y_shared[0]/size_shared[0];
					}
					//normals
					sp_normals[cluster_id].x = nx_shared[0]/(float)num_of_points[0];
					sp_normals[cluster_id].y = ny_shared[0]/(float)num_of_points[0];
					sp_normals[cluster_id].z = nz_shared[0]/(float)num_of_points[0];
				}
				else{
					pixel.x = x_shared[0]/size_shared[0];
					pixel.y = y_shared[0]/size_shared[0];
					sp_normals[cluster_id].x = -1.0f;
					sp_normals[cluster_id].y = -1.0f;
					sp_normals[cluster_id].z = -1.0f;
					sp_centers[cluster_id].x = 0.0f;
					sp_centers[cluster_id].y = 0.0f;
					sp_centers[cluster_id].z = 0.0f;
				}
				//pixel.x = (pixel.x<0) ? 0:pixel.x;
				//pixel.x = (pixel.x>=width) ? width:pixel.x;
				//pixel.y = (pixel.y<0) ? 0:pixel.y;
				//pixel.y = (pixel.y>=height) ? height:pixel.y;
				//pixel.x = (int)(x_shared[0]/size_shared[0]);
				//pixel.y = (int)(y_shared[0]/size_shared[0]);	
				mean[cluster_id].x = pixel.x;
				mean[cluster_id].y = pixel.y;
				mean[cluster_id].r = (unsigned char)(r);
				mean[cluster_id].g = (unsigned char)(g);
				mean[cluster_id].b = (unsigned char)(b);
				mean[cluster_id].size = size_shared[0];
			}
		}

}

template<int blockSize>
__global__ void calculateWeightedAverage(
	cv::gpu::GpuMat color_input,
	float3*		input_points,
	float3*		normals,
	SuperpixelSegmentation::label_distance* pixel_ld,
	SuperpixelSegmentation::superpixel* mean,
	float3*		sp_centers,
	float3*		sp_normals,
	float*		normals_variance,
	int2 window_size,
	int2 cluster_num,
	int width,
	int height,
	float color_sigma,
	float spatial_sigma,
	float normal_sigma,
	cv::gpu::GpuMat intr){
		//4*13=52 Byte 16384/52 = 17*17 threads
		__shared__ float r_shared[blockSize];
		__shared__ float g_shared[blockSize];
		__shared__ float b_shared[blockSize];
		__shared__ float x_shared[blockSize];
		__shared__ float y_shared[blockSize];
		__shared__ float size_shared[blockSize];
		//3d data
		__shared__ float xw_shared[blockSize];
		__shared__ float yw_shared[blockSize];
		__shared__ float zw_shared[blockSize];
		//__shared__ float num_of_points[blockSize];
		__shared__ int num_of_points[blockSize];

		//normal data
		__shared__ float nx_shared[blockSize];
		__shared__ float ny_shared[blockSize];
		__shared__ float nz_shared[blockSize];
		//normal variance
		__shared__ float variance_shared[blockSize];

		//thread id
		int tid = threadIdx.y*blockDim.x+threadIdx.x;
		r_shared[tid] = 0.0f;
		g_shared[tid] = 0.0f;
		b_shared[tid] = 0.0f;
		x_shared[tid] = 0.0f;
		y_shared[tid] = 0.0f;
		size_shared[tid] = 0.0f;
		//3d data
		xw_shared[tid] = 0.0f;
		yw_shared[tid] = 0.0f;
		zw_shared[tid] = 0.0f;
		//num_of_points[tid] = 0.0f;
		num_of_points[tid] = 0;

		//normal data
		nx_shared[tid] = 0.0f;
		ny_shared[tid] = 0.0f;
		nz_shared[tid] = 0.0f;
		//normal variance
		variance_shared[tid] = 0.0f;

		//current cluster
		int2 cluster_pos;
		cluster_pos.x = blockIdx.x;
		cluster_pos.y = blockIdx.y;
		int cluster_id = cluster_pos.y*cluster_num.x+cluster_pos.x;
		//assign threads around cluster
		int2 arounds;
		int2 ref_pixels;
		ref_pixels.x = window_size.x*2/blockDim.x+1;
		ref_pixels.y = window_size.y*2/blockDim.y+1;
		for(int yy=0; yy<ref_pixels.y; yy++){
			for(int xx=0; xx<ref_pixels.x; xx++){
				arounds.x = mean[cluster_id].x+(threadIdx.x-blockDim.x/2)*ref_pixels.x+xx;
				arounds.y = mean[cluster_id].y+(threadIdx.y-blockDim.y/2)*ref_pixels.y+yy;
				if(arounds.x>=0 && arounds.x<width && arounds.y>=0 && arounds.y<height){
					int around_id = pixel_ld[arounds.y*width+arounds.x].l;
					if(around_id == cluster_id){
						//calculate color weight
						float color_diff = powf((float)(color_input.data[(arounds.y*width+arounds.x)*3+0])-(float)(mean[cluster_id].r), 2) +
												powf((float)(color_input.data[(arounds.y*width+arounds.x)*3+1])-(float)(mean[cluster_id].g), 2) +
													powf((float)(color_input.data[(arounds.y*width+arounds.x)*3+2])-(float)(mean[cluster_id].b), 2); 
						float color_filter = expf(-color_diff/(2*powf(color_sigma, 2.0f)));
						//calculate spatial filter
						float spatial_diff = powf((float)(arounds.x-mean[cluster_id].x), 2.0f)+powf((float)(arounds.y-mean[cluster_id].y), 2.0f);
						float spatial_filter = expf(-spatial_diff/(2*powf(spatial_sigma, 2.0f)));
						//weighted average
						float r = (float)color_input.data[(arounds.y*width+arounds.x)*3]*color_filter*spatial_filter;
						float g = (float)color_input.data[(arounds.y*width+arounds.x)*3+1]*color_filter*spatial_filter;
						float b = (float)color_input.data[(arounds.y*width+arounds.x)*3+2]*color_filter*spatial_filter;
						r = r>255.0f ? 255.0f:r;
						g = g>255.0f ? 255.0f:g;
						b = b>255.0f ? 255.0f:b;
						r = r<0.0f ? 0.0f:r;
						g = g<0.0f ? 0.0f:g;
						b = b<0.0f ? 0.0f:b;
						r_shared[tid] += r;
						g_shared[tid] += g;
						b_shared[tid] += b;
						x_shared[tid] += arounds.x*color_filter*spatial_filter;
						y_shared[tid] += arounds.y*color_filter*spatial_filter;
						size_shared[tid] += color_filter*spatial_filter;
						//if(input_points[arounds.y*width+arounds.x].z>50.0f){
						//	if(normals[arounds.y*width+arounds.x].x != -1.0f ||
						//		normals[arounds.y*width+arounds.x].y != -1.0f ||
						//		 normals[arounds.y*width+arounds.x].z != -1.0f ){
						//			//calculate spatial filter
						//			spatial_diff = powf((float)(input_points[arounds.y*width+arounds.x].x-sp_centers[cluster_id].x), 2.0f) +
						//							powf((float)(input_points[arounds.y*width+arounds.x].y-sp_centers[cluster_id].y), 2.0f) + 
						//							 powf((float)(input_points[arounds.y*width+arounds.x].z-sp_centers[cluster_id].z), 2.0f);
						//			spatial_filter = expf(-spatial_diff/(2*powf(spatial_sigma, 2.0f)));
						//			//calculate normal filter
						//			float normal_diff = normals[arounds.y*width+arounds.x].x*sp_normals[cluster_id].x + 
						//										normals[arounds.y*width+arounds.x].y*sp_normals[cluster_id].y + 
						//											normals[arounds.y*width+arounds.x].z*sp_normals[cluster_id].z;
						//			normal_diff = normal_diff < 0.0f ? 0.0f:normal_diff; 
						//			float normal_filter = normal_diff;
						//			//3d points
						//			xw_shared[tid] += input_points[arounds.y*width+arounds.x].x*spatial_filter*normal_filter;
						//			yw_shared[tid] += input_points[arounds.y*width+arounds.x].y*spatial_filter*normal_filter;
						//			zw_shared[tid] += input_points[arounds.y*width+arounds.x].z*spatial_filter*normal_filter;
						//			//normals
						//			nx_shared[tid] += normals[arounds.y*width+arounds.x].x*spatial_filter*normal_filter;
						//			ny_shared[tid] += normals[arounds.y*width+arounds.x].y*spatial_filter*normal_filter;
						//			nz_shared[tid] += normals[arounds.y*width+arounds.x].z*spatial_filter*normal_filter;
						//			num_of_points[tid] += spatial_filter*normal_filter;
						//	}
						//}
						if(input_points[arounds.y*width+arounds.x].z>50.0f){
							if(normals[arounds.y*width+arounds.x].x != -1.0f ||
								normals[arounds.y*width+arounds.x].y != -1.0f ||
								 normals[arounds.y*width+arounds.x].z != -1.0f ){
									//calculate spatial filter
									spatial_diff = powf((float)(input_points[arounds.y*width+arounds.x].x-sp_centers[cluster_id].x), 2.0f) +
													powf((float)(input_points[arounds.y*width+arounds.x].y-sp_centers[cluster_id].y), 2.0f) + 
													 powf((float)(input_points[arounds.y*width+arounds.x].z-sp_centers[cluster_id].z), 2.0f);
									//spatial_filter = expf(-spatial_diff/(2*powf(spatial_sigma, 2.0f)));
									//calculate normal filter
									float normal_diff = normals[arounds.y*width+arounds.x].x*sp_normals[cluster_id].x + 
																normals[arounds.y*width+arounds.x].y*sp_normals[cluster_id].y + 
																	normals[arounds.y*width+arounds.x].z*sp_normals[cluster_id].z;
									normal_diff = normal_diff < 0.0f ? 0.0f:normal_diff; 
									//float normal_filter = normal_diff;
									if(acos(normal_diff) < (3.141592653f / 3.0f)){ 
									//3d points
									xw_shared[tid] += input_points[arounds.y*width+arounds.x].x;
									yw_shared[tid] += input_points[arounds.y*width+arounds.x].y;
									zw_shared[tid] += input_points[arounds.y*width+arounds.x].z;
									//normals
									nx_shared[tid] += normals[arounds.y*width+arounds.x].x;
									ny_shared[tid] += normals[arounds.y*width+arounds.x].y;
									nz_shared[tid] += normals[arounds.y*width+arounds.x].z;
									variance_shared[tid] += normal_diff;
									num_of_points[tid] ++;
								}
							}
						}
					}
				}
			}
		}
		__syncthreads();
		
		//calculate average
		if(blockSize >= 1024){
			if(tid < 512){
				r_shared[tid] += r_shared[tid+512];
				g_shared[tid] += g_shared[tid+512];
				b_shared[tid] += b_shared[tid+512];
				x_shared[tid] += x_shared[tid+512];
				y_shared[tid] += y_shared[tid+512];
				xw_shared[tid] += xw_shared[tid+512];
				yw_shared[tid] += yw_shared[tid+512];
				zw_shared[tid] += zw_shared[tid+512];
				nx_shared[tid] += nx_shared[tid+512];
				ny_shared[tid] += ny_shared[tid+512];
				nz_shared[tid] += nz_shared[tid+512];
				size_shared[tid] += size_shared[tid+512];
				num_of_points[tid] += num_of_points[tid+512];
				variance_shared[tid] += variance_shared[tid+512];
				__syncthreads();
			}
		}
		if(blockSize >= 512){
			if(tid < 256){
				r_shared[tid] += r_shared[tid+256];
				g_shared[tid] += g_shared[tid+256];
				b_shared[tid] += b_shared[tid+256];
				x_shared[tid] += x_shared[tid+256];
				y_shared[tid] += y_shared[tid+256];
				xw_shared[tid] += xw_shared[tid+256];
				yw_shared[tid] += yw_shared[tid+256];
				zw_shared[tid] += zw_shared[tid+256];
				nx_shared[tid] += nx_shared[tid+256];
				ny_shared[tid] += ny_shared[tid+256];
				nz_shared[tid] += nz_shared[tid+256];
				size_shared[tid] += size_shared[tid+256];
				num_of_points[tid] += num_of_points[tid+256];
				variance_shared[tid] += variance_shared[tid+256];
				__syncthreads();
			}
		}
		if(blockSize >= 256){
			if(tid < 128){
				r_shared[tid] += r_shared[tid+128];
				g_shared[tid] += g_shared[tid+128];
				b_shared[tid] += b_shared[tid+128];
				x_shared[tid] += x_shared[tid+128];
				y_shared[tid] += y_shared[tid+128];
				xw_shared[tid] += xw_shared[tid+128];
				yw_shared[tid] += yw_shared[tid+128];
				zw_shared[tid] += zw_shared[tid+128];
				nx_shared[tid] += nx_shared[tid+128];
				ny_shared[tid] += ny_shared[tid+128];
				nz_shared[tid] += nz_shared[tid+128];
				size_shared[tid] += size_shared[tid+128];
				num_of_points[tid] += num_of_points[tid+128];
				variance_shared[tid] += variance_shared[tid+128];
				__syncthreads();
			}
		}
		if(blockSize >= 128){
			if(tid < 64){
				r_shared[tid] += r_shared[tid+64];
				g_shared[tid] += g_shared[tid+64];
				b_shared[tid] += b_shared[tid+64];
				x_shared[tid] += x_shared[tid+64];
				y_shared[tid] += y_shared[tid+64];
				xw_shared[tid] += xw_shared[tid+64];
				yw_shared[tid] += yw_shared[tid+64];
				zw_shared[tid] += zw_shared[tid+64];
				nx_shared[tid] += nx_shared[tid+64];
				ny_shared[tid] += ny_shared[tid+64];
				nz_shared[tid] += nz_shared[tid+64];
				size_shared[tid] += size_shared[tid+64];
				num_of_points[tid] += num_of_points[tid+64];
				variance_shared[tid] += variance_shared[tid+64];
				__syncthreads();
			}
		}
		if(tid < 32){
			if(blockSize >= 64){
				r_shared[tid] += r_shared[tid+32];
				g_shared[tid] += g_shared[tid+32];
				b_shared[tid] += b_shared[tid+32];
				x_shared[tid] += x_shared[tid+32];
				y_shared[tid] += y_shared[tid+32];
				xw_shared[tid] += xw_shared[tid+32];
				yw_shared[tid] += yw_shared[tid+32];
				zw_shared[tid] += zw_shared[tid+32];
				nx_shared[tid] += nx_shared[tid+32];
				ny_shared[tid] += ny_shared[tid+32];
				nz_shared[tid] += nz_shared[tid+32];
				size_shared[tid] += size_shared[tid+32];
				num_of_points[tid] += num_of_points[tid+32];
				variance_shared[tid] += variance_shared[tid+32];
			}
			if(blockSize >= 32){
				r_shared[tid] += r_shared[tid+16];
				g_shared[tid] += g_shared[tid+16];
				b_shared[tid] += b_shared[tid+16];
				x_shared[tid] += x_shared[tid+16];
				y_shared[tid] += y_shared[tid+16];
				xw_shared[tid] += xw_shared[tid+16];
				yw_shared[tid] += yw_shared[tid+16];
				zw_shared[tid] += zw_shared[tid+16];
				nx_shared[tid] += nx_shared[tid+16];
				ny_shared[tid] += ny_shared[tid+16];
				nz_shared[tid] += nz_shared[tid+16];
				size_shared[tid] += size_shared[tid+16];
				num_of_points[tid] += num_of_points[tid+16];
				variance_shared[tid] += variance_shared[tid+16];
			}
			if(blockSize >= 16){
				r_shared[tid] += r_shared[tid+8];
				g_shared[tid] += g_shared[tid+8];
				b_shared[tid] += b_shared[tid+8];
				x_shared[tid] += x_shared[tid+8];
				y_shared[tid] += y_shared[tid+8];
				xw_shared[tid] += xw_shared[tid+8];
				yw_shared[tid] += yw_shared[tid+8];
				zw_shared[tid] += zw_shared[tid+8];
				nx_shared[tid] += nx_shared[tid+8];
				ny_shared[tid] += ny_shared[tid+8];
				nz_shared[tid] += nz_shared[tid+8];
				size_shared[tid] += size_shared[tid+8];
				num_of_points[tid] += num_of_points[tid+8];
				variance_shared[tid] += variance_shared[tid+8];
			}
			if(blockSize >= 8){
				r_shared[tid] += r_shared[tid+4];
				g_shared[tid] += g_shared[tid+4];
				b_shared[tid] += b_shared[tid+4];
				x_shared[tid] += x_shared[tid+4];
				y_shared[tid] += y_shared[tid+4];
				xw_shared[tid] += xw_shared[tid+4];
				yw_shared[tid] += yw_shared[tid+4];
				zw_shared[tid] += zw_shared[tid+4];
				nx_shared[tid] += nx_shared[tid+4];
				ny_shared[tid] += ny_shared[tid+4];
				nz_shared[tid] += nz_shared[tid+4];
				size_shared[tid] += size_shared[tid+4];
				num_of_points[tid] += num_of_points[tid+4];
				variance_shared[tid] += variance_shared[tid+4];
			}
			if(blockSize >= 4){
				r_shared[tid] += r_shared[tid+2];
				g_shared[tid] += g_shared[tid+2];
				b_shared[tid] += b_shared[tid+2];
				x_shared[tid] += x_shared[tid+2];
				y_shared[tid] += y_shared[tid+2];
				xw_shared[tid] += xw_shared[tid+2];
				yw_shared[tid] += yw_shared[tid+2];
				zw_shared[tid] += zw_shared[tid+2];
				nx_shared[tid] += nx_shared[tid+2];
				ny_shared[tid] += ny_shared[tid+2];
				nz_shared[tid] += nz_shared[tid+2];
				size_shared[tid] += size_shared[tid+2];
				num_of_points[tid] += num_of_points[tid+2];
				variance_shared[tid] += variance_shared[tid+2];
			}
			if(blockSize >= 2){
				r_shared[tid] += r_shared[tid+1];
				g_shared[tid] += g_shared[tid+1];
				b_shared[tid] += b_shared[tid+1];
				x_shared[tid] += x_shared[tid+1];
				y_shared[tid] += y_shared[tid+1];
				xw_shared[tid] += xw_shared[tid+1];
				yw_shared[tid] += yw_shared[tid+1];
				zw_shared[tid] += zw_shared[tid+1];
				nx_shared[tid] += nx_shared[tid+1];
				ny_shared[tid] += ny_shared[tid+1];
				nz_shared[tid] += nz_shared[tid+1];
				size_shared[tid] += size_shared[tid+1];
				num_of_points[tid] += num_of_points[tid+1];
				variance_shared[tid] += variance_shared[tid+1];
			}
		}
		//store center point
		if(tid == 0){
			if(size_shared[0] != 0){
				int r = (int)(r_shared[0]/size_shared[0])>255 ? 255:(int)(r_shared[0]/size_shared[0]);
				int g = (int)(g_shared[0]/size_shared[0])>255 ? 255:(int)(g_shared[0]/size_shared[0]);
				int b = (int)(b_shared[0]/size_shared[0])>255 ? 255:(int)(b_shared[0]/size_shared[0]);
				r = r<0 ? 0:r;
				g = g<0 ? 0:g;
				b = b<0 ? 0:b;
				int2 pixel;
				if(num_of_points[0] != 0){
					//centers
					sp_centers[cluster_id].x = xw_shared[0]/(float)num_of_points[0];
					sp_centers[cluster_id].y = yw_shared[0]/(float)num_of_points[0];
					sp_centers[cluster_id].z = zw_shared[0]/(float)num_of_points[0];
					//real to projective
					float2 norm;
					norm.x = sp_centers[cluster_id].x/sp_centers[cluster_id].z;
					norm.y = sp_centers[cluster_id].y/sp_centers[cluster_id].z;
					pixel.x = (int)(norm.x*intr.data[0] + intr.data[2]);
					pixel.y = (int)(intr.data[5] - norm.y*intr.data[4]);
					if(pixel.x<0 || pixel.x>=width || pixel.y<0 || pixel.y<=height){
						pixel.x = x_shared[0]/size_shared[0];
						pixel.y = y_shared[0]/size_shared[0];
					}
					//normals
					sp_normals[cluster_id].x = nx_shared[0]/(float)num_of_points[0];
					sp_normals[cluster_id].y = ny_shared[0]/(float)num_of_points[0];
					sp_normals[cluster_id].z = nz_shared[0]/(float)num_of_points[0];
					float n = sqrtf(sp_normals[cluster_id].x*sp_normals[cluster_id].x + 
										sp_normals[cluster_id].y*sp_normals[cluster_id].y +
											sp_normals[cluster_id].z*sp_normals[cluster_id].z);
					sp_normals[cluster_id].x /= n;
					sp_normals[cluster_id].y /= n;
					sp_normals[cluster_id].z /= n;
					//variance
					normals_variance[cluster_id] = variance_shared[0]/(float)num_of_points[0];
				}
				else{
					pixel.x = (int)(x_shared[0]/size_shared[0]);
					pixel.y = (int)(y_shared[0]/size_shared[0]);
					sp_normals[cluster_id].x = -1.0f;
					sp_normals[cluster_id].y = -1.0f;
					sp_normals[cluster_id].z = -1.0f;
					sp_centers[cluster_id].x = 0.0f;
					sp_centers[cluster_id].y = 0.0f;
					sp_centers[cluster_id].z = 0.0f;
					normals_variance[cluster_id] = 0.0f;
				}	
				//pixel.x = (pixel.x<0) ? 0:pixel.x;
				//pixel.x = (pixel.x>=width) ? width:pixel.x;
				//pixel.y = (pixel.y<0) ? 0:pixel.y;
				//pixel.y = (pixel.y>=height) ? height:pixel.y;
				//pixel.x = (int)(x_shared[0]/size_shared[0]);
				//pixel.y = (int)(y_shared[0]/size_shared[0]);	
				mean[cluster_id].x = pixel.x;
				mean[cluster_id].y = pixel.y;
				mean[cluster_id].r = (unsigned char)(r);
				mean[cluster_id].g = (unsigned char)(g);
				mean[cluster_id].b = (unsigned char)(b);
				mean[cluster_id].size = size_shared[0];
			}
			//else{
			//	sp_normals[cluster_id].x = -1.0f;
			//	sp_normals[cluster_id].y = -1.0f;
			//	sp_normals[cluster_id].z = -1.0f;
			//}
		}

}
void NormalAdaptiveSuperpixel::Segmentation(cv::gpu::GpuMat color_image, float3* points3d_device, float3* normals_device,
												float color_sigma, float spatial_sigma, float depth_sigma, float normal_sigma, int iteration){
		//init label distance
		initLD_NASP<<<dim3(width/32, height/32), dim3(32, 32)>>>
			(LD_Device, width, height, ClusterNum, Window_Size);
		//sample clusters, move centers
		sampleInitialClusters_NASP<8*8><<<dim3(ClusterNum.x, ClusterNum.y), dim3(8, 8)>>>
			(color_image, points3d_device, normals_device, meanData_Device, superpixelCenters_Device, superpixelNormals_Device, width, height, Window_Size);
		for(int i = 0; i < iteration; i++){
			//Set cluster IDs	
			calculateLD_NASP<8*8><<<dim3(width, height), dim3(8, 8)>>>
				(color_image, points3d_device, normals_device, LD_Device, meanData_Device, superpixelCenters_Device, superpixelNormals_Device,
						Labels_Device, Window_Size, width, height, color_sigma, spatial_sigma, depth_sigma, normal_sigma, ClusterNum, Intrinsic_Device);
		
			//calculate average value of superpixel
			analyzeClusters_NASP<16*16><<<dim3(ClusterNum.x, ClusterNum.y), dim3(16, 16)>>>
				(color_image, points3d_device, normals_device, LD_Device, meanData_Device, superpixelCenters_Device, superpixelNormals_Device,
					Window_Size, ClusterNum, width, height, Intrinsic_Device);
		
			//	cudaMemcpy(superpixelCenters_Host, superpixelCenters_Device, sizeof(float3)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
		//cudaMemcpy(superpixelNormals_Host, superpixelNormals_Device, sizeof(float3)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
		
		
		//for(int y=0; y<ClusterNum.y; y++){
		//	for(int x=0; x<ClusterNum.x; x++){
		//		std::cout <<"center: "<<"x "<<superpixelCenters_Host[y*ClusterNum.x+x].x <<" y"<<superpixelCenters_Host[y*ClusterNum.x+x].y <<" z"<<superpixelCenters_Host[y*ClusterNum.x+x].z<<std::endl;
		//			std::cout <<"normal: "<<"x "<<superpixelNormals_Host[y*ClusterNum.x+x].x <<" y"<<superpixelNormals_Host[y*ClusterNum.x+x].y << " z"<<superpixelNormals_Host[y*ClusterNum.x+x].z<<std::endl;
		//	}
		//}
		
		///calculate weighted average of superpixel
		calculateWeightedAverage<16*16><<<dim3(ClusterNum.x, ClusterNum.y), dim3(16, 16)>>>
				(color_image, points3d_device, normals_device, LD_Device, meanData_Device, superpixelCenters_Device, superpixelNormals_Device, NormalsVariance_Device,
					Window_Size, ClusterNum, width, height, color_sigma, spatial_sigma, normal_sigma, Intrinsic_Device);
		}
		cudaMemcpy(Labels_Host, Labels_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
		
		cudaMemcpy(meanData_Host, meanData_Device, sizeof(superpixel)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
		cudaMemcpy(superpixelCenters_Host, superpixelCenters_Device, sizeof(float3)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
		cudaMemcpy(superpixelNormals_Host, superpixelNormals_Device, sizeof(float3)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
		
		
		//for(int y=0; y<ClusterNum.y; y++){
		//	for(int x=0; x<ClusterNum.x; x++){
		//		std::cout <<"center: "<<"x "<<superpixelCenters_Host[y*ClusterNum.x+x].x <<" y"<<superpixelCenters_Host[y*ClusterNum.x+x].y <<" z"<<superpixelCenters_Host[y*ClusterNum.x+x].z<<std::endl;
		//			std::cout <<"normal: "<<"x "<<superpixelNormals_Host[y*ClusterNum.x+x].x <<" y"<<superpixelNormals_Host[y*ClusterNum.x+x].y << " z"<<superpixelNormals_Host[y*ClusterNum.x+x].z<<std::endl;
		//	}
		//}
}
