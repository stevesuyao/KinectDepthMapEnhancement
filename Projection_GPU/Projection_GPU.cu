#include "Projection_GPU.h"

__global__ void initTemp(float3* temp, int cx, int cy, float fx, float fy, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	temp[x + y * width].x = x;
	temp[x + y * width].y = y;
	temp[x + y * width].z = 1.0f;
	//projection
	temp[x + y * width].y = (float)cy - temp[x + y * width].y;			
	temp[x + y * width].x = temp[x + y * width].x - (float)cx;
	
	temp[x + y * width].x /= fx;
	temp[x + y * width].y /= fy;

	temp[x + y * width].x *= temp[x + y * width].z;
	temp[x + y * width].y *= temp[x + y * width].z;

}

__global__ void setPsuedoDepth(
	const float3* input_3d, 
	float3* plane_fitted, 
	float3* normalized, 
	const float4* nd, 
	const int* labels, 
	const float* variance, 
	int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	int l = labels[x + y * width];
	//float a = nd[l].x;
	//float b = nd[l].y;
	//float c = nd[l].z;
	//float d = nd[l].w;
	if(l > -1 && acos(variance[l]) < (3.141592653f / 8.0f)){
		float a = nd[y*width+x].x;
		float b = nd[y*width+x].y;
		float c = nd[y*width+x].z;
		float d = nd[y*width+x].w;

		float3* ref = &plane_fitted[x + y * width];
		ref->z = abs(d / (a * normalized[x + y * width].x + b * normalized[x + y * width].y + c));
		ref->x = ref->z*normalized[x + y * width].x;
		ref->y = ref->z*normalized[x + y * width].y;
	}
	else{
		plane_fitted[x + y * width].x = input_3d[y*width+x].x;
		plane_fitted[x + y * width].y = input_3d[y*width+x].y;
		plane_fitted[x + y * width].z = input_3d[y*width+x].z;
	}
}
__global__ void setPsuedoDepth(
	const float3* input_3d, 
	float3* plane_fitted, 
	float3* normalized, 
	const float4* nd, 
	const int* labels,
	int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int l = labels[x + y * width];
	float a = nd[l].x;
	float b = nd[l].y;
	float c = nd[l].z;
	float d = nd[l].w;
	if(l > -1 && abs((float)nd[l].x)<1.0f){
		float3* ref = &plane_fitted[x + y * width];
		ref->z = abs(d / (a * normalized[x + y * width].x + b * normalized[x + y * width].y + c));
		ref->x = ref->z*normalized[x + y * width].x;
		ref->y = ref->z*normalized[x + y * width].y;
	}
	else{
		plane_fitted[x + y * width].x = input_3d[y*width+x].x;
		plane_fitted[x + y * width].y = input_3d[y*width+x].y;
		plane_fitted[x + y * width].z = input_3d[y*width+x].z;
	}
}
__global__ void setPsuedoDepth(
	const float3* input_3d, 
	float3* plane_fitted, 
	float3* normalized, 
	const float3* normals, 
	const float3* centers, 
	const int* labels, 
	const float* variance, 
	int width, int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;


		int l = labels[x + y * width];
		float a = normals[l].x;
		float b = normals[l].y;
		float c = normals[l].z;
		float d = fabs(a*centers[l].x+b*centers[l].y+c*centers[l].z);
		if(l > -1){
			//float a = nd[y*width+x].x;
			//float b = nd[y*width+x].y;
			//float c = nd[y*width+x].z;
			//float d = nd[y*width+x].w;
			if(acos(variance[l]) <  (3.141592653f / 8.0f)){
				float3* ref = &plane_fitted[x + y * width];
				ref->z = abs(d / (a * normalized[x + y * width].x + b * normalized[x + y * width].y + c));
				ref->x = ref->z*normalized[x + y * width].x;
				ref->y = ref->z*normalized[x + y * width].y;
			}
			else{
				plane_fitted[x + y * width].x = input_3d[y*width+x].x;
				plane_fitted[x + y * width].y = input_3d[y*width+x].y;
				plane_fitted[x + y * width].z = input_3d[y*width+x].z;
			}
		}
		else{
			plane_fitted[x + y * width].x = input_3d[y*width+x].x;
			plane_fitted[x + y * width].y = input_3d[y*width+x].y;
			plane_fitted[x + y * width].z = input_3d[y*width+x].z;
		}
}

void Projection_GPU::initNormalized3D(){
	//initialize
	initTemp<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(Normalized3D_Device, Cx, Cy, Fx, Fy, width, height);
}

__device__ void _atomicMin(double* address, double* val){
	double old = *address, assumed;
	do{
		assumed = old;
		old = 
			__longlong_as_double(
			atomicCAS(
			(unsigned long long int*)address, 
			__double_as_longlong(assumed), 
			__double_as_longlong(
			(*((float*)val) > *((float*)&assumed)) ? assumed : *val				
			)
			)
			);

	}while(assumed != old);
}

__global__ void mrf_optimization(
	float3* optimized3d,
	float3* planefitted3d,
	float3* normalized_3d,
	const int* labels,
	int width,
	int height,
	int window_size,
	float K,
	float smooth_sigma){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(planefitted3d[y*width+x].z > 50.0f && fabs(optimized3d[y*width+x].z-planefitted3d[y*width+x].z) < optimized3d[y*width+x].z*0.01f){
		//mrf optimization
		float numerator = planefitted3d[y*width+x].z, denominator = 1.0f;
		for(int i = - window_size/2; i <= window_size/2; i++){		// y
			for(int j = -window_size/2; j <= window_size/2; j++){		// x
				int xj = x+j, yi = y+i;
				if(xj >= 0 && xj < width && yi >= 0 && yi < height && optimized3d[yi*width+xj].z > 50.0f ){
					//float distance = sqrt(pow(planefitted3d[y*width+x].x-input3d[y*width+x].x, 2) +
					//						pow(planefitted3d[y*width+x].y-input3d[y*width+x].y, 2) +
					//							pow(planefitted3d[y*width+x].z-input3d[y*width+x].z, 2));
					float diff = fabs(optimized3d[y*width+x].z-optimized3d[yi*width+xj].z);
					float depth_filter = K/(1+pow(diff, 2.0f));
					//calculate filter
					float filter = smooth_sigma*depth_filter;
					numerator += optimized3d[yi*width+xj].z*filter; 
					denominator += filter;
				}
			}
		}
		if(denominator != 0.0f){
			float depth = numerator/denominator;	
			optimized3d[y*width+x].z = depth;
			optimized3d[y*width+x].x = normalized_3d[y*width+x].x*depth;
			optimized3d[y*width+x].y = normalized_3d[y*width+x].y*depth;
			}
		}
}
__global__ void variance_optimization(
	float3* optimized3d,
	const float* variance, 
	float3* planefitted3d,
	float3* normalized_3d,
	const int* labels,
	int width,
	int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(planefitted3d[y*width+x].z > 50.0f){
				//if(fabs((float)optimized3d[y*width+x].z-(float)planefitted3d[y*width+x].z)>optimized3d[y*width+x].z*0.1f){
				//		optimized3d[y*width+x].x = input3d[y*width+x].x;
				//		optimized3d[y*width+x].y = input3d[y*width+x].y;
				//		optimized3d[y*width+x].z = input3d[y*width+x].z;
				//}
				if(fabs((float)optimized3d[y*width+x].z-(float)planefitted3d[y*width+x].z)<optimized3d[y*width+x].z*0.01f &&
						labels[y*width+x] > -1 && (acos(variance[labels[y*width+x]]) < (3.141592653f / 8.0f))){
						optimized3d[y*width+x].z = planefitted3d[y*width+x].z*variance[labels[y*width+x]]+optimized3d[y*width+x].z*(1.0f-variance[labels[y*width+x]]);
						//planefitted3d[y*width+x].z = planefitted3d[y*width+x].z*(1.0f-variance[y*width+x])+input3d[y*width+x].z*variance[y*width+x];
						optimized3d[y*width+x].x = normalized_3d[y*width+x].x*optimized3d[y*width+x].z;
						optimized3d[y*width+x].y = normalized_3d[y*width+x].y*optimized3d[y*width+x].z;
				}
		
		}
}
void Projection_GPU::PlaneProjection(const float4* nd_device, const int* labels_device, const float* variance_device, const float3* points3d_device){
	//Ç∑Ç◊ÇƒÇÃì_ÇïΩñ fittingÇ∑ÇÈ
	//plane projection
	setPsuedoDepth<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, nd_device, labels_device, variance_device, width, height);

	//optimization
	cudaMemcpy(Optimized3D_Device, points3d_device, width * height * sizeof(float3), cudaMemcpyDeviceToDevice);
	//for(int i=0; i<20; i++){
	//mrf_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(Optimized3D_Device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height, 5, 0.5f, 1.0f);
	//}

	variance_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(Optimized3D_Device, variance_device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height);

	//3DÅ®2D
	//Device to Host
	cudaMemcpy(PlaneFitted3D_Host, points3d_device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(Optimized3D_Host, Optimized3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
}

void Projection_GPU::PlaneProjection(const float4* nd_device, const int* labels_device, const float3* points3d_device){
	//Ç∑Ç◊ÇƒÇÃì_ÇïΩñ fittingÇ∑ÇÈ
	//plane projection
	setPsuedoDepth<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, nd_device, labels_device, width, height);

	//optimization
	cudaMemcpy(Optimized3D_Device, points3d_device, width * height * sizeof(float3), cudaMemcpyDeviceToDevice);
	for(int i=0; i<20; i++){
	mrf_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(Optimized3D_Device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height, 5, 0.5f, 1.0f);
	}
	//cudaMemcpy(Optimized3D_Device, PlaneFitted3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToDevice);
	//variance_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(Optimized3D_Device, variance_device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height);

	//3DÅ®2D
	//Device to Host
	cudaMemcpy(PlaneFitted3D_Host, points3d_device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(Optimized3D_Host, Optimized3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
}


void Projection_GPU::PlaneProjection(
	const float3* normals_device, 
	const float3* centers_device, 
	const int* labels_device,
	const float* variance_device, 
	const float3* points3d_device){	
		//Ç∑Ç◊ÇƒÇÃì_ÇïΩñ fittingÇ∑ÇÈ
		//plane projection
		setPsuedoDepth<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, 
			normals_device, centers_device, labels_device, variance_device, width, height);

		//InputÇ∆ÇÃî‰är
		cudaMemcpy(Optimized3D_Device, points3d_device, width * height * sizeof(float3), cudaMemcpyDeviceToDevice);
		for(int i=0; i<20; i++){
		mrf_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
				(Optimized3D_Device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height, 5, 0.5f, 1.0f);
		}
		//3DÅ®2D
		//Device to Host
		cudaMemcpy(PlaneFitted3D_Host, PlaneFitted3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(Optimized3D_Host, Optimized3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
	
		//for(int y=0; y<height; y++){
		//	for(int x=0; x<width; x++){
		//		std::cout << PlaneFitted3D_Host[y*width+x].z <<std::endl;
		//	}
		//}
}