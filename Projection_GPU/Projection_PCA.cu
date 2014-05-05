#include "Projection_PCA.h"

__global__ void initTempPCA(float3* temp, int cx, int cy, float fx, float fy, int width, int height){
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

__global__ void setPsuedoDepthPCA(
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
	float a = nd[l].x;
	float b = nd[l].y;
	float c = nd[l].z;
	float d = nd[l].w;
	if(l > -1 && abs(a)<=1.0f){
		//float a = nd[y*width+x].x;
		//float b = nd[y*width+x].y;
		//float c = nd[y*width+x].z;
		//float d = nd[y*width+x].w;

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

void Projection_PCA::initNormalized3D(){
	//initialize
	initTempPCA<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(Normalized3D_Device, Cx, Cy, Fx, Fy, width, height);
}

__device__ void _atomicMinPCA(double* address, double* val){
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

__global__ void eigenvalues_optimizationPCA(
	float3* optimized3d,
	const float* eigenvalues, 
	float3* planefitted3d,
	float3* normalized_3d,
	const int* labels,
	int width,
	int height,
	float eigenvalue_sigma){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(planefitted3d[y*width+x].z > 50.0f){
				//if(fabs((float)optimized3d[y*width+x].z-(float)planefitted3d[y*width+x].z)>optimized3d[y*width+x].z*0.1f){
				//		optimized3d[y*width+x].x = input3d[y*width+x].x;
				//		optimized3d[y*width+x].y = input3d[y*width+x].y;
				//		optimized3d[y*width+x].z = input3d[y*width+x].z;
				//}
				if(fabs((float)optimized3d[y*width+x].z-(float)planefitted3d[y*width+x].z)<optimized3d[y*width+x].z*0.01f && labels[y*width+x] > -1 ){
						//optimized3d[y*width+x].z = planefitted3d[y*width+x].z*eigenvalues[y*width+x]+optimized3d[y*width+x].z*(1.0f-variance[labels[y*width+x]]);
						//planefitted3d[y*width+x].z = planefitted3d[y*width+x].z*(1.0f-variance[y*width+x])+input3d[y*width+x].z*variance[y*width+x];
						float weight = expf(-eigenvalue_sigma/(2*pow(eigenvalues[y*width+x], 2.0f)));
						optimized3d[y*width+x].z = weight*optimized3d[y*width+x].z+(1.0f-weight)*planefitted3d[y*width+x].z;
						optimized3d[y*width+x].x = normalized_3d[y*width+x].x*optimized3d[y*width+x].z;
						optimized3d[y*width+x].y = normalized_3d[y*width+x].y*optimized3d[y*width+x].z;
				}
		
		}
}
void Projection_PCA::PlaneProjection(const float4* nd_device, const int* labels_device, const float* eigenvalues_device, const float3* points3d_device){
	//Ç∑Ç◊ÇƒÇÃì_ÇïΩñ fittingÇ∑ÇÈ
	//plane projection
	setPsuedoDepthPCA<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, nd_device, labels_device, eigenvalues_device, width, height);

	//optimization
	cudaMemcpy(Optimized3D_Device, points3d_device, width * height * sizeof(float3), cudaMemcpyDeviceToDevice);

	//for(int i=0; i<20; i++){
	//mrf_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(Optimized3D_Device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height, 5, 0.5f, 1.0f);
	//}

	//eigenvalues_optimizationPCA<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(Optimized3D_Device, eigenvalues_device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height, 0.0f);
	//cudaMemcpy(Optimized3D_Device, PlaneFitted3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToDevice);

	//3DÅ®2D
	//Device to Host
	cudaMemcpy(PlaneFitted3D_Host, points3d_device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(Optimized3D_Host, Optimized3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
}
