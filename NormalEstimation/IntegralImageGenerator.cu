//////////////////////////////////////////////////
// contents :generate integral image from depthmap
// create 	:2013/03/17
// modefied :
// writer   :Takuya Ikeda 
// other	:GPU part
//////////////////////////////////////////////////


#include "IntegralImagegenerator.h"

//integral image scan
//unsigned type
__global__ void scan_rowsUI(unsigned* input, unsigned* output, int n){
	__shared__ unsigned temp[M_WIDTH];

	int bl = blockIdx.x * M_WIDTH;
	int tdx = threadIdx.x; int offset = 1;
	temp[2*tdx] = input[bl+2*tdx];
	temp[2*tdx+1] = input[bl+2*tdx+1];
	for(int d = n >> 1; d > 0; d >>= 1){
		__syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if(tdx == 0){
		temp[n - 1] = 0;
	}
	for(int d = 1; d < n; d *= 2){
		offset >>= 1; __syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			unsigned t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	output[bl+2*tdx] = temp[2*tdx]     + input[bl+2*tdx];
	output[bl+2*tdx+1] = temp[2*tdx+1] + input[bl+2*tdx+1];
}

__global__ void scan_colsUI(unsigned* input, unsigned* output, int n){
	__shared__ unsigned temp[M_HEIGHT];
	int y = threadIdx.x * 2 * M_WIDTH;
	int blx = blockIdx.x; int offset = 1;
	int tdx = threadIdx.x;
	temp[2*tdx] = input[y+blx];
	temp[2*tdx+1] = input[y+blx+M_WIDTH];
	for(int d = n >> 1; d > 0; d >>= 1){
		__syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if(tdx == 0){
		temp[n - 1] = 0;
	}
	for(int d = 1; d < n; d *= 2){
		offset >>= 1; __syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			unsigned t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[y+blx]		  = temp[2*tdx]   + input[y+blx];
	output[y+blx+M_WIDTH] = temp[2*tdx+1] + input[y+blx+M_WIDTH];
}

//copy normal to Max array
__global__ void copyNormalToMaxUI(const unsigned* d, unsigned* dM, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	dM[y*M_WIDTH+x] = d[y*width+x];
}
//copy Max to normal array
__global__ void copyMaxToNormaxlUI(const unsigned* dM, unsigned* d, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	d[y*width+x] = dM[y*M_WIDTH+x];
}
//Initialization
__global__ void InitializeUI(unsigned* ui, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	ui[y*width+x] = 0;
}

__global__ void checkValidVertex(const float3* vertex, unsigned* check, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	int acs = y*width+x;
	if(vertex[acs].z != 0.0)
		check[acs] = 1;
	else
		check[acs] = 0;
}

void IntegralImagegenerator::computeCount(void){
//compute valid cont integral image
	dim3 block(BLOCKDIM, BLOCKDIM), grid(M_HEIGHT/BLOCKDIM, M_WIDTH/BLOCKDIM);
	InitializeUI<<<grid, block>>>(dinMui, M_WIDTH, M_HEIGHT);
	InitializeUI<<<grid, block>>>(doutMui, M_WIDTH, M_HEIGHT);
	//count integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	//check valid vertex
	checkValidVertex<<<grid, block>>>(vertexMap, IntegralCount, width, height);
	//[d] copy to [dM]
	copyNormalToMaxUI<<<grid, block>>>(IntegralCount, dinMui, width, height);
	//scan rows
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsUI<<<grid, block>>>(dinMui, doutMui, M_WIDTH);
	//scan cols
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsUI<<<grid, block>>>(doutMui, dinMui, M_HEIGHT);
	//[dM] copy to [d]
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copyMaxToNormaxlUI<<<grid, block>>>(dinMui, IntegralCount, width, height);
}


//double type
__global__ void scan_rowsD(double* input, double* output, int n){
	__shared__ double temp[M_WIDTH];

	int bl = blockIdx.x * M_WIDTH;
	int tdx = threadIdx.x; int offset = 1;
	temp[2*tdx] = input[bl+2*tdx];
	temp[2*tdx+1] = input[bl+2*tdx+1];
	for(int d = n >> 1; d > 0; d >>= 1){
		__syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if(tdx == 0){
		temp[n - 1] = 0;
	}
	for(int d = 1; d < n; d *= 2){
		offset >>= 1; __syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			double t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	output[bl+2*tdx] = temp[2*tdx]     + input[bl+2*tdx];
	output[bl+2*tdx+1] = temp[2*tdx+1] + input[bl+2*tdx+1];
}

__global__ void scan_colsD(double* input, double* output, int n){
	__shared__ double temp[M_HEIGHT];
	int y = threadIdx.x * 2 * M_WIDTH;
	int blx = blockIdx.x; int offset = 1;
	int tdx = threadIdx.x;
	temp[2*tdx] = input[y+blx];
	temp[2*tdx+1] = input[y+blx+M_WIDTH];
	for(int d = n >> 1; d > 0; d >>= 1){
		__syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if(tdx == 0){
		temp[n - 1] = 0;
	}
	for(int d = 1; d < n; d *= 2){
		offset >>= 1; __syncthreads();
		if(tdx < d){
			int ai = offset*(2*tdx+1)-1;
			int bi = offset*(2*tdx+2)-1;
			double t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[y+blx]		  = temp[2*tdx]   + input[y+blx];
	output[y+blx+M_WIDTH] = temp[2*tdx+1] + input[y+blx+M_WIDTH];
}

//copy Normal to Max
__global__ void copy_X_NormalToMaxD(double3* d, double* dM, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	dM[y*M_WIDTH+x] = d[y*width+x].x;
}
__global__ void copy_X_NormalToMaxD(float3* d, double* dM, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	dM[y*M_WIDTH+x] = (double)d[y*width+x].x;
}
__global__ void copy_Y_NormalToMaxD(double3* d, double* dM, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	dM[y*M_WIDTH+x] = d[y*width+x].y;
}
__global__ void copy_Y_NormalToMaxD(float3* d, double* dM, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	dM[y*M_WIDTH+x] = (double)d[y*width+x].y;
}
__global__ void copy_Z_NormalToMaxD(double3* d, double* dM, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	dM[y*M_WIDTH+x] = d[y*width+x].z;
}

__global__ void copy_Z_NormalToMaxD(float3* d, double* dM, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	dM[y*M_WIDTH+x] = (double)d[y*width+x].z;
}


//copy Max to Normal
__global__ void copyMaxToNormalD(const double* dM, double* d, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	d[y*width+x] = dM[y*M_WIDTH+x];
}

__global__ void copy_X_MaxToNormalD(const double* dM, double3* d, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	d[y*width+x].x = dM[y*M_WIDTH+x];
}
__global__ void copy_Y_MaxToNormalD(const double* dM, double3* d, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	d[y*width+x].y = dM[y*M_WIDTH+x];
}
__global__ void copy_Z_MaxToNormalD(const double* dM, double3* d, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	d[y*width+x].z = dM[y*M_WIDTH+x];
}



__global__ void InitializeD(double* f, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	f[y*width+x] = 0.0;
}

void IntegralImagegenerator::computeZ(void){
	//SDC method
	dim3 block(BLOCKDIM, BLOCKDIM), grid(M_HEIGHT/BLOCKDIM, M_WIDTH/BLOCKDIM);
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	//z component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	//[d] copy to [dM]
	copy_Z_NormalToMaxD<<<grid, block>>>(vertexMap, dinMd, width, height);
	//scan rows
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	//scan cols
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	//[dM] copy to [d]
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copyMaxToNormalD<<<grid, block>>>(dinMd, IntegralZ, width, height);
}

void IntegralImagegenerator::computeXYZ(void){
	dim3 block(BLOCKDIM, BLOCKDIM), grid(M_HEIGHT/BLOCKDIM, M_WIDTH/BLOCKDIM);
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	//x component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_X_NormalToMaxD<<<grid, block>>>(vertexMap, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_X_MaxToNormalD<<<grid, block>>>(dinMd, IntegralXYZ, width, height);

	//y component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	copy_Y_NormalToMaxD<<<grid, block>>>(vertexMap, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_Y_MaxToNormalD<<<grid, block>>>(dinMd, IntegralXYZ, width, height);

	//z component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	copy_Z_NormalToMaxD<<<grid, block>>>(vertexMap, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_Z_MaxToNormalD<<<grid, block>>>(dinMd, IntegralXYZ, width, height);

}

__global__ void elementWideMult(float3* vertex, double3* xxxyxz, double3* yyyzzz, int width, int height){
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	int acs = y*width+x;
	xxxyxz[acs].x = (double)vertex[acs].x*(double)vertex[acs].x;
	xxxyxz[acs].y = (double)vertex[acs].x*(double)vertex[acs].y;
	xxxyxz[acs].z = (double)vertex[acs].x*(double)vertex[acs].z;
	yyyzzz[acs].x = (double)vertex[acs].y*(double)vertex[acs].y;
	yyyzzz[acs].y = (double)vertex[acs].y*(double)vertex[acs].z;
	yyyzzz[acs].z = (double)vertex[acs].z*(double)vertex[acs].z;
	return;
}

void IntegralImagegenerator::computeAllComponentMull(void){
	dim3 block(BLOCKDIM, BLOCKDIM), grid(width / BLOCKDIM, height / BLOCKDIM);
	elementWideMult<<<grid, block>>>(vertexMap, IntegralXXXYXZ, IntegralYYYZZZ, width, height);
}

void IntegralImagegenerator::computeXXXYXZ(void){
	dim3 block(BLOCKDIM, BLOCKDIM), grid(M_HEIGHT/BLOCKDIM, M_WIDTH/BLOCKDIM);
	//xx component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	copy_X_NormalToMaxD<<<grid, block>>>(IntegralXXXYXZ, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_X_MaxToNormalD<<<grid, block>>>(dinMd, IntegralXXXYXZ, width, height);

	//xy component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	copy_Y_NormalToMaxD<<<grid, block>>>(IntegralXXXYXZ, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_Y_MaxToNormalD<<<grid, block>>>(dinMd, IntegralXXXYXZ, width, height);

	//xz component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	copy_Z_NormalToMaxD<<<grid, block>>>(IntegralXXXYXZ, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_Z_MaxToNormalD<<<grid, block>>>(dinMd, IntegralXXXYXZ, width, height);
}

void IntegralImagegenerator::computeYYYZZZ(void){
	dim3 block(BLOCKDIM, BLOCKDIM), grid(M_HEIGHT/BLOCKDIM, M_WIDTH/BLOCKDIM);

	//yy component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	copy_X_NormalToMaxD<<<grid, block>>>(IntegralYYYZZZ, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_X_MaxToNormalD<<<grid, block>>>(dinMd, IntegralYYYZZZ, width, height);

	//yz component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	copy_Y_NormalToMaxD<<<grid, block>>>(IntegralYYYZZZ, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_Y_MaxToNormalD<<<grid, block>>>(dinMd, IntegralYYYZZZ, width, height);

	//zz component integral image
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	InitializeD<<<grid, block>>>(dinMd, M_WIDTH, M_HEIGHT);
	InitializeD<<<grid, block>>>(doutMd, M_WIDTH, M_HEIGHT);
	copy_Z_NormalToMaxD<<<grid, block>>>(IntegralYYYZZZ, dinMd, width, height);
	block.x = M_WIDTH/2, block.y = 1, grid.x = height, grid.y = 1;
	scan_rowsD<<<grid, block>>>(dinMd, doutMd, M_WIDTH);
	block.x = M_HEIGHT/2, block.y = 1, grid.x = width, grid.y = 1;
	scan_colsD<<<grid, block>>>(doutMd, dinMd, M_HEIGHT);
	block.x = BLOCKDIM, block.y = BLOCKDIM, grid.x = width/BLOCKDIM, grid.y = height/BLOCKDIM;
	copy_Z_MaxToNormalD<<<grid, block>>>(dinMd, IntegralYYYZZZ, width, height);
}


void IntegralImagegenerator::generateIntegralImage(int method){
	//compute valid cont integral image
	computeCount();

	if(method == SDC){ // 0 = SDC method
		//compute integral for z components
		computeZ();
	} else if(method == CM){ // 1 = CM method
		//CV method
		computeXYZ();
		computeAllComponentMull();
		computeXXXYXZ();
		computeYYYZZZ();
	}

	return;
}
