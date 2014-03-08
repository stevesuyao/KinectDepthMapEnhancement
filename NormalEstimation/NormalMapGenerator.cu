/////////////////////////////////////////////////////////////////////////////////
// contents :NormalMap generation from SmoothingAreaMap, IntegralImage, vertexMap
// create 	:2013/03/17
// modefied :
// writer   :Takuya Ikeda 
// other	:GPU part
/////////////////////////////////////////////////////////////////////////////////

#include "NormalMapGenerator.h"

__device__ unsigned getFiniteElementsCount(unsigned* integralCount, int u, int v, int uu, int vv, int width, int height){
	const unsigned upper_left_idx      = v * width + u;
	const unsigned upper_right_idx     = upper_left_idx + uu;
	const unsigned lower_left_idx      = (v + vv) * width + u;
	const unsigned lower_right_idx     = lower_left_idx + uu;
	return (integralCount[lower_right_idx] + integralCount[upper_left_idx] - 
		integralCount[lower_left_idx] - integralCount[upper_right_idx]);
}

__device__ double getSumFromIntegralImageD(double* integralDepth, int u, int v, int uu, int vv, int width, int height){
	const unsigned upper_left_idx      = v * width + u;
	const unsigned upper_right_idx     = upper_left_idx + uu;
	const unsigned lower_left_idx      = (v + vv) * width + u;
	const unsigned lower_right_idx     = lower_left_idx + uu;
	return (integralDepth[lower_right_idx] + integralDepth[upper_left_idx] - 
		integralDepth[lower_left_idx] - integralDepth[upper_right_idx]);
}

__global__ void computeNormalSDC_GPU(float3* normal, float3* vertice, double* integral, unsigned* integralCount, float* smoothingMap, int border, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int acs = y*width+x;
	//float bad_point = std::numeric_limits<float>::quiet_NaN ();
	float bad_point = -1.0;

	//set bad point by border
	if(x < border || x > width-border || y < border || y > height - border){
		normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
		return;
	}
	float smoothing = smoothingMap[acs];

	//use depth dependent smoothing			
	if(smoothing <= 2.0f){
		normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
		return;
	} else {
		//compute normal point
		int rect_width_ = smoothing;
		int rect_width_2_ = rect_width_ >> 1;
		int rect_width_4_ = rect_width_ >> 2;
		int rect_height_ = smoothing;
		int rect_height_2_ = rect_width_ >> 1;
		int rect_height_4_ = rect_width_ >> 2;

		int cont = getFiniteElementsCount(integralCount, x-rect_width_2_-1, y-rect_height_2_-1, rect_width_, rect_height_, width, height);
		if(cont == 0){
			normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
			return;
		}
		unsigned count_L_z = getFiniteElementsCount (integralCount, x - rect_width_2_-1, y - rect_height_4_-1, rect_width_2_, rect_height_2_, width, height);
		unsigned count_R_z = getFiniteElementsCount (integralCount, x                  , y - rect_height_4_-1, rect_width_2_, rect_height_2_, width, height);
		unsigned count_U_z = getFiniteElementsCount (integralCount, x - rect_width_4_-1, y - rect_height_2_-1, rect_width_2_, rect_height_2_, width, height);
		unsigned count_D_z = getFiniteElementsCount (integralCount, x - rect_width_4_-1, y                   , rect_width_2_, rect_height_2_, width, height);

		if (count_L_z == 0 || count_R_z == 0 || count_U_z == 0 || count_D_z == 0){
			normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
			return;
		}
		double mean_L_z = getSumFromIntegralImageD(integral, x - rect_width_2_-1, y - rect_height_4_-1, rect_width_2_, rect_height_2_, width, height) / count_L_z;
		double mean_R_z = getSumFromIntegralImageD(integral, x                  , y - rect_height_4_-1, rect_width_2_, rect_height_2_, width, height) / count_R_z;
		double mean_U_z = getSumFromIntegralImageD(integral, x - rect_width_4_-1, y - rect_height_2_-1, rect_width_2_, rect_height_2_, width, height) / count_U_z;
		double mean_D_z = getSumFromIntegralImageD(integral, x - rect_width_4_-1, y                   , rect_width_2_, rect_height_2_, width, height) / count_D_z;

		float3 pointL = vertice[acs - rect_width_4_ - 1];
		float3 pointR = vertice[acs + rect_width_4_ + 1];
		float3 pointU = vertice[acs - rect_height_4_ * width - 1];
		float3 pointD = vertice[acs + rect_height_4_ * width + 1];

		const double mean_x_z = mean_R_z - mean_L_z;
		const double mean_y_z = mean_D_z - mean_U_z;

		const double mean_x_x = (double)pointR.x - (double)pointL.x;
		const double mean_x_y = (double)pointR.y - (double)pointL.y;
		const double mean_y_x = (double)pointD.x - (double)pointU.x;
		const double mean_y_y = (double)pointD.y - (double)pointU.y;

		float normal_x =  (float)(mean_x_z * mean_y_y - mean_x_y * mean_y_z);
		float normal_y =  (float)(-(mean_x_x * mean_y_z - mean_x_z * mean_y_x));
		float normal_z =  (float)(mean_x_y * mean_y_x - mean_x_x * mean_y_y);

		const float normal_length = (normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);

		if (normal_length == 0.0f){
			normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
			return;
		}

		// See if we need to flip any plane normals
		float vp_x = -vertice[acs].x;
		float vp_y = -vertice[acs].y;
		float vp_z = -vertice[acs].z;

		// Dot product between the (viewpoint - point) and the plane normal
		float cos_theta = (vp_x * normal_x + vp_y * normal_y + vp_z * normal_z);

		// Flip the plane normal
		if (cos_theta <= 0 && normal[acs].x != bad_point && normal[acs].y != bad_point && normal[acs].z != bad_point){
			normal_x *= -1.0f;
			normal_y *= -1.0f;
			normal_z *= -1.0f;
		}

		const float scale = 1.0f / sqrt (normal_length);

		normal[acs].x = normal_x * scale;
		normal[acs].y = normal_y * scale;
		normal[acs].z = normal_z * scale;
	}
}


__device__ double3 getSumFromIntegralImageD3(double3* integral, int u, int v, int uu, int vv, int width, int height){
	const unsigned upper_left_idx      = v * width + u;
	const unsigned upper_right_idx     = upper_left_idx + uu;
	const unsigned lower_left_idx      = (v + vv) * width + u;
	const unsigned lower_right_idx     = lower_left_idx + uu;
	double3 tmp;
	tmp.x = integral[lower_right_idx].x + integral[upper_left_idx].x - integral[lower_left_idx].x - integral[upper_right_idx].x;
	tmp.y = integral[lower_right_idx].y + integral[upper_left_idx].y - integral[lower_left_idx].y - integral[upper_right_idx].y;
	tmp.z = integral[lower_right_idx].z + integral[upper_left_idx].z - integral[lower_left_idx].z - integral[upper_right_idx].z;
	return tmp;
}

__device__ void computeRoots2(const double b, const double c, double3* roots){
	roots->x = 0.0f;
	double d = (b*b-4.0f*c);
	if(d < 0.0)
		d = 0.0f;
	double sd = sqrt(d);
	roots->z = 0.5f*(b+sd);
	roots->y = 0.5f*(b-sd);
}

__device__ void computeRoots(double* mat, double3* roots){
	double c0 = mat[0]*mat[4]*mat[8] + 2.0f*mat[1]*mat[2]*mat[5] - mat[0]*mat[5]*mat[5] - 
		mat[4]*mat[2]*mat[2] - mat[8]*mat[1]*mat[1];
	double c1 = mat[0]*mat[4] - mat[1]*mat[1] + mat[0]*mat[8] - 
		mat[2]*mat[2] + mat[4]*mat[8] - mat[5]*mat[5];
	double c2 = mat[0] + mat[4] + mat[8];
	if(fabs(c0) < FLT_EPSILON){
		computeRoots2(c2, c1, roots);
	} else {
		const double s_inv3 = 1.0f/3.0f;
		const double s_sqrt3 = sqrt(3.0f);
		double c2_over_3 = c2*s_inv3;
		double a_over_3 = (c1 - c2*c2_over_3)*s_inv3;
		if(a_over_3 > 0.0)
			a_over_3 = 0.0f;
		double half_b = 0.5f * (c0 + c2_over_3*(2.0f*c2_over_3*c2_over_3 - c1));
		double q = half_b*half_b + a_over_3*a_over_3*a_over_3;
		if(q > 0.0)
			q = 0.0f;
		// compute the eigenvalues by solving for the roots of the polynomial.
		double rho = sqrt(-a_over_3);
		double theta = atan2(sqrt(-q), half_b)*s_inv3;
		double cos_theta = cos(theta);
		double sin_theta = sin(theta);
		roots->x =c2_over_3 + 2.0f*rho*cos_theta;
		roots->y =c2_over_3 - rho * (cos_theta + s_sqrt3*sin_theta);
		roots->z =c2_over_3 - rho * (cos_theta - s_sqrt3*sin_theta);
		if(roots->x >= roots->y){
			double tmp = roots->y;
			roots->y = roots->x;
			roots->x = tmp;
		}
		if(roots->y >= roots->z){
			double tmp = roots->z;
			roots->z = roots->y;
			roots->y = tmp;
			if(roots->x >= roots->y){
				double tmp = roots->y;
				roots->y = roots->x;
				roots->x = tmp;
			}
		}
		if(roots->x <= 0)
			computeRoots2(c2, c1, roots);
	}
}

__device__ void computeEigenValueAndVector(double* m, double* eigenValue, double3* eigenVector){
	double scaledMat[9];
	double scale = -100.0;
	for(int i = 0; i < 9; i++){
		double tmp = abs(m[i]);
		if(tmp > scale){
			scale = tmp;
		}
	}
	if(scale <= DBL_MIN)
		scale = 1.0;
	for(int i = 0; i < 9; i++){
		scaledMat[i] = m[i] / scale;
	}

	double3 eigenvalues;
	computeRoots(scaledMat, &eigenvalues);
	*eigenValue = eigenvalues.x * scale;
	scaledMat[0] -= eigenvalues.x, scaledMat[4] -= eigenvalues.x, scaledMat[8] -= eigenvalues.x;
	double3 vec1, vec2, vec3;

	//To recheck this cross product 
	vec1.x = scaledMat[1]*scaledMat[5] - scaledMat[2]*scaledMat[4];
	vec1.y = scaledMat[2]*scaledMat[3] - scaledMat[0]*scaledMat[5];
	vec1.z = scaledMat[0]*scaledMat[4] - scaledMat[1]*scaledMat[3];

	vec2.x = scaledMat[1]*scaledMat[8] - scaledMat[2]*scaledMat[7];
	vec2.y = scaledMat[2]*scaledMat[6] - scaledMat[0]*scaledMat[8];
	vec2.z = scaledMat[0]*scaledMat[7] - scaledMat[1]*scaledMat[6];

	vec3.x = scaledMat[4]*scaledMat[8] - scaledMat[5]*scaledMat[7];
	vec3.y = scaledMat[5]*scaledMat[6] - scaledMat[3]*scaledMat[8];
	vec3.z = scaledMat[3]*scaledMat[7] - scaledMat[4]*scaledMat[6];
	double len1 = sqrt(vec1.x*vec1.x + vec1.y*vec1.y + vec1.z*vec1.z);
	double len2 = sqrt(vec2.x*vec2.x + vec2.y*vec2.y + vec2.z*vec2.z);
	double len3 = sqrt(vec3.x*vec3.x + vec3.y*vec3.y + vec3.z*vec3.z);
	if(len1 >= len2 && len1 >= len3){
		eigenVector->x = vec1.x / len1;
		eigenVector->y = vec1.y / len1;
		eigenVector->z = vec1.z / len1;
	} else if(len2 >= len1 && len2 >= len3) {
		eigenVector->x = vec2.x / len2;
		eigenVector->y = vec2.y / len2;
		eigenVector->z = vec2.z / len2;
	} else {
		eigenVector->x = vec3.x / len3;
		eigenVector->y = vec3.y / len3;
		eigenVector->z = vec3.z / len3;
	}
	return;
}

__global__ void computeNormalCM_GPU(float3* normal, double3* xyz, double3* xxxyxz, double3* yyyzzz, unsigned* integralCount, float* smoothingMap, int border, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int acs = y*width+x;

	//float bad_point = std::numeric_limits<float>::quiet_NaN ();
	float bad_point = -1.0;

	//set bad point by border
	if(x <= border || x >= width-border || y <= border || y >= height - border){
		normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
		return;
	}
	//convariance matrix
	double matrix[9];
	//smoothing size
	float smoothing = smoothingMap[acs];

	//use depth dependent smoothing			
	if(smoothing <= 2.0f){
		normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
		return;
	} else {
		//compute normal point
		int rect_width_ = smoothing;
		int rect_width_2_ = rect_width_ >> 1;
		int rect_height_ = smoothing;
		int rect_height_2_ = rect_width_ >> 1;

		unsigned cont = getFiniteElementsCount(integralCount, x-rect_width_2_-1, y-rect_height_2_-1, rect_width_, rect_height_, width, height);

		if(cont == 0){
			normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
			return;
		}
		double3 tmpxyz    = getSumFromIntegralImageD3(xyz, x-rect_width_2_-1, y-rect_height_2_-1, rect_width_, rect_height_, width, height);
		double3 tmpxxxyxz = getSumFromIntegralImageD3(xxxyxz, x-rect_width_2_-1, y-rect_height_2_-1, rect_width_, rect_height_, width, height);
		double3 tmpyyyzzz = getSumFromIntegralImageD3(yyyzzz, x-rect_width_2_-1, y-rect_height_2_-1, rect_width_, rect_height_, width, height);

		//make convariance matrix
		matrix[0] =				tmpxxxyxz.x - (tmpxyz.x*tmpxyz.x / cont);
		matrix[1] = matrix[3] = tmpxxxyxz.y - (tmpxyz.x*tmpxyz.y / cont);
		matrix[2] = matrix[6] = tmpxxxyxz.z - (tmpxyz.x*tmpxyz.z / cont);
		matrix[4] =				tmpyyyzzz.x - (tmpxyz.y*tmpxyz.y / cont);
		matrix[5] = matrix[7] = tmpyyyzzz.y - (tmpxyz.y*tmpxyz.z / cont);
		matrix[8] =				tmpyyyzzz.z - (tmpxyz.z*tmpxyz.z / cont);

		//compute eigen value and eigen vector
		double eigenValue;
		double3 eigenVector;
		computeEigenValueAndVector(matrix, &eigenValue, &eigenVector);

		if(eigenVector.z < 0.0f){
			normal[acs].x = (float)eigenVector.x, normal[acs].y = (float)-eigenVector.y, normal[acs].z = (float)eigenVector.z;
		} else {
			normal[acs].x = (float)-eigenVector.x, normal[acs].y = (float)eigenVector.y, normal[acs].z = (float)-eigenVector.z;
		}
	}
}

__global__ void computeRestNormalGPU(float3* normal, float3* vertice, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int acs = y*width+x;
	if(normal[acs].x == -1.0 && 
			normal[acs].y == -1.0 &&
				normal[acs].z == -1.0){
		int r = 1;
		if( vertice[y*width+(x+r)].z == 0.0) 
			r=-1;
		float3 v_h;
		float3 v_v;
		float3 ph01 = vertice[(y  )*width+x+r];
		float3 ph02 = vertice[(y  )*width+x  ];
		float3 pv01 = vertice[(y+r)*width+x  ];
		float3 pv02 = vertice[(y  )*width+x  ];

		v_h.x = (ph01.x-ph02.x);
		v_h.y = (ph01.y-ph02.y);
		v_h.z = (ph01.z-ph02.z);
		v_v.x = (pv01.x-pv02.x);
		v_v.y = (pv01.y-pv02.y);
		v_v.z = (pv01.z-pv02.z);

		float3* n  = &normal[acs];
		//compute cross product
		if( ph02.z != 0.0){
			n->x = v_h.z * v_v.y - v_h.y * v_v.z;
			n->y =-(v_h.x * v_v.z - v_h.z * v_v.x);
			n->z = v_h.y * v_v.x - v_h.x * v_v.y;
			float norm = sqrt(pow(n->x , 2) + pow(n->y, 2) + pow(n->z, 2));
			if(norm > 0.0){
				n->x /= -norm, n->y /= -norm, n->z /= -norm;				
			}
		}
	}
	if(normal[acs].x != -1.0 || 
			normal[acs].y != -1.0 ||
				normal[acs].z != -1.0){
		normal[acs].x *= -1.0f;
		normal[acs].y *= 1.0f;
		normal[acs].z *= -1.0f;
	}
}
__global__ void computeNormalBilateralGPU(float3* normal, float3* vertice, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int acs = y*width+x;
	if(vertice[acs].z == 0.0){
		float bad_point = -1.0f;
		normal[acs].x = bad_point, normal[acs].y = bad_point, normal[acs].z = bad_point;
		return;
	}
	int r = 1;
	if( vertice[y*width+(x+r)].z == 0.0) 
		r=-1;
	float3 v_h;
	float3 v_v;
	float3 ph01 = vertice[(y  )*width+x+r];
	float3 ph02 = vertice[(y  )*width+x  ];
	float3 pv01 = vertice[(y+r)*width+x  ];
	float3 pv02 = vertice[(y  )*width+x  ];

	v_h.x = (ph01.x-ph02.x);
	v_h.y = (ph01.y-ph02.y);
	v_h.z = (ph01.z-ph02.z);
	v_v.x = (pv01.x-pv02.x);
	v_v.y = (pv01.y-pv02.y);
	v_v.z = (pv01.z-pv02.z);

	float3* n  = &normal[acs];
	//compute cross product
	if( ph02.z != 0.0){
		n->x = v_h.z * v_v.y - v_h.y * v_v.z;
		n->y =-(v_h.x * v_v.z - v_h.z * v_v.x);
		n->z = v_h.y * v_v.x - v_h.x * v_v.y;
		float norm = sqrt(pow(n->x , 2) + pow(n->y, 2) + pow(n->z, 2));
		if(norm > 0.0){
			n->x /= -norm, n->y /= -norm, n->z /= -norm;				
		}
	}
	n->x *= -1.0f;
	n->y *= 1.0f;
	n->z *= -1.0f;
}

void NormalMapGenerator::computeNormal(float3* vertices_device){
	dim3 blockSize(BLOCKDIM, BLOCKDIM), gridSize(width / BLOCKDIM, height / BLOCKDIM);
	////initialize normalMap
	//compute normal by using integral image
	if(normal_estimation_method_ == SDC){
		computeNormalSDC_GPU<<<gridSize, blockSize>>>(normalMap, vertices_device, iig.getIntegralZ(), iig.getIntegralCount(), samg.getFinalSmoothingMap(), (int)samg.normal_smoothing_size_, width, height);
	} else if(normal_estimation_method_ == CM){
		computeNormalCM_GPU<<<gridSize, blockSize>>>(normalMap, iig.getIntegralXYZ(), iig.getIntegralXXXYXZ(), iig.getIntegralYYYZZZ(), iig.getIntegralCount(), samg.getFinalSmoothingMap(), (int)samg.normal_smoothing_size_, width, height);
	} else if(normal_estimation_method_ == BILATERAL){
		computeNormalBilateralGPU<<<gridSize, blockSize>>>(normalMap, vertices_device, width, height);
		return;
	}
	//bilateral normal estimation in invalid pixel
	computeRestNormalGPU<<<gridSize, blockSize>>>(normalMap, vertices_device, width, height);
}

__global__ void computeSegmentNormalImgGPU(float3* segmentMap, float3* normalMap, bool* mask, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int acs = y*width+x;
	if(mask[acs])
		segmentMap[acs] = normalMap[acs];
	else 
		segmentMap[acs].x = -1.0f, segmentMap[acs].y = -1.0f, segmentMap[acs].z = -1.0f;
}

cv::Mat NormalMapGenerator::getNormalImg(void){
	float3 * normal;
	normal = new float3[width*height];
	cudaMemcpy(normal, normalMap,sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
	cv::Mat normalMat(height, width, CV_8UC3);
	for(int y=0;y<height;y++){
		uchar *p = normalMat.ptr(y);
		for(int x=0;x<width;x++){
			//std::cout << "x: "<<normal[y*width+x].x<<", y: "<<normal[y*width+x].y<<", z:"<<normal[y*width+x].z<<std::endl;
			p[x*3+0] = (int)(255*(normal[y*width+x].x+1.0)/2);
			p[x*3+1] = (int)(255*(normal[y*width+x].y+1.0)/2);
			p[x*3+2] = (int)(255*(normal[y*width+x].z+1.0)/2);
		}
	}
	delete normal;
	return normalMat;
}

cv::Mat NormalMapGenerator::getSegmentNormalImg(bool*mask){
	float3 * segmentNormalMapHost;
	segmentNormalMapHost = new float3[width*height];

	dim3 blockSize(BLOCKDIM, BLOCKDIM), gridSize(width / BLOCKDIM, height / BLOCKDIM);
	computeSegmentNormalImgGPU<<<gridSize, blockSize>>>(segmentNormalMap, normalMap, mask, width, height);

	cudaMemcpy(segmentNormalMapHost, segmentNormalMap, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
	cv::Mat normalSementMat(480, 640, CV_8UC3);
	for(int y=0;y<height;y++){
		uchar *p = normalSementMat.ptr(y);
		for(int x=0;x<width;x++){
				//std::cout << segmentNormalMapHost[y*width+x].x<<", "<<segmentNormalMapHost[y*width+x].y <<", "<<segmentNormalMapHost[y*width+x].z<<std::endl;
		
				p[x*3+0] = (int)(255*(segmentNormalMapHost[y*width+x].x+1.0)/2);
				p[x*3+1] = (int)(255*(segmentNormalMapHost[y*width+x].y+1.0)/2);
				p[x*3+2] = (int)(255*(segmentNormalMapHost[y*width+x].z+1.0)/2);
		}
	}
	delete segmentNormalMapHost;
	return normalSementMat;
}

void NormalMapGenerator::saveNormalImg(char* str){
	float3 * normal;
	normal = new float3[width*height];
	cudaMemcpy(normal, normalMap,sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
	cv::Mat normalMat(480, 640, CV_8UC3);
	for(int y=0;y<height;y++){
		uchar *p = normalMat.ptr(y);
		for(int x=0;x<width;x++){
			p[x*3+0] = (int)(255*(normal[y*width+x].x+1.0)/2);
			p[x*3+1] = (int)(255*(normal[y*width+x].y+1.0)/2);
			p[x*3+2] = (int)(255*(normal[y*width+x].z+1.0)/2);
		}
	}
	delete normal;
	char filename[30];
	sprintf(filename,"%s.bmp", str);
	cv::imwrite(filename, normalMat);
}

void NormalMapGenerator::saveSegmentNormalImg(char* str, bool*mask){
	float3 * segmentNormalMapHost;
	segmentNormalMapHost = new float3[width*height];

	dim3 blockSize(BLOCKDIM, BLOCKDIM), gridSize(width / BLOCKDIM, height / BLOCKDIM);
	computeSegmentNormalImgGPU<<<gridSize, blockSize>>>(segmentNormalMap, normalMap, mask, width, height);

	cudaMemcpy(segmentNormalMapHost, segmentNormalMap, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
	cv::Mat normalSementMat(480, 640, CV_8UC3);
	for(int y=0;y<height;y++){
		uchar *p = normalSementMat.ptr(y);
		for(int x=0;x<width;x++){
				p[x*3+0] = (int)(255*(segmentNormalMapHost[y*width+x].x+1.0)/2);
				p[x*3+1] = (int)(255*(segmentNormalMapHost[y*width+x].y+1.0)/2);
				p[x*3+2] = (int)(255*(segmentNormalMapHost[y*width+x].z+1.0)/2);
		}
	}
	delete segmentNormalMapHost;
	char filename[30];
	sprintf(filename,"%s.bmp", str);
	cv::imwrite(filename, normalSementMat);
}
__global__ void setInput(float3* input, float3* vertices_map, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	vertices_map[y*width+x].x = input[y*width+x].x/1000.0f;
	vertices_map[y*width+x].y = input[y*width+x].y/1000.0f;
	vertices_map[y*width+x].z = input[y*width+x].z/1000.0f;
}

void NormalMapGenerator::generateNormalMap(float3* vertices_device){
	setInput<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(vertices_device, verticeMap, width, height);
	//compute Depth Integral Image
	iig.setInput(verticeMap);
	iig.generateIntegralImage(normal_estimation_method_);
	//compute Smoothing Area Map
	samg.setVerticeMap(verticeMap);
	samg.generateFinalSmoothingAreaMap();
	//compute normalMap
	computeNormal(verticeMap);
}
