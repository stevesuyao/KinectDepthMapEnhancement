#ifndef UNIVERSAL_DIMENSION_CONVERTOR_H
#define UNIVERSAL_DIMENSION_CONVERTOR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "thrust\device_ptr.h"
#include "thrust\device_vector.h"
#include "thrust\transform.h"
#include "thrust\iterator\counting_iterator.h"
#include "thrust\iterator\transform_iterator.h"
#include "thrust\iterator\zip_iterator.h"
#include "thrust\fill.h"
#include "thrust\reduce.h"
#include <opencv2\opencv.hpp>


class  DimensionConvertor{
public:
	struct convert_ptr{
		float fx;
		float fy;
		int width, height;
		int cx;
		int cy;
		__host__ convert_ptr(int width,	int height,	float fx, float fy,	int cx,	int cy){
				this->fx = fx;
				this->fy = fy;
				this->cx = cx;
				this->cy = cy;
				this->width = width;
				this->height = height;
		}

		 __host__ __device__ float3 operator()(float3 in){
			
			in.y = cy - in.y;			
			in.x = in.x - cx;
			
			in.x /= fx;
			in.y /= fy;

			in.x *= in.z;
			in.y *= in.z;
			//in.x /= 1000.0f;
			//in.y /= 1000.0f;
			//in.z /= 1000.0f;
			return in;
		}		


		__host__ __device__ float3 operator()(thrust::tuple<float, int> in){
			float3 r;

			//Depth
			r.z = (float)thrust::get<0>(in);

			//Decompose 1D index to (x, y)
			r.y = (float)(thrust::get<1>(in) / width);
			r.x = (float)(thrust::get<1>(in) % width);

			return operator()(r);
		}
	};
	
	struct convert_ptr_int{
		float fx;
		float fy;
		int width, height;
		int cx;
		int cy;
		__host__ convert_ptr_int(int width,	int height,	float fx, float fy,	int cx,	int cy){
				this->fx = fx;
				this->fy = fy;
				this->cx = cx;
				this->cy = cy;
				this->width = width;
				this->height = height;
		}

		__host__ __device__ float3 operator()(float3 in){
			in.y = cy - in.y/2.0f;			
			in.x = in.x/2.0f - cx;
			
			in.x /= fx;
			in.y /= fy;

			in.x *= in.z;
			in.y *= in.z;
			return in;					
		}

		__host__ __device__ float3 operator()(thrust::tuple<float, int> in){
			float3 r;

			//Depth
			r.z = thrust::get<0>(in);

			//Decompose 1D index to (x, y)
			r.y = (float)(thrust::get<1>(in) / (width*2));
			r.x = (float)(thrust::get<1>(in) % (width*2));

			return operator()(r);
		}
	};

	struct convert_rtp {
		float fx;
		float fy;
		int cx;
		int cy;
		int width;
		int height;
		
		__host__ convert_rtp(
			int width,
			int height,
			float fx,
			float fy,
			int cx,
			int cy	) {
				this->fx = fx;
				this->fy = fy;
				this->cx = cx;
				this->cy = cy;
				this->width = width;
				this->height = height;
		}
		__host__ __device__ float3 operator()(float3 in){
			float3 out;
			//‰æ‘œ“à‚É“Š‰e‚³‚ê‚È‚¢‚Æ‚«
			if(abs(in.z) < 1.0){
				out.x = -1.0;
				out.y = -1.0;
			} 
			//“Š‰e‚É¬Œ÷‚µ‚½‚Æ‚«
			else {
				out.x = in.x / in.z;
				out.y = in.y / in.z;
				out.x *= fx;
				out.y *= fy;

				out.x = out.x + cx;
				out.y = cy - out.y;
			}
				out.z = in.z;	
			return out;
		}
	};



	DimensionConvertor(){};
	~DimensionConvertor();


	void setCameraParameters(const cv::Mat_<double> intrinsic, int width, int height);

	void projectiveToReal(float* data, float3* out);
	void projectiveToReal(float3* data, float3* out);
	void projectiveToRealInterp(float* data, float3* out);
	void realToProjective(float3* data, float3* out);



private:
	float Fx;
	float Fy;
	int Cx;
	int Cy;
	int Width;
	int Height;
};


#endif