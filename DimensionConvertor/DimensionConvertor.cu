#include "DimensionConvertor.h"

void DimensionConvertor::projectiveToReal(float* data, float3* out){
	thrust::counting_iterator<int> index(0);
	float3 reset;
	reset.x = 0.0;
	reset.y = 0.0;
	reset.z = 0.0;
	thrust::fill(
		thrust::device_ptr<float3>(out),
		thrust::device_ptr<float3>(out) + Width*Height,
		reset
	);

	thrust::device_ptr<float> dev_ptr(data);
	
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(dev_ptr, index)),
		thrust::make_zip_iterator(thrust::make_tuple(dev_ptr + Width*Height, index + Width*Height)),
		thrust::device_ptr<float3>(out), 
		convert_ptr(Width, Height, Fx, Fy, Cx, Cy)
	);
}

void DimensionConvertor::projectiveToReal(float3* data, float3* out){
	
	thrust::transform(
		thrust::device_ptr<float3>(data),
		thrust::device_ptr<float3>(data) + Width*Height,
		thrust::device_ptr<float3>(out),
		convert_ptr(Width, Height, Fx, Fy, Cx, Cy)
	);
}

void DimensionConvertor::realToProjective(float3* data, float3* out){
	
	thrust::transform(
		thrust::device_ptr<float3>(data),
		thrust::device_ptr<float3>(data) + Width*Height,
		thrust::device_ptr<float3>(out),
		convert_rtp(Width, Height, Fx, Fy, Cx, Cy)
	);
}
void DimensionConvertor::projectiveToRealInterp(float* data, float3* out){

	thrust::counting_iterator<int> index(0);
	float3 reset;
	reset.x = 0.0;
	reset.y = 0.0;
	reset.z = 0.0;

	thrust::fill(
		thrust::device_ptr<float3>(out),
		thrust::device_ptr<float3>(out) + Width*Height,
		reset
	);

	thrust::device_ptr<float> dev_ptr(data);

	//thrust::transform(
	//	thrust::make_zip_iterator(thrust::make_tuple(dev_ptr, index)),
	//	thrust::make_zip_iterator(thrust::make_tuple(dev_ptr + size, index + size)),
	//	thrust::device_ptr<float3>(out), 
	//	cvt
	//);



	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(dev_ptr, index)),
		thrust::make_zip_iterator(thrust::make_tuple(dev_ptr + Width*Height, index + Width*Height)),
		thrust::device_ptr<float3>(out), 
		convert_ptr_int(Width, Height, Fx, Fy, Cx, Cy)
	);


}

