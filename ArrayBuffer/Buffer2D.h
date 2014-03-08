#ifndef BUFFER2D
#define BUFFER2D

#include "ArrayBuffer.h"
#include <XnCppWrapper.h>



class Buffer2D : public ArrayBuffer{
public:

	explicit Buffer2D(int width, int height);
	
	virtual void insertData(float* data);

	virtual void insertData(weighted_d* data);

	virtual void getDepthMap(float* out);

	virtual void getWeightMap(float* out);

	virtual void updateData(float* data);

	void insertData(float2* data);

	void insertData(xn::DepthMetaData* data);


private:



	const float DEPTH_THRESHOLD;

};



#endif