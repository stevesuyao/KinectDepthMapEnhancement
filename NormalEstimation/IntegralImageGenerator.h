//////////////////////////////////////////////////
// contents :generate integral image from depthmap
// create 	:2013/03/17
// modefied :
// writer   :Takuya Ikeda 
// other	:
//////////////////////////////////////////////////

#ifndef _INTEGRALIMAGEgenerator_H_
#define _INTEGRALIMAGEgenerator_H_


#include <cutil_inline.h>
#include <cuda.h>

#define M_WIDTH 1024
#define M_HEIGHT 512
#define BLOCKDIM 16

class IntegralImagegenerator{

public:
	IntegralImagegenerator(int w, int y);
	~IntegralImagegenerator(void);
	void setInput(float3* in);
	void generateIntegralImage(int method);

	double* getIntegralZ(void);
	double3* getIntegralXYZ(void);
	double3* getIntegralXXXYXZ(void);
	double3* getIntegralYYYZZZ(void);
	unsigned* getIntegralCount(void);
	
private:
	int width, height;
	static const int SDC = 0, CM = 1, BILATERAL = 2;
	//tmp array for computing integral image
	double *dinMd, *doutMd;
	unsigned* dinMui, *doutMui;

	float3 *vertexMap;

	//for SDC
	double *IntegralZ;
	//for CM
	double3 *IntegralXYZ;
	double3 *IntegralXXXYXZ;
	double3 *IntegralYYYZZZ;
	//count
	unsigned *IntegralCount;

	void initMemory(void);
	void computeCount(void);
	void computeZ(void);
	void computeXYZ(void);
	void computeAllComponentMull(void);
	void computeXXXYXZ(void);
	void computeYYYZZZ(void);
};
#endif 