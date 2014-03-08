#ifndef KINECT_H
#define KINECT_H

#include <XnCppWrapper.h>
#include <opencv2/opencv.hpp>


class Kinect
{
public:
	Kinect();
	virtual ~Kinect();
	static const int			Width;
	static const int			Height;
	void						SetImageGenerator(xn::ImageGenerator* image_generator);
	void						SetDepthGenerator(xn::DepthGenerator* depth_generator);
	void						SetKinectID(int kinect_id);
	void						ProjectToReal(XnPoint3D& proj, XnPoint3D& real);
	void						RealToProject(XnPoint3D& real, XnPoint3D& proj);
	xn::DepthMetaData*			GetDepthMD()const;
	cv::Mat_<cv::Vec3b>			GetColorImage()const;
	cv::Mat_<cv::Vec3b>			GetDepthImage();
	cv::Mat_<double>			GetIntrinsicMatrix()const;
	int							GetKinectID()const;
	int							getMaxDepth()const;
	void						InitAllData();		
	void						UpdateAllData();		
	void						CreateDepthImage();
protected:
	int							Kinect_ID;
	cv::Mat_<double>			Intrinsic_Matrix;
	xn::ImageGenerator*			Image_Generator;  //imageÇÃèÓïÒ
	xn::DepthGenerator*			Depth_Generator;	//depthÇÃèÓïÒ
	xn::ImageMetaData*			ImageMD;
	cv::Mat_<cv::Vec3b>			Color_Image;
	xn::DepthMetaData*			DepthMD;
	cv::Mat_<cv::Vec3b>			Depth_Image;
	//void						SavePointcloud(double range_near, double range_far);
};

class SingleKinect: public Kinect
{
public:
	SingleKinect();
	~SingleKinect();
	void					UpdateContextAndData();
	//void					ShowImage();
private:
	xn::Context*			Kinect_Context;
	void					Registration();
};

#endif