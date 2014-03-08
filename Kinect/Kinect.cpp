#include "Kinect.h"
#include <sstream>
#include <iostream>
#include <stdexcept>
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>

//using namespace std;

const int Kinect::Width = 640;
const int Kinect::Height = 480;

Kinect::Kinect():
	Image_Generator(0),
	Depth_Generator(0),
	ImageMD(0),
	Color_Image(Kinect::Height, Kinect::Width),
	DepthMD(0),
	Depth_Image(Kinect::Height, Kinect::Width),
	Kinect_ID(0),
	Intrinsic_Matrix(3, 3){}
//アドレスに0を代入してアクセス違反を防ぐ
Kinect::~Kinect(){
	delete   Image_Generator;
	delete   Depth_Generator;
	delete     	     ImageMD;
	delete  	     DepthMD;
	Image_Generator = 0;
	Depth_Generator = 0;
	ImageMD = 0;
	DepthMD = 0;
}
void Kinect::SetImageGenerator(xn::ImageGenerator* image_generator){
	Image_Generator = image_generator;
}
void Kinect::SetDepthGenerator(xn::DepthGenerator* depth_generator){
	Depth_Generator = depth_generator;
}
void Kinect::SetKinectID(int kinect_id){
	Kinect_ID = kinect_id;
}
xn::DepthMetaData*  Kinect::GetDepthMD()const{
	return DepthMD;
}

cv::Mat_<cv::Vec3b> Kinect::GetColorImage()const{
	return Color_Image;
}
cv::Mat_<cv::Vec3b> Kinect::GetDepthImage(){
	CreateDepthImage();
	return Depth_Image;
}
cv::Mat_<double> Kinect::GetIntrinsicMatrix()const{
	return Intrinsic_Matrix;
}
int Kinect::GetKinectID()const{
	return Kinect_ID;
}
int Kinect::getMaxDepth()const{
	return Depth_Generator->GetDeviceMaxDepth();
}
void Kinect::ProjectToReal(XnPoint3D& proj, XnPoint3D& real){
	Depth_Generator->ConvertProjectiveToRealWorld(1, &proj, &real);
}
void Kinect::RealToProject(XnPoint3D& real, XnPoint3D& proj){
	Depth_Generator->ConvertRealWorldToProjective(1, &real, &proj);
}
void Kinect::InitAllData(){

	// デプスの座標をイメージに合わせる
	xn::AlternativeViewPointCapability viewPoint =
		Depth_Generator->GetAlternativeViewPointCap();
	//ビューポイント機能を取得する
	//ビューポイントを指定したノードに合わせる
	viewPoint.SetViewPoint(*Image_Generator);
	//depthMD作成
	DepthMD = new xn::DepthMetaData();
	//imageMD作成
	ImageMD = new xn::ImageMetaData();
	//RGB画像の作成
	XnMapOutputMode outputmode;
	Image_Generator->GetMapOutputMode(outputmode);
	//Color画像の作成
	//Color_Image = cv::Mat_<cv::Vec3b>(Kinect::Height, Kinect::Width);
	//Depth画像の作成
	//Depth_Image = cv::Mat_<cv::Vec3b>(Kinect::Height, Kinect::Width);
	//IntrinsicMatrix
	//Intrinsic_Matrix = cv::Mat_<double>(3, 3);
	XnDouble pixel_size;
	unsigned long long F;
	Depth_Generator->GetIntProperty("ZPD", F);
	Depth_Generator->GetRealProperty("ZPPS", pixel_size);
	Intrinsic_Matrix = (cv::Mat_<double>(3, 3) << (double)F/(double)(2.0*pixel_size), 0.0, Kinect::Width/2.0,
													0.0, (double)F/(double)(2.0*pixel_size), Kinect::Height/2.0,
														0.0, 0.0, 1.0);
}

void Kinect::UpdateAllData(){
	//imageMetadata作成
	Image_Generator->GetMetaData(*ImageMD);
	//depthMetadata作成
	Depth_Generator->GetMetaData(*DepthMD);
	//color_image作成
	memcpy(Color_Image.data, ImageMD->Data(), Color_Image.step * Color_Image.rows); 
	cv::cvtColor(Color_Image, Color_Image, CV_RGB2BGR);
}

void Kinect::CreateDepthImage(){
	
	//デプスの傾向を計算する(アルゴリズムはNiSimpleViewer.cppを利用)
	const int MAX_DEPTH = Depth_Generator->GetDeviceMaxDepth();
	//デプスの最大値取得
	std::vector<float> depth_hist(MAX_DEPTH);

	unsigned int points = 0;
	//points デプスがとれた全点の数
	const XnDepthPixel* pDepth = DepthMD->Data();
	for (XnUInt y = 0; y < DepthMD->YRes(); ++y) {
		for (XnUInt x = 0; x < DepthMD->XRes(); ++x, ++pDepth) {
			if (*pDepth != 0) {
				//depth_hist[*pDepth]++;
				//std::cout << depth_data[y*Image_Size->width+x]<< std::endl;
				depth_hist[*pDepth]++;
				points++;
			}
			//もしデプスがとれたらpDepthの位置のデプスの値番目のヒストグラムをカウント
			//デプスがとれた点の数もカウント
		}
	}

	for (int i = 1; i < MAX_DEPTH; ++i) {
		depth_hist[i] += depth_hist[i-1];
		//全部のデプス値のヒストグラムの値についてそれより値が小さいものの値を加算
		//これによりデプスの大きな値のヒストグラムは大きな値になる
	}

	if ( points != 0) {
		for (int i = 1; i < MAX_DEPTH; ++i) {
			depth_hist[i] =
				(float)(256 * (1.0f - (depth_hist[i] / points)));
			//デプスが大きいものほど小さいヒストグラムを割り当てる
			//std::cout << depth_hist[i] << std::endl;
		}
	}

	for(unsigned int y=0; y < DepthMD->YRes(); y++){
		for(unsigned int x=0; x < DepthMD->XRes(); x++){
			Depth_Image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, (uchar)depth_hist[(*DepthMD)(x, y)], (uchar)depth_hist[(*DepthMD)(x, y)]); 
		}
	}
}

//void Kinect::SavePointcloud(double range_near, double range_far){
//	//取得した点群
//	pcl::PointCloud<pcl::PointXYZRGB> cloud;
//	//各点の情報
//	pcl::PointXYZRGB points;
//	XnPoint3D proj, real;
//	for(unsigned int y=0; y<DepthMD->YRes(); y++){
//		for(unsigned int x=0; x<DepthMD->XRes(); x++){
//			double depth_tmp = (*DepthMD)(x, y);
//			if(depth_tmp > range_near*1000 && depth_tmp < range_far*1000){ 
//				//画像と距離の対応を取得
//				proj.X = (XnFloat)x;
//				proj.Y = (XnFloat)y;
//				proj.Z = (XnFloat)depth_tmp;
//				//世界座標へ変換
//				Depth_Generator->ConvertProjectiveToRealWorld(1, &proj, &real);
//				points.x = (float)(real.X / 1000.0);
//				points.y = (float)(real.Y / 1000.0);
//				points.z = (float)(real.Z / 1000.0);
//				points.b = (unsigned char)Color_Image.at<cv::Vec3b>(y, x).val[0];
//				points.g = (unsigned char)Color_Image.at<cv::Vec3b>(y, x).val[1];
//				points.r = (unsigned char)Color_Image.at<cv::Vec3b>(y, x).val[2];
//				//pointcloudに格納
//				cloud.push_back(points);
//			}
//		}
//	}
//	pcl::io::savePCDFile("input_points.pcd", cloud);
//}

SingleKinect::SingleKinect(){
	//ContextとGeneratorの初期化
	Registration();
	//memberの初期化
	InitAllData();
}

SingleKinect::~SingleKinect(){
	delete Kinect_Context;
	Kinect_Context = 0;
}
void SingleKinect::UpdateContextAndData(){
	Kinect_Context->WaitAndUpdateAll();
	UpdateAllData();
}
//void SingleKinect::ShowImage(){
//
//	bool ShouldRun = true;
//	char key;
//	while( ShouldRun ){
//		//データ更新を待つ
//		Kinect_Context->WaitAndUpdateAll();
//		UpdateAllData();
//		//depthimage作成
//		CreateDepthImage();
//		//カラー画像表示
//		cv::imshow("color_image", Color_Image);
//		key = cv::waitKey(1);
//		cv::imshow("depth_image", Depth_Image);
//		key = cv::waitKey(1);  
//		if( key == 's' ){
//			cv::imwrite("color_image.jpg", Color_Image);
//			std::cout << "save color_image" << std::endl;
//			//pointcloud保存
//			SavePointcloud(0.0, 10.0);
//			std::cout << "save pointcloud" << std::endl;
//			ShouldRun = false;
//		}
//	}
//}

void SingleKinect::Registration(){
	// コンテキストの初期化
	Kinect_Context = new xn::Context();
	XnStatus rc = Kinect_Context->InitFromXmlFile("Kinect/SamplesConfig.xml");
	if (rc != XN_STATUS_OK) {
		throw std::runtime_error(xnGetStatusString(rc));
	}

	// イメージジェネレータの作成
	Image_Generator = new xn::ImageGenerator();
	//指定したノードのインスタンスを作成する
	rc = Kinect_Context->FindExistingNode(XN_NODE_TYPE_IMAGE, *Image_Generator);
	if (rc != XN_STATUS_OK) {
		throw std::runtime_error(xnGetStatusString(rc));
	}

	// デプスジェネレータの作成
	Depth_Generator = new xn::DepthGenerator();
	//指定したノードのインスタンスを作成する
	rc = Kinect_Context->FindExistingNode(XN_NODE_TYPE_DEPTH, *Depth_Generator);
	if (rc != XN_STATUS_OK) {
		throw std::runtime_error(xnGetStatusString(rc));
	}

}
