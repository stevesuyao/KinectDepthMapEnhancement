#include "Cluster.h"
#include <map>
#include <iostream>
#include <XnCppWrapper.h>


//ClusterponentƒNƒ‰ƒX‚Ìƒƒ“ƒoŠÖ”
Cluster::Cluster():
isPlane(true)
{	
	Cluster_3Dpoints = cv::Mat();
	Cluster_RGB = cv::Mat();
	Cluster_Distance = 0; 
	Cluster_Normal.x = 0;
	Cluster_Normal.y = 0;
	Cluster_Normal.z = 0;
	Cluster_Center.x = 0.0;
	Cluster_Center.y = 0.0;
	Cluster_Center.z = 0.0;
}
Cluster::~Cluster(){
	Cluster_Distance = 0;
	Cluster_Normal.x = 0;
	Cluster_Normal.y = 0;
	Cluster_Normal.z = 0;
	Cluster_Center.x = 0.0;
	Cluster_Center.y = 0.0;
	Cluster_Center.z = 0.0;
}
void Cluster::SetClusterDistance(double& Cluster_dist){
	Cluster_Distance = Cluster_dist;
}
void Cluster::SetPlaneDistance(double& plane_dist){
	Plane_Distance = plane_dist;
}
void Cluster::SetNormal(float3& normal){
	Cluster_Normal = normal;
}
void Cluster::AddCluster3Dpoints(const cv::Mat& point){
	//cv::Mat point_3d = (cv::Mat_<double>(1, 3) << point.at<double>(0, 0), point.at<double>(0, 1), point.at<double>(0, 1));
	Indices.push_back(Cluster_3Dpoints.rows);
	Cluster_3Dpoints.push_back(point);
}
void Cluster::ClearCluster3Dpoints(){
	Cluster_3Dpoints.release();
}
void Cluster::ClearClusterRGB(){
	Cluster_RGB.release();
}
int Cluster::GetClusterSize()const{
	return Cluster_3Dpoints.rows;
}
double Cluster::GetClusterDistance()const{
	return Cluster_Distance;
}
float3	Cluster::GetNormal()const{
	return Cluster_Normal;
}
cv::Mat* Cluster::GetCluster3Dpoints(){
	return &Cluster_3Dpoints;
}
void Cluster::canPlane(bool is_plane){
	isPlane = is_plane;
}
bool Cluster::canPlane(double& eigenvalues1, double& eigenvalues2){
	double threshold = Cluster_Distance*2.0;
	isPlane = !(eigenvalues2 > threshold || eigenvalues1 > 10.0);
	return isPlane;
	/*
	if(eigenvalues2 > threshold || eigenvalues1 > 2.0)
		isPlane = false;
	else
		isPlane = true;
	return isPlane;*/
}
bool Cluster::canPlane()const{
	return isPlane;
}
std::vector<int>* Cluster::GetIndices(){
	return &Indices;
}
void Cluster::SetClusterCenter(cv::Point3d& g){
	Cluster_Center = g;
}
cv::Point3d	Cluster::GetClusterCenter(){
	return Cluster_Center;
}