#ifndef Cluster_H
#define Cluster_H

#include "opencv2/opencv.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

class Cluster
{
public:
	Cluster();
	~Cluster();
	void				SetClusterDistance(double& Cluster_dist);
	void				SetPlaneDistance(double& plane_dist);
	void				SetNormal(float3& normal);
	void				AddCluster3Dpoints(const cv::Mat& point);
	void				ClearCluster3Dpoints();
	void				ClearClusterRGB();
	int					GetClusterSize()const;
	double				GetClusterDistance()const; 
	float3				GetNormal()const;
	void				canPlane(bool is_plane);
	bool				canPlane(double& eigenvalues1, double& eigenvalues2);
	bool				canPlane()const;
	cv::Mat*			GetCluster3Dpoints();
	std::vector<int>*	GetIndices();
	void				SetClusterCenter(cv::Point3d& g);
	cv::Point3d			GetClusterCenter();
private:
	double				Cluster_Distance;			//重心までの距離
	double				Plane_Distance;				//平面までの距離
	float3				Cluster_Normal;				//clusterの法線ベクトル
	cv::Mat				Cluster_3Dpoints;			//Clusterの3Dpoint
	cv::Mat				Cluster_RGB;				//RGB情報				
	bool				isPlane;					//平面にできるかのフラグ
	std::vector<int>	Indices;
	cv::Point3d			Cluster_Center;
};

#endif