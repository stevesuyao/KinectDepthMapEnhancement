#include "TOFDepthInterpolation.h"
#include "SuperpixelSegmentation\DepthAdaptiveSuperpixel.h"
#include "EdgeRefinedSuperpixel\EdgeRefinedSuperpixel.h"
#include "DimensionConvertor\DimensionConvertor.h"
#include "NormalEstimation\NormalMapGenerator.h"
#include "LabelEquivalenceSeg\LabelEquivalenceSegPCA.h"
#include "Projection_GPU\Projection_PCA.h"
#include "Cluster\Cluster.h"
#include "Kinect\Kinect.h"

TOFDepthInterpolation::TOFDepthInterpolation(int width, int height):
	Width(width),
	Height(height){
		DASP = new DepthAdaptiveSuperpixel(width, height);
		SP = new DepthAdaptiveSuperpixel(width, height);
		ERS = new EdgeRefinedSuperpixel(width, height);
		Convertor = new DimensionConvertor();
		spMerging = new LabelEquivalenceSegPCA(width, height);
		cudaMalloc(&EdgeEnhanced3DPoints_Device, width * height * sizeof(float3));
		cudaMallocHost(&EdgeEnhanced3DPoints_Host, width * height * sizeof(float3));
	}
TOFDepthInterpolation::~TOFDepthInterpolation(){
	delete DASP;
	DASP = 0;
	delete SP;
	SP = 0;
	delete ERS;
	ERS = 0;
	delete spMerging;
	spMerging = 0;
	delete Projector;
	Projector = 0;
	cudaFree(EdgeEnhanced3DPoints_Device);
	cudaFree(EdgeEnhanced3DPoints_Host);
	delete [] Cluster_Array;
	Cluster_Array = 0;
	cudaFree(ClusterND_Host);
	cudaFree(ClusterND_Device);
	cudaFree(ClusterCenter_Host);
	cudaFree(ClusterCenter_Device);
	cudaFree(ClusterEigenvalues_Host);
	cudaFree(ClusterEigenvalues_Device);
	
}
void TOFDepthInterpolation::SetParametor(int rows, int cols, cv::Mat_<double> intrinsic){
	sp_rows = rows;
	sp_cols = cols;
	SP->SetParametor(sp_rows, sp_cols, intrinsic);
	DASP->SetParametor(sp_rows, sp_cols, intrinsic);
	Convertor->setCameraParameters(intrinsic, Width, Height);
	Projector = new Projection_PCA(Width, Height, intrinsic);
	Cluster_Array = new Cluster[rows*cols];
	cudaMallocHost(&ClusterND_Host, sizeof(float4)*rows*cols);
	cudaMalloc(&ClusterND_Device, sizeof(float4)*rows*cols);
	cudaMallocHost(&ClusterCenter_Host, sizeof(float3)*rows*cols);
	cudaMalloc(&ClusterCenter_Device, sizeof(float3)*rows*cols);
	cudaMallocHost(&ClusterEigenvalues_Host, sizeof(float)*rows*cols);
	cudaMalloc(&ClusterEigenvalues_Device, sizeof(float)*rows*cols);
}
void TOFDepthInterpolation::Process(float* depth_device, float3* points_device, cv::gpu::GpuMat color_device){
	//segmentation
	SP->Segmentation(color_device, points_device, 200.0f, 10.0f, 0.0f, 5);
	DASP->Segmentation(color_device, points_device, 0.0f, 10.0f, 200.0f, 5);
	//edge refinement
	ERS->EdgeRefining(SP->getLabelDevice(), DASP->getLabelDevice(), depth_device, color_device);
	//convert to realworld
	Convertor->projectiveToReal(ERS->getRefinedDepth_Device(), EdgeEnhanced3DPoints_Device);
	cudaMemcpy(EdgeEnhanced3DPoints_Host, EdgeEnhanced3DPoints_Device, sizeof(float3)*Width*Height, cudaMemcpyDeviceToHost);
	//clustering
	for(int y=0; y<Height; y++){
		for(int x=0; x<Width; x++){
			int label = ERS->getRefinedLabels_Host()[y*Width+x];
			if(label != -1){
				//store points into cluster
				cv::Mat point = (cv::Mat_<double>(1, 3) << (double)EdgeEnhanced3DPoints_Host[y*Width+x].x, 
																	(double)EdgeEnhanced3DPoints_Host[y*Width+x].y, 
																		(double)EdgeEnhanced3DPoints_Host[y*Width+x].z);
				//push back
				Cluster_Array[label].AddCluster3Dpoints(point);
			}
		}
	}
	//normal estimation using PCA
	//Clusterごとにパラメータを計算
	for(int cluster_id=0; cluster_id <sp_rows*sp_cols; cluster_id++){	
		//各領域の点群の数の閾値
		if(Cluster_Array[cluster_id].GetClusterSize() >= 3){
			//PCA
			cv::PCA PrincipalComponent(*(Cluster_Array[cluster_id].GetCluster3Dpoints()), cv::Mat(), CV_PCA_DATA_AS_ROW);
			//法線ベクトルを求める
			float3 nor;
			nor.x = (float)PrincipalComponent.eigenvectors.at<double>(2,0);
			nor.y = (float)PrincipalComponent.eigenvectors.at<double>(2,1);
			nor.z = (float)PrincipalComponent.eigenvectors.at<double>(2,2);
			//重心ベクトル
			cv::Point3d g;
			g.x = PrincipalComponent.mean.at<double>(0, 0);
			g.y = PrincipalComponent.mean.at<double>(0, 1);
			g.z = PrincipalComponent.mean.at<double>(0, 2);
			//重心までの距離
			double g_distance = sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
			Cluster_Array[cluster_id].SetClusterDistance(g_distance);
			ClusterCenter_Host[cluster_id].x = (float)g.x;
			ClusterCenter_Host[cluster_id].y = (float)g.y;
			ClusterCenter_Host[cluster_id].z = (float)g.z;
			Cluster_Array[cluster_id].SetClusterCenter(g);
			//平面と原点の距離
			double plane_d_tmp = nor.x * g.x + nor.y * g.y + nor.z * g.z;
			//normalが反対方向を向いた場合
			if(plane_d_tmp < 0){
				nor.x *= -1.0;
				nor.y *= -1.0;
				nor.z *= -1.0;
			}
			//normal格納
			Cluster_Array[cluster_id].SetNormal(nor);
			ClusterND_Host[cluster_id].x = nor.x;
			ClusterND_Host[cluster_id].y = nor.y;
			ClusterND_Host[cluster_id].z = nor.z;
			//std::cout << nor.x <<", "<<nor.y<<", "<<nor.z<<std::endl;
			//平面までの距離取得
			double plane_d = fabs(plane_d_tmp);
			Cluster_Array[cluster_id].SetPlaneDistance(plane_d);
			ClusterND_Host[cluster_id].w = (float)plane_d;
			//eigenvalueを使って平面か判断
			ClusterEigenvalues_Host[cluster_id] = (float)PrincipalComponent.eigenvalues.at<double>(2, 0);
			//double eigenvalues1 = PrincipalComponent.eigenvalues.at<double>(0, 0)/PrincipalComponent.eigenvalues.at<double>(1, 0);
			//double eigenvalues2 = PrincipalComponent.eigenvalues.at<double>(2, 0);
			////平面にできないとき
			//if(!Cluster_Array[cluster_id].canPlane(eigenvalues1, eigenvalues2)){
			//	/*Plane_Num--;
			//	Cluster_ND[cluster_id].x = 5.0;
			//	Cluster_ND[cluster_id].y = 5.0;
			//	Cluster_ND[cluster_id].z = 5.0;*/
			//}
		}
		//clusterの３次元点が少なすぎるとき
		else{
			Cluster_Array[cluster_id].canPlane(false);
			ClusterND_Host[cluster_id].x = 5.0;
			ClusterND_Host[cluster_id].y = 5.0;
			ClusterND_Host[cluster_id].z = 5.0;
		}
		//マップをclear
		Cluster_Array[cluster_id].ClearCluster3Dpoints();
	}	
	//SingleKinect kinect;
	//for(int y=0; y<Height; y++){
	//	for(int x=0; x<Width; x++){
	//		XnPoint3D proj, real;
	//		proj.X = x;
	//		proj.Y = y;
	//		proj.Z = 1;
	//		kinect.ProjectToReal(proj, real);
	//		int label = ERS->getRefinedLabels_Host()[y*Width+x];
	//		if(label != -1 && Cluster_Array[label].canPlane()){
	//			EdgeEnhanced3DPoints_Host[y*Width+x].z = abs(ClusterND_Host[label].w/(ClusterND_Host[label].x*(float)real.X+
	//																					ClusterND_Host[label].y*(float)real.Y+
	//																					ClusterND_Host[label].z));
	//			//std::cout <<EdgeEnhanced3DPoints_Host[y*Width+x].z<<", "<<ClusterND_Host[label].w<<std::endl;
	//			EdgeEnhanced3DPoints_Host[y*Width+x].x = EdgeEnhanced3DPoints_Host[y*Width+x].z*(float)real.X;
	//			EdgeEnhanced3DPoints_Host[y*Width+x].y = EdgeEnhanced3DPoints_Host[y*Width+x].z*(float)real.Y;
	//		}
	//		else{
	//			EdgeEnhanced3DPoints_Host[y*Width+x].x = 0.0f;
	//			EdgeEnhanced3DPoints_Host[y*Width+x].y = 0.0f;
	//			EdgeEnhanced3DPoints_Host[y*Width+x].z = 0.0f;
	//		}
	//	}
	//}
	//cudaMemcpy(EdgeEnhanced3DPoints_Device, EdgeEnhanced3DPoints_Host, sizeof(float3)*Width*Height, cudaMemcpyHostToDevice);
	cudaMemcpy(ClusterEigenvalues_Device, ClusterEigenvalues_Host, sizeof(float)*sp_rows*sp_cols, cudaMemcpyHostToDevice);
	cudaMemcpy(ClusterND_Device, ClusterND_Host, sizeof(float4)*sp_rows*sp_cols, cudaMemcpyHostToDevice);
	cudaMemcpy(ClusterCenter_Device, ClusterCenter_Host, sizeof(float3)*sp_rows*sp_cols, cudaMemcpyHostToDevice);
	cv::Mat_<cv::Vec3b> normalImage(Height, Width);
	for(int y=0; y<Height; y++){
		for(int x=0; x<Width; x++){
			int id = ERS->getRefinedLabels_Host()[y*Width+x];
			if(id==-1)
				normalImage.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 0, 0);
			else{
				//std::cout << ClusterND_Host[id].x << ", "<<ClusterND_Host[id].y <<", "<<ClusterND_Host[id].z <<std::endl;
				normalImage.at<cv::Vec3b>(y,x).val[0] = (unsigned char)(255.0f*(ClusterND_Host[id].x+1.0f)/2.0f);
				normalImage.at<cv::Vec3b>(y,x).val[1] = (unsigned char)(255.0f*(ClusterND_Host[id].y+1.0f)/2.0f);
				normalImage.at<cv::Vec3b>(y,x).val[2] = (unsigned char)(255.0f*(ClusterND_Host[id].z+1.0f)/2.0f);
			}
		}
	}
	cv::imshow("normal_cluster", normalImage);
	cv::waitKey(0);
	//superpixel merging
	spMerging->labelImage(ClusterND_Device, ERS->getRefinedLabels_Device(), ClusterCenter_Device, ClusterEigenvalues_Device);
	//plane projection
	Projector->PlaneProjection(ClusterND_Device, ERS->getRefinedLabels_Device(), ClusterEigenvalues_Device, EdgeEnhanced3DPoints_Device);
}
float*	TOFDepthInterpolation::getRefinedDepth_Device(){
	return ERS->getRefinedDepth_Device();
}
float*	TOFDepthInterpolation::getRefinedDepth_Host(){
	return ERS->getRefinedDepth_Host();
}
float3*	TOFDepthInterpolation::getOptimizedPoints_Device(){
	return Projector->GetOptimized3D_Device();
}
float3*	TOFDepthInterpolation::getOptimizedPoints_Host(){
	return Projector->GetOptimized3D_Host();
}