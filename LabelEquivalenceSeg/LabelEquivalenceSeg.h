#ifndef LABEL_EQUIVALENCE_SEG
#define LABEL_EQUIVALENCE_SEG
#include <opencv2\opencv.hpp>
#include <cuda_runtime.h>
#include <opencv2\gpu\gpu.hpp>

class LabelEquivalenceSeg{
public:
		typedef struct __align__(4){
			unsigned char r;
			unsigned char g;
			unsigned char b;
		}rgb;

		LabelEquivalenceSeg(int width, int height);
        ~LabelEquivalenceSeg();
		void labelImage(float3* cluster_normals_device, int* cluster_label_device, float3* cluster_centers_device, float* variance_device);
		//void labelImage(float4* cluster_nd_device, int* cluster_label_device, float3* cluster_centers_device);
		void viewSegmentResult();
		cv::Mat_<cv::Vec3b> getSegmentResult();
		cv::Mat_<cv::Vec3b>	getNormalImg();
        float4* getMergedClusterND_Device()const;
		int* getMergedClusterLabel_Device()const;
		float4* getMergedClusterND_Host()const;
		int* getMergedClusterLabel_Host()const;
		float* getMergedClusterVariance_Device()const;
		float* getMergedClusterVariance_Host()const;
		int* getMergedClusterSize_Device()const;
		void releaseVideo();

private:
        int width;
        int height;
		//parameter for all points
		float4*	InputND_Device;
		rgb*	spColor_Device;
        float4* MergedClusterND_Device;
		float4* MergedClusterND_Host;
		float3* MergedClusterCenters_Device;
		float3* MergedClusterCenters_Host;
		int*	MergedClusterLabel_Device;
		int*	MergedClusterLabel_Host;
		float*  MergedClusterVariance_Device;
		float*  MergedClusterVariance_Host;
        //for calculation
		int *ref, *ref_host;
		int* merged_cluster_size;
		float3* sum_of_merged_cluster_normals;
		float3* sum_of_merged_cluster_centers;
        
		//for visualization
        cv::Mat_<cv::Vec3b> show;
		cv::Mat_<cv::Vec3b> normalImage;
        cv::VideoWriter	Writer;
		unsigned char** color_pool;
		void postProcess(int* cluster_label_device, float3* cluster_center_device);
                     
};
#endif