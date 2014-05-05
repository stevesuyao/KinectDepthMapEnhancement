#ifndef LABEL_EQUIVALENCE_SEG_PCA_H
#define LABEL_EQUIVALENCE_SEG_PCA_H
#include <opencv2\opencv.hpp>
#include <cuda_runtime.h>


class LabelEquivalenceSegPCA{
public:
		LabelEquivalenceSegPCA(int width, int height);
        ~LabelEquivalenceSegPCA();

		void labelImage(float4* cluster_nd_device, int* cluster_label_device, float3* cluster_center_device, float* eigenvalues_device);
        void viewSegmentResult();

        float4* getMergedClusterND_Device()const;
		int* getMergedClusterLabel_Device()const;
		float4* getMergedClusterND_Host()const;
		int* getMergedClusterLabel_Host()const;
		float* getMergedClusterEigenvalues_Device()const;
		cv::Mat_<cv::Vec3b> getSegmentResult();
		void releaseVideo();

private:
        int width;
        int height;
		//parameter for all points
		float4*	InputND_Device;
        float4* MergedClusterND_Device;
		float4* MergedClusterND_Host;
		int*	MergedClusterLabel_Device;
		int*	MergedClusterLabel_Host;
        int *ref, *ref_host;
		float*	MergedClusterEigenvalues_Device;
        //else
		int* merged_cluster_size;
		float3* sum_of_merged_cluster_normals;
		float3* sum_of_merged_cluster_centers;
        float* sum_of_merged_cluster_eigenvalues;
		//for visualization
        cv::Mat_<cv::Vec3b> show;
        cv::VideoWriter	Writer;
		unsigned char** color_pool;
		void postProcess(int* cluster_label_device, float3* cluster_center_device, float* eigenvalues_device);
                     
};
#endif