#include "SuperpixelSegmentation.h"
#include <ctime>

SuperpixelSegmentation::SuperpixelSegmentation(int width, int height):
	SegmentedRandomColor(height, width),
	SegmentedColor(height, width){
		this->width = width;
		this->height = height;
		SegmentedWriter = cv::VideoWriter::VideoWriter("SegmentedRandom.avi", CV_FOURCC('X','V','I','D'), 3.0, cv::Size(width, height));
		cudaMallocHost(&Labels_Host, sizeof(int)*width*height);
		cudaMalloc(&Labels_Device, sizeof(int)*width*height);
		cudaMalloc(&LD_Device, sizeof(label_distance)*width*height);
}
SuperpixelSegmentation::~SuperpixelSegmentation(){
	cudaFree(Labels_Host);
	cudaFree(Labels_Device);
	cudaFree(meanData_Host);
	cudaFree(meanData_Device);
	cudaFree(LD_Device);
	delete [] RandomColors;
}
void SuperpixelSegmentation::releaseVideo(){
	SegmentedWriter.release();
}
void SuperpixelSegmentation::SetParametor(int rows, int cols){
	//number of clusters
	ClusterNum.x = cols;
	ClusterNum.y = rows;
	//grid(window) size
	Window_Size.x = width/cols;
	Window_Size.y = height/rows;
	//Init GPU memory
	initMemory();						
	//Random colors
	for(int i=0; i<ClusterNum.x*ClusterNum.y; i++){
		int3 tmp;
		tmp.x = rand()%255;
		tmp.y = rand()%255;
		tmp.z = rand()%255;
		RandomColors[i] = tmp;
	}
}
void SuperpixelSegmentation::initMemory(){
	//superpixel data
	cudaMallocHost(&meanData_Host, sizeof(superpixel) * ClusterNum.x*ClusterNum.y);	
	cudaMalloc(&meanData_Device, sizeof(superpixel) * ClusterNum.x*ClusterNum.y);	
	//Random color
	RandomColors = new int3[ClusterNum.x*ClusterNum.y];
}

cv::Mat_<cv::Vec3b> SuperpixelSegmentation::getSegmentedImage(cv::Mat_<cv::Vec3b> input_host, int options){
	
	if(options == Line){
	//cudaMemcpy(Labels_Host, Labels_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	input_host.copyTo(SegmentedColor);

	for(int y=0; y<height-1; y++){
		for(int x=0; x<width-1; x++){
			if(Labels_Host[y*width+x] !=  Labels_Host[(y+1)*width+x]){
				//SegmentedColor.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
				SegmentedColor.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			}
			if(Labels_Host[y*width+x] !=  Labels_Host[y*width+x+1]){
				//SegmentedColor.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
				SegmentedColor.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			}
		}
	}

	}
	else{
	//int* size = new int[40*80];
	//int3* rgb = new int3[40*80];
	//int3* center = new int3[40*80];
	//int3 init;
	//init.x = 0;
	//init.y = 0;
	//init.z = 0;
	//for(int i=0; i<40*80; i++){
	//	size[i] = 0;
	//	rgb[i] = init;
	//	center[i] = init;
	//}
	//for(int y=0; y<height; y++){
	//	for(int x=0; x<width; x++){
	//		int id = Labels_Host[y*width+x];
	//		if(id != -1){
	//			size[id]++;
	//			rgb[id].x += (int)input_host.at<cv::Vec3b>(y, x).val[0];
	//			rgb[id].y += (int)input_host.at<cv::Vec3b>(y, x).val[1];
	//			rgb[id].z += (int)input_host.at<cv::Vec3b>(y, x).val[2];
	//			center[id].x += x;
	//			center[id].y += y;
	//		}
	//	}
	//}
	//cudaMemcpy(Labels_Host, Labels_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	cudaMemcpy(meanData_Host, meanData_Device, sizeof(superpixel)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			int id = Labels_Host[y*width+x];
			if(id != -1){
				//std::cout << id <<std::endl;
				SegmentedColor.at<cv::Vec3b>(y, x).val[0] = meanData_Host[id].r;
				SegmentedColor.at<cv::Vec3b>(y, x).val[1] = meanData_Host[id].g;
				SegmentedColor.at<cv::Vec3b>(y, x).val[2] = meanData_Host[id].b;
				//if(SegmentedColor.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0))
				//	std::cout << "id: "<< id <<std::endl;
				//SegmentedColor.at<cv::Vec3b>(y, x).val[0] = (unsigned char)(rgb[id].x/size[id]);
				//SegmentedColor.at<cv::Vec3b>(y, x).val[1] = (unsigned char)(rgb[id].y/size[id]);
				//SegmentedColor.at<cv::Vec3b>(y, x).val[2] = (unsigned char)(rgb[id].z/size[id]);
			}												
			else
				SegmentedColor.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
		}
	}
	}
	

	
	//delete [] size;
	//delete [] rgb;
	//delete [] center;
	//
	return SegmentedColor;
}

cv::Mat_<cv::Vec3b> SuperpixelSegmentation::getRandomColorImage(){
	//cudaMemcpy(Labels_Host, Labels_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	//int* size = new int[40*80];
	//int3* center = new int3[40*80];
	//
	//int3 init;
	//init.x = 0;
	//init.y = 0;
	//init.z = 0;
	//for(int i=0; i<40*80; i++){
	//	size[i] = 0;
	//	center[i] = init;
	//}
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			int id = Labels_Host[y*width+x];
			if(id != -1){
				SegmentedRandomColor.at<cv::Vec3b>(y, x).val[0] = (unsigned char)RandomColors[id].x;
				SegmentedRandomColor.at<cv::Vec3b>(y, x).val[1] = (unsigned char)RandomColors[id].y;
				SegmentedRandomColor.at<cv::Vec3b>(y, x).val[2] = (unsigned char)RandomColors[id].z;
				//size[id]++;
				//center[id].x += x;
				//center[id].y += y;
			}
			else
				SegmentedRandomColor.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
		}
	}
	//double ave = 0.0;
	cudaMemcpy(meanData_Host, meanData_Device, sizeof(superpixel)*ClusterNum.x*ClusterNum.y, cudaMemcpyDeviceToHost);
	//for(int y=0; y<height; y++){
	//	for(int x=0; x<width; x++){
	//		int id = Labels_Host[y*width+x];
	//		if(id != -1){
	//			int2 ref;
	//			if(x==meanData_Host[id].x && y== meanData_Host[id].y){
	//				for(int yy=-1; yy<=1; yy++){
	//					for(int xx=-1; xx<=1; xx++){
	//						ref.x = x+xx >= 0 ? x+xx:0;
	//						ref.x = x+xx < width ? x+xx:width;
	//						ref.y = y+yy>= 0 ? y+yy:0;
	//						ref.y = y+yy< height ? y+yy:height;
	//						SegmentedRandomColor.at<cv::Vec3b>(ref.y, ref.x) = cv::Vec3b(0, 0, 0);
	//					}
	//				}
	//			}
	//			if(x==center[id].x/size[id] && y== center[id].y/size[id]){
	//				ave += sqrt(pow((double)(meanData_Host[id].x-x),2.0)+pow((double)(meanData_Host[id].y-y),2.0));
	//				//std::cout << "CPU: (x, y): "<<x<<", "<<y<<std::endl;
	//		//std::cout << "GPU: (x, y): "<<meanData_Host[id].x<<", "<<meanData_Host[id].y<<std::endl;
	//				for(int yy=-1; yy<=1; yy++){
	//					for(int xx=-1; xx<=1; xx++){
	//						ref.x = x+xx >= 0 ? x+xx:0;
	//						ref.x = x+xx < width ? x+xx:width;
	//						ref.y = y+yy>= 0 ? y+yy:0;
	//						ref.y = y+yy< height ? y+yy:height;
	//						SegmentedRandomColor.at<cv::Vec3b>(ref.y, ref.x) = cv::Vec3b(255, 255, 255);
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
	//std::cout << "ave: "<<ave/(40*80)<<std::endl;
	//delete [] size;
	//delete [] center;
	
	//SegmentedWriter << SegmentedRandomColor;
	//cv::imshow(name, SegmentedRandomColor);
	return SegmentedRandomColor;
}
int* SuperpixelSegmentation::getLabelDevice(){
	return Labels_Device;
}
SuperpixelSegmentation::superpixel*	SuperpixelSegmentation::getMeanDataDevice(){
	return meanData_Device;
}