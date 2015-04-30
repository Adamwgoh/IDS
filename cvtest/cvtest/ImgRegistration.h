#pragma once
#include "stdafx.h"
#include "depthPlaneDetector.h"

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/common/common_headers.h>

class ImageRegistration{
	//public function
public :
	//constructor
	ImageRegistration::ImageRegistration();
	ImageRegistration::ImageRegistration(cv::Size winsize);

	void multipeakstitch(cv::Mat image1, cv::Mat image2);
	cv::Mat inverse_dft(cv::Mat* complex_src);
	std::vector<cv::Mat> dft(cv::Mat* img);
	cv::Mat stitch(cv::Mat img1, cv::Mat img2, int stitchx, int stitchy);
	double getCrossVal();
	double ImageRegistration::calcCrossVal(cv::Mat img1, cv::Mat img2, int offx, int offy, cv::Size window);

	std::pair<int,int> getColorOffset(cv::Mat img1, cv::Mat img2, int start_winx, int start_winy, cv::Size window_size);
	std::pair<int, int> convertOffset(cv::Size src, cv::Size targ, int offsetx, int offsety);
	std::pair<std::pair<int,int>,double> Norm_CrossCorr(cv::Mat L_src, cv::Mat R_src, int startx, int starty, cv::Size window);

	pcl::PointCloud<pcl::PointXYZ>::Ptr RANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cvtMat2Cloud(cv::Mat* src);
	cv::Mat cvCloud2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
	
	//private function
private :
	void reMap2(cv::Mat& src, cv::Mat& dst);
	void	saveDepthFrame(const char* filename, cv::Mat* image, int framecount);
	double	calcNCC(cv::Mat ref, cv::Mat target, int offsetx, int offsety, cv::Size window);


	//public variables
public :



	//private variables
private : 
	const static int VGA_WIDTH = 640;
	const static int VGA_HEIGHT = 480;
	const static int QQVGA_WIDTH = 160;
	const static int QQVGA_HEIGHT = 120;
	double ncc_val;
	std::pair<int,int>	offset_coord;
	std::vector<cv::Mat> complex_img;
	cv::Mat mag_img;
	depthPlaneDetector planedetector;

};