#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <sstream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/common/common_headers.h>
#include <DepthSense.hxx>

#include <utility>
#include <iostream>

class ImageRegistration{
	//public function
public :
	void multipeakstitch(cv::Mat image1, cv::Mat image2);
	cv::Mat inverse_dft(cv::Mat* complex_src);
	std::vector<cv::Mat> dft(cv::Mat* img);
	cv::Mat stitch(cv::Mat img1, cv::Mat img2, int stitchx, int stitchy);
	double getCrossVal();
	double ImageRegistration::calcCrossVal(cv::Mat img1, cv::Mat img2, int offx, int offy, cv::Size window);
	std::pair<int,int> getOffset();
	std::pair<std::pair<int,int>,double> Norm_CrossCorr(cv::Mat L_src, cv::Mat R_src, double startx, double starty, cv::Size window);
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
	double ncc_val;
	std::pair<int,int>	offset_coord;
	std::vector<cv::Mat> complex_img;
	cv::Mat mag_img;
	

};