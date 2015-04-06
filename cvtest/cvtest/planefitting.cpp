#include "planefitting.h"
#include <cv.h>
#include <highgui.h>
#include <opencv2\stitching\stitcher.hpp>
#include <sstream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>

#include <utility>
#include <iostream>

void RANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud){
	/**
	 * RANSAC Procedures
	 * 1) Create a Sample Consensus(SAC) model for detecting planes
	 * 2) Create a RANSAC algorithm, paramterized on epsilon = 3cm
	 * 3) computer best model
	 * 4) retrieve best set of inliers
	 * 5) retrieve correlated plane model coefficients
	 **/
	//create plane model pointer
	// TODO: input is a plane model, please fix.
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model
	
		(new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (inputcloud));//input is the cloud

	//Create RANSAC object, insert plane model, 0.03 is 3 for epsilon in RANSAC parameter
	pcl::RandomSampleConsensus<pcl::PointXYZ> sac (model, 0.03);

	//perform segmentation
	bool result = sac.computeModel();

	//get inlier indices
	boost::shared_ptr<std::vector<int> > inliers (new std::vector<int>);
	sac.getInliers (*inliers);
	
	//get model coefficients
	Eigen::VectorXf coeff;
	sac.getModelCoefficients (coeff);
}

cv::Mat getHistogram(cv::Mat data){
	cv::Mat rawdata = data;
	cv::Mat* graph = new cv::Mat(rawdata.rows, rawdata.cols, CV_8UC1);
	float average_val = 0;
	for (int row = 0; row < rawdata.rows; row++){
		for(int col = 0; col < rawdata.cols; col++){
			average_val += rawdata.data[row + (col)*(rawdata.rows)];
		}

		average_val /= rawdata.rows;
		graph->data[row + ((int) average_val*rawdata.rows)] = 255;
	}

	return *graph;
}

