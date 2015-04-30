#pragma once

#include "stdafx.h"
#include "opencv2/features2d/features2d.hpp";
#include "opencv2/nonfree/features2d.hpp";
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"


class ColorProcessor{
//public functions
public:
	//constructor
	ColorProcessor::ColorProcessor();

	cv::Mat RGB2HSV(cv::Mat RGBsrc);

	cv::Mat findInterestPoints(cv::Mat src);
	cv::Mat getHueImage(cv::Mat rgbsrc);
	cv::Mat getValueImage(cv::Mat hsvsrc);
	cv::Mat getSaturationImage(cv::Mat hsvsrc);

	std::vector<cv::KeyPoint> getInterestPoints(cv::Mat src);
	cv::Mat MatchingKeypoints(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2);

	cv::Mat getHueHistogram(cv::Mat Huesrc);
	cv::Mat DilateRegionFilling(cv::Mat img);

	cv::Mat displayDepthGraph(cv::Mat data);
	std::vector<cv::Mat> findDx(cv::Mat graph, int window_width);

//private functions
private:
	double ColorProcessor::calcStandardDeviation(cv::Mat src, int window_width, int startx);
//public variables
public:

//private variables
private:

};