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
	std::pair<cv::Rect,cv::Rect> getMarkers(cv::Mat marker, cv::Mat frame);
	
	cv::Mat getHistogram(cv::Mat src);
	cv::Mat getHueHistogram(cv::Mat Huesrc);
	cv::Mat DilateRegionFilling(cv::Mat img);

	cv::Mat displayDepthGraph(cv::Mat data);
	cv::Mat displayColor(cv::Vec3b colorval);

	std::vector<cv::Mat> findDx(cv::Mat src, cv::Mat graph, int window_width);
	std::vector<cv::Vec4i> HoughLines(cv::Mat src);
	double compareColours(cv::Vec3b L_lastcolour,cv::Vec3b R_firstcolour);
	void reduceColor64(cv::Mat& src, cv::Mat& dst);
	std::vector<cv::Vec3b> GetClassesColor(cv::Mat src, int k, int iteration);
	cv::Mat ColorClusteredImg(cv::Mat src, int k, int iteration);


//private functions
private:
	double ColorProcessor::calcStandardDeviation(cv::Mat src, int window_width, int startx);
	std::vector<cv::KeyPoint> getInterestPoints(cv::Mat src);
	cv::Rect MatchingKeypoints(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2);
	cv::Rect getInterestObject(cv::Mat object, cv::Mat scene);

//public variables
public:

//private variables
private:

};