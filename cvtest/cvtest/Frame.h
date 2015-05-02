#pragma once
#include "stdafx.h"

class Frame{

public:
	//constructor
	Frame::Frame();
	Frame::Frame(cv::Mat cimg, cv::Mat dimg, cv::Size window=cv::Size(61,30));

	void setColourImage(cv::Mat cimg);
	void setDepthImage(cv::Mat dimg);

	void storeColour(cv::Vec3b colour);	
	void setLeftColour(cv::Vec3b colour);
	void setRightColour(cv::Vec3b colour);
	
	void storeDepthdev_window(cv::Rect window);
	void storeColourdev_window(cv::Rect window);
	void storeColourdev_window(std::vector<cv::Rect> window_x);
	void storeDepthdev_window(std::vector<cv::Rect> window_x);
	void storeDominantColour(cv::Vec3b colour);
	void storeLeftMarker(cv::Rect roi);
	void storeRightMarker(cv::Rect roi);

	std::vector<cv::Vec3b> getColours();
	std::vector<cv::Rect> getDepthDeviations();
	std::vector<cv::Rect> getColorDeviations();

	int getSequenceVal();
	int getWindowWidth();
	cv::Size getWindowSize();
	cv::Mat getColourImage();
	cv::Mat getDepthImage();
	cv::Rect getLeftMarker();
	cv::Rect getRightMarker();
	cv::Vec3b getDominantColour();
private:

public:
	cv::Mat colourimg;
	cv::Mat depthimg;
private:
	cv::Rect left_marker, right_marker;
	int sequence;
	cv::Size windowsize;
	cv::Vec3b leftcolour, rightcolour;
	cv::Vec3b dominant_colour;

	std::vector<cv::Vec3b> colours;
	std::vector<cv::Rect> depthdev_xs;
	std::vector<cv::Rect> colourdev_xs;

};