#include "Frame.h"

Frame::Frame(){
	depthdev_xs = std::vector<cv::Rect>();
	colourdev_xs = std::vector<cv::Rect>();
	colours = std::vector<cv::Vec3b>();
}

Frame::Frame(cv::Mat cimg, cv::Mat dimg, cv::Size window):
colourimg(cimg), depthimg(dimg), windowsize(window){
	
	depthdev_xs = std::vector<cv::Rect>();
	colourdev_xs = std::vector<cv::Rect>();
	colours = std::vector<cv::Vec3b>();
}

cv::Mat Frame::getColourImage(){
	return colourimg;
}

cv::Mat Frame::getDepthImage(){
	return depthimg;
}

void Frame::setColourImage(cv::Mat cimg){
	cimg.copyTo(colourimg);
}

void Frame::setDepthImage(cv::Mat dimg){
	dimg.copyTo(depthimg);
}

void Frame::storeColour(cv::Vec3b colour){
	colours.push_back(colour);
}

void Frame::storeDominantColour(cv::Vec3b colour){
	dominant_colour = colour;
}

void Frame::storeDepthdev_window(cv::Rect window){
	depthdev_xs.push_back(window);
}

void Frame::storeColourdev_window(cv::Rect window){
	colourdev_xs.push_back(window);
}

void Frame::storeColourdev_window(std::vector<cv::Rect> window){
	colourdev_xs = window;
}

void Frame::storeDepthdev_window(std::vector<cv::Rect> window){
	depthdev_xs = window;
}

void Frame::storeLeftMarker(cv::Rect roi){
	left_marker = roi;
}

void Frame::storeRightMarker(cv::Rect roi){
	right_marker = roi;
}

std::vector<cv::Vec3b> Frame::getColours(){
	
	return colours;
}

std::vector<cv::Rect> Frame::getDepthDeviations(){
	return depthdev_xs;
}

std::vector<cv::Rect> Frame::getColorDeviations(){
	return colourdev_xs;
}

cv::Vec3b Frame::getDominantColour(){
	return dominant_colour;
}

cv::Size Frame::getWindowSize(){
	return windowsize;
}

int Frame::getWindowWidth(){
	return windowsize.width;
}

cv::Rect Frame::getLeftMarker(){
	return left_marker;
}

cv::Rect Frame::getRightMarker(){
	return right_marker;
}