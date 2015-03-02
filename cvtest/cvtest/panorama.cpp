// cvtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <stitching\stitcher.hpp>

int main(int argc, TCHAR* argv[])
{
	
	std::ostringstream ss;
	cv::Mat frame1, frame2;
	std::vector<cv::Mat> frames;
	std::vector<cv::Rect> rois;
	cv::Mat pano;
	//for(int i = 0; i < 25; i++){
		int i = 1;
		ss.clear();
		ss << "..\\..\\clouds\\";
		ss << i;
		ss << "dframe.jpg";
		cv::String url = ss.str();
		printf("%s\n", url);
		cv::waitKey(0);
		cv::Mat frame = cv::imread(url);
		
		frames.push_back(frame);
	//}

	cv::waitKey(30);
	cv::Stitcher stitch = cv::Stitcher::createDefault(false);
	stitch.stitch(frames, pano);
	cv::imshow("panorama", pano);
	cv::waitKey(0);

	//frame1 = cv::imread("..\\panorama_image1.jpg");
	//frame2 = cv::imread("..\\panorama_image2.jpg");
	//cv::imshow("frame1",frame1);
	//cv::imshow("frame2", frame2);
	//cv::Rect rect = cvRect(frame1.cols/2, 0, frame1.cols/2, frame1.rows); //second half of the first image
	//cv::Rect rect2 = cvRect(0, 0, frame2.cols/2, frame2.rows); //first half of the second image

	//rois.push_back(rect);
	//rois.push_back(rect2);
	//frames.push_back(frame1);
	//frames.push_back(frame2);

	//cv::waitKey(5000);
	////printf("frame1 : %d, frame2 : %d, pano : %d\n", frame1.channels(), frame2.channels(), pano.channels());
	//cv::Stitcher stitch = cv::Stitcher::createDefault(false);
	//stitch.stitch(frames, pano);
	//cv::imshow("panorama",pano);

	//cv::waitKey(1000000000);
	
}

