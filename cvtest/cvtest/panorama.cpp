// cvtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <stitching\stitcher.hpp>
#include <sstream>

cv::Mat getHistogram(cv::Mat data){
	cv::Mat rawdata = data;
	cv::Mat* graph = new cv::Mat(rawdata.rows, rawdata.cols, CV_8UC1);
	float average_val = 0;
	for (int row = 0; row < rawdata.rows; row++){
		for(int col = 0; col < rawdata.cols; col++){
			//printf("data val is %d\n", rawdata.data[row + (col)*(rawdata.rows)]);
			average_val += rawdata.data[row + (col)*(rawdata.rows)];
			//printf("curr row : %d, curr col : %d, current average val : %f\n", row, col, average_val);
		}

		average_val /= rawdata.rows;
		printf("average_val : %f\n", average_val);
		graph->data[row + ((int) average_val*rawdata.rows)] = 255;
	}

	return *graph;
}

int main(int argc, TCHAR* argv[])
{
	

	cv::Mat frame1, frame2;
	std::vector<cv::Mat> frames;
	std::vector<cv::Rect> rois;
	std::ostringstream ss;
	//cv::Mat pano = *new cv::Mat();
	//for(int i = 1; i < 3; i++){
	//	//int i = 1;
	//	
	//	ss << "rawdata\\";
	//	ss << i;
	//	ss << "dframe.jpg";
	//	cv::String url = "";
	//	url = ss.str();
	//	printf("url : %s", url.c_str());
	//	cv::waitKey(0);
	//	cv::Mat frame = *new cv::Mat();
	//	frame = cv::imread(url);
	//	
	//	printf("Frame width : %d, height : %d\n", frame.rows, frame.cols);
	//	frames.push_back(frame);
	//	url = "";
	//	ss.clear();
	//	ss.str("");
	//}

	//cv::waitKey(30);
	//cv::Stitcher stitch = cv::Stitcher::createDefault(false);
	//stitch.stitch(frames, pano);

	//	cv::imshow("panorama", pano);
	frame1 = cv::imread("rawdata\\15dframe.jpg");
	frame2 = getHistogram(frame1);
	cv::imshow("histogram", frame2);
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

