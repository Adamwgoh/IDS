// cvtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <opencv2\stitching\stitcher.hpp>
#include <sstream>

#include <utility>
#include <iostream>

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

/**
* calculates the normalized cross-correlation between the reference image and the target image.
* A single run of this function only produces one cross-correlation value for a single translation.
* Normalized cross correlation formula used is : 
* Encc = (E[I0(xi) - ^I0][I1(xi+u) - ^I1])/(Sqrt(E[I0(xi) - ^I0]^2[I1(xi) - ^I1]^2))
*
* Excerpt taken from Computer Vision: Algorithms and Applications by Richard Szeliski
**/
float calcNCC(cv::Mat reference, cv::Mat target, int offx, int offy){
	//this assumes both images are of the same size, as the system uses the same sensor capturing time-series images
	//The offset of the target image from the reference image
	int offsetx = offx;
	int offsety = offy;
	//the referencing image
	cv::Mat ref = reference;//I1
	//the target image
	cv::Mat targ = target;//I0
	int ref_val, targ_val;
	float corr_val, norm_corr_val, ref_nval, targ_nval;
	 
	 //calculate mean of each images
	 for(int i = 0; i < (targ.cols*targ.rows)-3; i++){
		 //target dont move, so dont need to add offset to it

		targ_nval += (targ.data[i] + targ.data[i + 1] + targ.data[i + 2])/3;
	 }
	 //^I = sumof I0(x)/N
	 targ_nval /= targ.elemSize();

	
	 for(int j = offsety; j < ref.cols; j++){
		 for(int i = offsetx; i < ref.rows; i++){
			 //ref is the one only calculating the area of overlapping		 
			 ref_nval += ((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2])/3);
		 }
	 }
	 
	 //mean of number of pixels in the patch
	// printf("lol");
	 
	 ref_nval /= (ref.cols-offsety)*(ref.rows-offsetx);
	 //printf("ref_nval = %f\n", ref_nval);
	 for(int j = offsety; j < ref.cols; j++){
		 for(int i = offsetx; i < ref.rows; i++){
			//value is converted to greyscale first before calculating its difference from the mean
			corr_val += (((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2]) /3) - ref_nval)*
				(((targ.at<cv::Vec3b>(i-offsetx,j-offsety)[0] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[1] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[2])/3) - targ_nval);
//			if(i != 0 || j != 0){
				//printf("sum %d, ref_nval %f\n",((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2]) /3), ref_nval);
//			}
			//standard deviation of both images
			norm_corr_val += (std::pow( (double) (((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2]) /3) - ref_nval),2)*
				(std::pow( (double) (((targ.at<cv::Vec3b>(i-offsetx,j-offsety)[0] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[1] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[2])/3) - targ_nval),2)) );
		 }
	 }
	 //printf("corr val : %f, norm_corr_val : %f\n", corr_val, norm_corr_val);
	float Ecc = corr_val / std::sqrt(norm_corr_val);

	//return value should be in the range of [-1,1]
	return Ecc;
}

int main(int argc, TCHAR* argv[])
{
	//target image is the one not moving
	cv::Mat target = cv::imread("rawdata\\setthree_with_markers\\3cframe.jpg");
	//the reference image that is checked for a certain patch
	cv::Mat ref = cv::imread("rawdata\\setthree_with_markers\\2cframe.jpg");
	//printf("target channels : %d, ref channels : %d\n", target.channels(), ref.channels());
	float max_val = 0;
	int max_offsetx;
	int max_offsety;
	

	for(int offy = 0; offy < ref.cols; offy++){
		for(int offx = 0; offx < ref.rows; offx++){
			float ecc = calcNCC(ref, target, offx, offy);
			printf("ecc with offset (%d,%d) is : %f\n", offx, offy, ecc);
			if(ecc > max_val){
				max_val = ecc;
				max_offsetx = offx;
				max_offsety = offy;
			}	
		}
	}

	printf("peak val is :%f\n", max_val);
	while(true){
	}

}

