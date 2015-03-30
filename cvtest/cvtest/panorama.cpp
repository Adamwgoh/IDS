// cvtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
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

std::vector<cv::Mat> testdft(cv::Mat* img);
void reMap2(cv::Mat& src, cv::Mat& dst);
cv::Mat inverse_dft(cv::Mat* complex_img);
cv::Mat stitch(cv::Mat img1, cv::Mat img2, int x, int y);
void multipeakstitch(cv::Mat img1, cv::Mat img2);
std::pair<std::pair<int,int>,float> Norm_CrossCorr(cv::Mat imgleft, cv::Mat imgright, double wcoverage, double hcoverage);

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
		//printf("average_val : %f\n", average_val);
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
float calcNCC(cv::Mat reference, cv::Mat target, int offx=0, int offy=0, double wcoverage=1.0, double hcoverage=1.0){
	//this assumes both images are of the same size, as the system uses the same sensor capturing time-series images
	//The offset of the target image from the reference image
	//the referencing image
	cv::Mat ref = reference;//I1
	//the target image
	cv::Mat targ = target;//I0
	if(ref.channels() != targ.channels()){
		std::cout << "two images have different number of channels!\n" << std::endl;
		return 0;
	}

	int startx = ref.cols - (ref.cols*wcoverage);
	int starty = ref.rows - (ref.rows*hcoverage);
	int offsetx = offx;
	int offsety = offy;

	float ref_val, targ_val;
	float corr_val, norm_corr_val1, norm_corr_val2, ref_nval, targ_nval;
	
	//tries and calculate the mean of the overlapping patch of image
	if(ref.channels() == 1){
		 //calculate mean of each images
		//mean for target image, the image that moves, mean of whole image
		for( int j = 0; j < ((targ.cols-offx-startx)); j++ ){
			for(int i = 0; i < (targ.rows-offy-starty); i++){
				targ_nval += (int) targ.at<uchar>(i,j);
			}
		}

		 //printf("targ_nval : %f, N is %f",targ_nval, (double) ((targ.cols-offx)*(targ.rows-offy)) );

		 //^I = sumof I0(x)/N
		 targ_nval /= ((targ.cols-offx-startx)*(targ.rows-offy-starty));

		//mean for reference image, the image that stay stationary, mean of the overlayyed image
		 for(int j = startx + offx; j < ref.cols; j++){
			 for(int i = starty + offy; i < ref.rows; i++){
				 //ref is the one only calculating the area of overlapping		 
				 ref_nval += (int) ref.at<uchar>(i,j);
			 }
		 }	

		 ref_nval /= ((ref.cols-startx-offx)*(ref.rows-starty-offy));
		
	}

	 

	 //calculate Ecc in that window with the given image mean
	 if(ref.channels() == 1){
		 
		 for(int j = startx + offsety; j < ref.cols; j++){
			 for(int i = startx + offsetx; i < ref.rows; i++){
				corr_val += ((int) ref.at<uchar>(i,j) - ref_nval)*((int) targ.at<uchar>(i,j) - targ_nval);
				norm_corr_val1 += ( (double) ((int) ref.at<uchar>(i,j) - ref_nval) );
				norm_corr_val2 += ( (double) ((int) targ.at<uchar>(i,j) - targ_nval) );
			 }
		 }
	 }

	//printf("corr val : %f, norm_corr_val1 : %f, norm_corr_val2 : %f\n", corr_val, norm_corr_val1, norm_corr_val2);
	 
	 float Ecc = corr_val / std::sqrt( std::pow(norm_corr_val1,2)*std::pow(norm_corr_val2, 2) );
	//printf("at offset_coord (%d,%d), Ecc : %f\n", offsetx, offsety, Ecc);
	if(Ecc > 1 || Ecc < -1){
		printf("Ecc %d\n", Ecc);
		cv::waitKey(0);
	}

	//return value should be in the range of [-1,1]
	return Ecc;
}

/**
*	Calculates normalized cross-correlation between two image with the given window coverage in percentage,
*	that is, the value must be between 0 to 100
*	Output is the highest cross-correlation value within the given window.
**/
std::pair<std::pair<int,int>,float> Norm_CrossCorr(cv::Mat imgleft, cv::Mat imgright, double wcoverage, double hcoverage){
	std::vector<float> corr_list = std::vector<float>();
	std::vector<std::pair<int,int>> coords = std::vector<std::pair<int,int>>();
	std::pair<int,int> coord = std::pair<int,int>();

	std::vector<std::pair<std::pair<int,int>,float>> result = std::vector<std::pair<std::pair<int,int>,float> >();
	//std::vector<std::pair<std::pair<int,int>,float>> peaks = std::vector<std::pair<std::pair<int,int>,float> >();
	float corr = 0;
	int finalx, finaly = 0;
	//calculate corr based on the percentage of coverage
	printf("wcoverage : %d, hcoverage : %d\n", (int) (((double) imgleft.cols)*wcoverage), (int) (((double) imgleft.rows)*hcoverage));
	int coverx = (int) imgleft.cols - (((double) imgleft.cols)*wcoverage);
	int covery = (int) imgleft.rows - (((double) imgleft.rows)*hcoverage);
	
	for(int offsety = covery; offsety < (imgleft.rows); offsety++){
		for(int offsetx = coverx; offsetx < (imgleft.cols); offsetx++){
			float corrval = calcNCC(imgleft, imgright, offsetx, offsety, wcoverage, hcoverage);
			if(corrval < -1 || corrval > 1){
				
				printf("coorval : %f, offsetx : %d, offsety : %d\n", corrval, offsetx, offsety);
			}
			//printf("current coord (%d,%d), corrval : %f\n", offsetx, offsety, corrval);

			coord = std::make_pair(offsetx,offsety);
			result.push_back(std::make_pair(coord, corrval));

		}
	}

	//look for the highest peak and return it
	for(int i = 0; i < result.size(); i++){
		if(corr < result[i].second){
			finalx = result[i].first.first;
			finaly = result[i].first.second;
			corr = result[i].second;
		}
	}
	printf(" biggest corr : %f, coord (%d,%d)\n", corr, finalx, finaly);
	
	return std::make_pair(std::make_pair(finalx,finaly),corr);
}

/**
 * Remaps the image to be in the size of power of 2. This is important as Danielson-Lanczos Lemma needs matrices with size in the power of 2
 * If the input image size is not in the power of 2, zeros-padding will be added to the size of the nearest power of 2
 */
cv::Mat reMap(cv::Mat img){
	cv::Mat image;
	int rows = img.rows;
	int cols = img.cols;
	int newrows = 0;
	int newcols = 0;
	

	if((rows & (rows -1) ) != 0){///bitwise element addition not equals 0, which is not of ^2
		//convert it to the nearest power of 2 size
		newrows = std::pow(2, ceil(log((float) rows)/log(2.0)) );
	}
	
	if((cols & (cols-1)) != 0){
		//convert it to the nearest power of 2 size
		newcols = std::pow(2, ceil(log((float) cols)/log(2.0)) );
	}

	if(newrows == 0){	newrows = rows;	}
	if(newcols == 0){	newcols = cols;	}
	
		if(img.channels() ==1){
			image = *new cv::Mat(newrows, newcols, CV_8UC1 );	
		}else if(img.channels() == 3){
			cv::Mat image(newrows, newcols, CV_8UC1);
		}

	for(int j = 0; j < newcols; j++){
		for(int i = 0; i < newrows; i++){
			if(image.channels() == 1){
				//printf("i : %d, j : %d\n", i,j);
				if(i < img.rows && j < img.cols){
					image.at<uchar>(i,j) = (uchar) img.at<uchar>(i,j);
				}else{
					image.at<uchar>(i,j) = (uchar) 0;
				}
				
			}else{
				if(i < img.rows && j < img.cols){
					image.at<cv::Vec3b>(i,j) = img.at<cv::Vec3b>(i,j);
				}else{
					image.at<cv::Vec3b>(i,j)[0] = 0;
					image.at<cv::Vec3b>(i,j)[1] = 0;
					image.at<cv::Vec3b>(i,j)[2] = 0;
				}
			}
		}//end for inner loop
	}//end for loop

	return image;
}

/**
 * receives an image and do bit reversal on all its values for both dimensions.
 * This refers to the excerpt from Numerical Recipes in C where the splitting of the term is.
 * By doing a right bit reversion, the terms are actually broken down into individual terms that can be applied FFT on.
 * 
 **/
std::vector<uchar>* bitReversal(std::vector<uchar> *data ){
	
	int j = 1;
	for(int i = 1; i < data->size(); i++){
		if(j > i){
			//swap
			std::swap(data->at(j), data->at(i));
		}
		int m = data->size()/2;
		while(j > m){
			j -= m;
			m /= 2;
		}
		j += m;
		
	}

	return data;
}

std::vector<uchar>* FFT(std::vector<uchar> *data, bool inverse){
	
	float two_pi = CV_PI*2;
	int mmax = 1;
	int N = data->size();
	double theta;
	double wtemp;
	double wpr;
	double wpi;
	double wr;
	double wi;
	int istep;
	//mmax is the number of times the calculation should be at maximum,
	//once the maximum amount of times to calculate goes over the data size, it stops
	while( N > mmax){//nof point transform	
		//euler's angle value (2Pi/N)
		if(inverse){
			theta = two_pi/mmax;	
		}else{
			theta = -(two_pi/mmax);	
		}
		//next number of splits and calculations
		istep = mmax << 1;
		wtemp = std::sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi = std::sin(theta);
		wr = 1.0;
		wi = 0.0;
		//half of theta


		//printf("Outer loop : mmax : %d, istep : %d\n", mmax, istep);
		//internal loop
		for(int m = 0; m < mmax; m++){//size of point transform
			//printf("	inner loop1 : m : %d, weight = %f\n", m, wr);
			//cv::waitKey(0);
			//get the Es and Os from the data. 
			for(int i = m; i < N-mmax; i+= istep){
				int j = i + mmax;
				//printf("		innerloop1 : i : %d, j = %d\n", i,j);
				double tempreal = wr*data->at(j) - wi*0;
				double tempimag = wr*0 + wi*data->at(j);
				data->at(j) = (uchar) (data->at(i) - tempreal);
				data->at(i) += (uchar) tempreal;
				//printf("data at j : %d, data at i : %d\n", data->at(j), data->at(i));
			}
			wr = (wtemp=wr)*wpr -wi*wpi + wr;
			wi = wi*wpr+wtemp*wpi+wi;
		}
		mmax = istep;

	}//end while
	
	if(inverse){
		for(int i = 0; i < istep; i++){
			data->at(i) /= istep;
			
		}
	}

	return data;
}


/**
 *	Perform FFT on a matrix
 **/
cv::Mat FFT(cv::Mat* img, int width, int height, bool inverse){//data, cols(640), rows(480), false
	//get rows and cols out of Mat and put in these arrays
	cv::Mat raw;
	img->copyTo(raw);
	cv::Mat* data = &raw;
	std::vector<uchar>* row = new std::vector<uchar>(width);
	std::vector<uchar>* col = new std::vector<uchar>(height);

	//perform fft on each row can stuff result back into the matrix
	for(int j = 0; j < height; j++){//480
		for(int x = 0; x < width; x++){//640
			row->at(x)  = (uchar) data->at<uchar>(j,x);
	
		}

		bitReversal(row);
		std::vector<uchar>* temp_row = FFT(row, false);
		for(int x = 0; x < width; x++){
			data->at<uchar>(j,x) = (uchar) temp_row->at(x);
		}
	}

	//perform fft on each col can stuff result back into the matrix
	for(int i = 0; i < width; i++){//640
		//printf("i : %d\n", i);
		for(int y = 0; y < height; y++){//480
			col->at(y)  = (uchar) data->at<uchar>(y,i);
		}

		bitReversal(col);
		//printf("size of col %d\n", sizeof(col));
		std::vector<uchar>* temp_col = FFT(col, false);
		for(int j = 0; j < height; j++){	
			data->at<uchar>(j,i) = (uchar) temp_col->at(j);
		}
	}

	return *data;
} 


int main(int argc, TCHAR* argv[])
{
	clock_t timer;
	timer = clock();
	//target image is the one not moving
	cv::Mat img1 = cv::imread("rawdata\\test\\left_testframe.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat dept1 = cv::imread("rawdata\\setthree_with_markers\\4depthframe.jpg");
	//the reference image that is checked for a certain patch
	cv::Mat img2 = cv::imread("rawdata\\test\\right_testframe.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	cv::GaussianBlur(img1, img1, cv::Size(21,21), 0, 0);
	cv::GaussianBlur(img2, img2, cv::Size(21,21), 0,0);

	cv::Mat* target = &img2;
	cv::Mat* ref = &img1;
	//multipeakstitch(img1,img2);
	std::pair<std::pair<int,int>, float> result = Norm_CrossCorr(*ref,*target,1.0,1.0);
	cv::Mat corresult = stitch(*ref,*target, result.first.first, result.first.second);
	std::ostringstream stream;
	stream << "rawdata\\result\\";
	stream << "nccstitch.jpg";
	cv::String filename = stream.str();
	cv::imwrite(filename, corresult);
	timer = clock() - timer;
	printf("Total program runtime %f\n",((float) timer/CLOCKS_PER_SEC) );
	cv::waitKey(0);
}

void multipeakstitch(cv::Mat img1, cv::Mat img2){
	cv::Mat ref,target;
	img1.copyTo(ref);
	img2.copyTo(target);
	cv::Mat* refimg = &ref;
	cv::Mat* targimg = &target;
	std::vector<cv::Mat> results1 = testdft(refimg);
	std::vector<cv::Mat> results2 = testdft(targimg);
	//get the multiplication of two
	cv::Mat convo = results1[0].mul(results2[0]);
	cv::Mat result = inverse_dft(&convo);
	
	cv::imshow("result", result);
	cv::waitKey(30);

	//look for the offset with highest value. The values are now cross-correlation coefficients
	std::vector<double> corrs;
	std::vector<int> xs, ys;
	double corr;
	int x,y;
	for(int j = 0; j < result.rows; j++){
		for(int i = 0; i < result.cols; i++){

			if(corr < result.at<double>(j,i)){
				xs.push_back(i); ys.push_back(j);
				corrs.push_back(result.at<double>(j,i));
				x = i; y = j;
				//printf("val at (j,i) = %f\n", result.at<double>(j,i));
				corr = result.at<double>(j,i);

			}
		}
	}
	printf("total peaks : %d\n", corrs.size());
	printf("highest corr val : %f, x : %d, y : %d\n", corr, x, y);
	stitch(img1,img2, x,y);
	for(int i = corrs.size()-1; i > corrs.size()-7; i--){
		//get the top 6 peaks
		cv::Mat result = stitch(img1,img2, xs.at(i), ys.at(i));

		std::ostringstream stream;
		stream << "rawdata\\result\\";
		stream << i;
		stream << "dftstitch.jpg";
		cv::String filename = stream.str();
		cv::imwrite(filename, result);

	}
}

cv::Mat stitch(cv::Mat img1, cv::Mat img2, int x, int y){
	cv::Mat final(cv::Size(img1.cols + img2.cols - (img1.cols-x), img1.rows + img2.rows - (img1.rows-x)),CV_8UC1);
	cv::Mat roi1(final, cv::Rect(0, 0,  img2.cols, img2.rows));
	cv::Mat roi2(final, cv::Rect(x, y, img1.cols, img1.rows));
	img1.copyTo(roi1);
	img2.copyTo(roi2);

	imshow("stitch", final);
	cv::waitKey(0);
	
	return final;
}

cv::Mat inverse_dft(cv::Mat* complex_img){

	//calculating the idft
	cv::Mat inverse;
	std::vector<cv::Mat> splitImage;
	cv::dft(*complex_img, inverse, cv::DFT_INVERSE);
	cv::split(inverse, splitImage);
	inverse = splitImage[0];
	normalize(inverse, inverse, 0, 1, CV_MINMAX);

	return inverse;
}

/**
 * returns the dft result and the magnitude image in a vector
 * 
 **/
std::vector<cv::Mat> testdft(cv::Mat* img){

	cv::Mat src, result, magnitude;
	cv::Mat real_image, imag_image, complex_image;
	int src_cols, src_rows, remap_cols, remap_rows;
	std::vector<cv::Mat> splitImage;
	std::vector<cv::Mat> outputs;
	img->copyTo(src);

	remap_rows = cv::getOptimalDFTSize( src.rows);
	remap_cols = cv::getOptimalDFTSize( src.cols);
	//add padding to make it power of 2. additional rows and cols are filled with zeros
	//cv::copyMakeBorder(src, real_image, 0, remap_rows - src.rows, 0, remap_cols - src.cols,
	//	IPL_BORDER_CONSTANT, cv::Scalar::all(0));
	img->convertTo(real_image,CV_64F);
	imag_image = *new cv::Mat(real_image.size(), CV_64F);
	imag_image.zeros(real_image.size(), CV_64F);
	//combine the real and imaginary part together into a single Mat
	splitImage.push_back(real_image);
	splitImage.push_back(imag_image);

	cv::Mat dst;
	cv::merge(splitImage, complex_image);
	//perform dft on the complex image on the ones without zeros border
	//roi can be edited to do offsets of convolution later
	magnitude = *new cv::Mat(remap_rows, remap_cols, CV_64FC2);
	magnitude.zeros(remap_rows, remap_cols, CV_64FC2);
	cv::Mat roi(magnitude, cv::Rect(0, 0, src.cols, src.rows));
	complex_image.copyTo(roi);

	cv::dft(magnitude, dst);
	outputs.push_back(dst);
	//split the complex image into real, imaginary parts 
	cv::Mat magI;
	cv::split(dst, splitImage);

	cv::magnitude(splitImage[0], splitImage[1], magI);

	//log and normalize to reduce the range of values so that they are visible
	log(magI + 1, magI);
	reMap2(magI, magI);
	normalize(magI, magI, 0, 1, CV_MINMAX);
	outputs.push_back(magI);

	return outputs;
}

/**
 * Rearranges the quadrants where the first and third are swap, as well as the second and fourth swapped
 */
void reMap2(cv::Mat& src, cv::Mat& dst){

	int cx = src.cols/2;
	int cy = src.rows/2;

	cv::Mat q0(src, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(src, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(dst, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(dst, cv::Rect(cx, cy, cx, cy));

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);


	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

}