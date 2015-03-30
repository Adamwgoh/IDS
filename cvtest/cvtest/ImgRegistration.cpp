#include "ImgRegistration.h"
#include "depthPlaneDetector.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

/**
 * Uses DFT with correlation theorem of Fourier in attempt to find for correlation score
 **/
void ImageRegistration::multipeakstitch(cv::Mat image1, cv::Mat image2){
		cv::Mat ref,target;
	image1.copyTo(ref);
	image2.copyTo(target);
	cv::Mat* refimg = &ref;
	cv::Mat* targimg = &target;
	std::vector<cv::Mat> results1 = dft(refimg);
	std::vector<cv::Mat> results2 = dft(targimg);
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
	stitch(image1,image2, x,y);
	for(int i = corrs.size()-1; i > corrs.size()-7; i--){
		//get the top 6 peaks
		cv::Mat result = stitch(image1,image2, xs.at(i), ys.at(i));

		std::ostringstream stream;
		stream << "rawdata\\result\\";
		stream << i;
		stream << "dftstitch.jpg";
		cv::String filename = stream.str();
		cv::imwrite(filename, result);

	}
}

cv::Mat ImageRegistration::inverse_dft(cv::Mat* complex_src){

	//calculating the idft
	cv::Mat inverse;
	std::vector<cv::Mat> splitImage;
	cv::dft(*complex_src, inverse, cv::DFT_INVERSE);
	cv::split(inverse, splitImage);
	inverse = splitImage[0];
	normalize(inverse, inverse, 0, 1, CV_MINMAX);

	return inverse;
}

std::vector<cv::Mat> ImageRegistration::dft(cv::Mat* img){
	cv::Mat src, result, magnitude;
	cv::Mat real_image, imag_image, complex_image;
	int src_cols, src_rows, remap_cols, remap_rows;
	std::vector<cv::Mat> splitImage;
	std::vector<cv::Mat> outputs;
	img->copyTo(src);

	remap_rows = cv::getOptimalDFTSize( src.rows);
	remap_cols = cv::getOptimalDFTSize( src.cols);
	src.convertTo(real_image,CV_64F);
	imag_image = *new cv::Mat(real_image.size(), CV_64F);
	imag_image.zeros(real_image.size(), CV_64F);
	//combine the real and imaginary part together into a single Mat
	splitImage.push_back(real_image);
	splitImage.push_back(imag_image);

	cv::Mat dst;
	//merge the real and imaginary image into a single complex image
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
 *
 **/
cv::Mat ImageRegistration::stitch(cv::Mat img1, cv::Mat img2, int stitchx, int stitchy){
	cv::Mat final(cv::Size(img1.cols + img2.cols - (img1.cols-stitchx), img1.rows + img2.rows - (img1.rows-stitchy)),CV_8UC1);
	cv::Mat roi1(final, cv::Rect(0, 0,  img2.cols, img2.rows));
	cv::Mat roi2(final, cv::Rect(stitchx, stitchy, img1.cols, img1.rows));
	img1.copyTo(roi1);
	img2.copyTo(roi2);

	imshow("stitch", final);
	
	return final;
}

/**
*	Calculates normalized cross-correlation between two image with the given window coverage in percentage,
*	that is, the value must be between 0 to 100
*	Output is the highest cross-correlation value within the given window.
**/
std::pair<std::pair<int,int>,double> ImageRegistration::Norm_CrossCorr(cv::Mat L_src, cv::Mat R_src, double startx, double starty, cv::Size window){
	std::vector<double> corr_list = std::vector<double>();
	std::vector<std::pair<int,int>> coords = std::vector<std::pair<int,int>>();
	std::pair<int,int> coord = std::pair<int,int>();

	std::vector<std::pair<std::pair<int,int>,double>> result = std::vector<std::pair<std::pair<int,int>,double> >();
	//std::vector<std::pair<std::pair<int,int>,double>> peaks = std::vector<std::pair<std::pair<int,int>,double> >();
	double corr = 0;
	int finalx, finaly = 0;
	//calculate corr based on the percentage of coverage
	printf("wcoverage : %d, hcoverage : %d\n", (int) (((double) L_src.cols)), (int) (((double) L_src.rows)));
	int coverx = 0;
	int covery = 0;
	
	for(int offsety = covery; offsety < (L_src.rows); offsety++){
		for(int offsetx = coverx; offsetx < (L_src.cols); offsetx++){
			printf("offsetx : %d, offsety : %d\n", offsetx, offsety);
			double corrval = calcCrossVal(L_src, L_src, offsetx, offsety,cv::Size(0,0));
			assert(corrval <= 1);	assert(corrval >= -1);
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

double ImageRegistration::getCrossVal(){
	return ncc_val;
}

/**
* calculates the normalized cross-correlation between the reference image and the target image.
* A single run of this function only produces one cross-correlation value for a single translation.
* Normalized cross correlation formula used is : 
* Encc = (E[I0(xi) - ^I0][I1(xi+u) - ^I1])/(Sqrt(E[I0(xi) - ^I0]^2[I1(xi) - ^I1]^2))
*
* Excerpt taken from Computer Vision: Algorithms and Applications by Richard Szeliski
**/
double ImageRegistration::calcCrossVal(cv::Mat img1, cv::Mat img2, int offx=0, int offy=0, cv::Size window=cv::Size(0,0)){
	//reference image and target image. Target is the moving window and reference is the stationary
	cv::Mat ref, targ;
	img1.copyTo(ref);
	img2.copyTo(targ);
	ref.convertTo(ref, CV_8UC1);
	targ.convertTo(targ, CV_8UC1);
	//make sure both images are of the same size and channels
	assert(img1.channels() ==1 && img2.channels() == 1);
	assert(img1.rows == img2.rows && img1.cols == img2.cols);

	int offsetx = offx;
	int offsety = offy;

	double ref_val, targ_val;
	double corr_val, norm_corr_val1, norm_corr_val2, ref_nval, targ_nval;
	double ref_variance = 0;
	double targ_variance = 0;

	//calculate mean of the window patch, the moving window is the target image
	for(int j = 0; j < targ.cols-offsetx; j++){
		for(int i = 0; i < targ.rows-offsety; i++){
	
			targ_nval += (int) targ.at<uchar>(i,j);
		}
	}

	targ_nval = targ_nval / ((targ.cols-offsetx)*(targ.rows-offsety));

	//mean for reference image, the image that stay stationary
	for(int j = offsetx; j < ref.cols; j++){
		for(int i = offsety; i < ref.rows; i++){

			ref_nval += (int) ref.at<uchar>(i,j);
		}
	}
	
	ref_nval = ref_nval / ((ref.cols-offsetx)*(ref.rows-offsety));

	//calculate Ecc with the given mean for each image
	for(int j = offsetx; j < ref.cols; j++){
		for(int i = offsety; i < ref.rows; i++){

			int ref_val = (int) ref.at<uchar>(i,j);
			int targ_val = (int) targ.at<uchar>(i-offsety, j-offsetx);
			ref_variance +=  ((double) ref_val - ref_nval);
			targ_variance += ((double) targ_val - targ_nval);
		}
	}

	corr_val = ref_variance*targ_variance;
	//printf("corr_val : %f\n", corr_val);
	//printf("denom : %f\n", std::sqrt( std::pow(norm_corr_val1,2)*std::pow(norm_corr_val2,2) ));
	double denom = std::sqrt( std::pow(ref_variance,2)*std::pow(targ_variance,2) );

	//value is too small
	if(denom <= 0.000000000001 && denom > 0){
		return 0;
	}else if(corr_val <= 0.000000000001 &&  corr_val > 0){
		return 0;
	}

	assert(denom != 0 && corr_val != 0);
	double Ecc = corr_val / denom;


	//printf("ref_variance %f, targ_variance : %f, corr_val : %f, denom : %f\n", ref_variance, targ_variance, corr_val, denom);
	printf("Ecc val : %f\n", Ecc);
	if(Ecc <= 1 && Ecc >= -1){
		printf("ref_variance %f, targ_variance : %f, corr_val : %f, denom : %f\n", ref_variance, targ_variance, corr_val, denom);
	}
	assert(Ecc <= 1 && Ecc >= -1);

	return Ecc;
}

	/**
	 * RANSAC Procedures
	 * 1) Create a Sample Consensus(SAC) model for detecting planes
	 * 2) Create a RANSAC algorithm, paramterized on epsilon = 3cm
	 * 3) computer best model
	 * 4) retrieve best set of inliers
	 * 5) retrieve correlated plane model coefficients
	 **/
pcl::PointCloud<pcl::PointXYZ>::Ptr ImageRegistration::RANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud){
	
	//create plane model pointer
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model
		(new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (inputcloud));

	//create RANSAC object, insert plane model, 0.03 is 3 for epsilon in RANSAC parameter
	pcl::RandomSampleConsensus<pcl::PointXYZ> sac(model, 0.03);
	printf("computing model");
	//perform segmentation
	bool result = sac.computeModel();
	printf("computer model");
	//get inlier indices
	std::vector<int> inliers = std::vector<int>();
	sac.getInliers(inliers);

	//get model coefficients
	Eigen::VectorXf coeff;
	sac.getModelCoefficients(coeff);

	//copy inliers to another pointcloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);
	//return pointcloud
	pcl::copyPointCloud<pcl::PointXYZ>(*inputcloud, inliers, *final);

	return final;
}

/**converts an Mat image into a pcl pointCloud
**/
pcl::PointCloud<pcl::PointXYZ>::Ptr ImageRegistration::cvtMat2Cloud(cv::Mat* src){
	cv::Mat img;
	src->copyTo(img);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	cloud->resize(img.cols*img.rows);
	cloud->width = img.cols;
	cloud->height = img.rows;
	cloud->is_dense = false;

	if(img.channels() == 1){
		for(int j = 0; j < img.cols; j++){
			for(int i = 0; i < img.rows; i++){
				//printf("j : %d, i : %d\n", j, i);
				pcl::PointXYZ point(j, i,(int) img.at<uchar>(i,j));

				cloud->at(j, i) = point;
			}
		}		
	}else if(img.channels() == 3){
		//TODO: conversion for RGB. For function's modularity
		assert(img.channels() == 1);
	}

	return cloud;
}

/**
 * converts a pcl pointCloud to an Mat image
 * only 8bit single channel for now
 **/
cv::Mat ImageRegistration::cvCloud2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
	cv::Mat result = cv::Mat(cloud->height, cloud->width, CV_8UC1);
	printf("result cols : %d, rows : %d\n", result.cols, result.rows);
	for(int j = 0; j < cloud->width; j++){
		for(int i = 0; i < cloud->height; i++){
			
			result.at<uchar>(i,j) = (uchar) cloud->points[i*cloud->width + j].z;
		}
	}

	return result;
}

std::pair<int,int>	ImageRegistration::getOffset(){
	return offset_coord;
}

/**
 * Rearranges the quadrants where the first and third are swap, as well as the second and fourth swapped
 */
void ImageRegistration::reMap2(cv::Mat& src, cv::Mat& dst){

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

int main(){
	clock_t timer;
	timer = clock();
	//target image is the one not moving
	cv::Mat img1 = cv::imread("rawdata\\test\\left_testframe.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat depth1 = cv::imread("rawdata\\setthree_with_markers\\4depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//the reference image that is checked for a certain patch
	cv::Mat img2 = cv::imread("rawdata\\test\\right_testframe.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	cv::GaussianBlur(img1, img1, cv::Size(21,21), 0, 0);
	cv::GaussianBlur(img2, img2, cv::Size(21,21), 0,0);
	cv::medianBlur(depth1, depth1, 3);

	cv::Mat* target = &img2;
	cv::Mat* ref = &img1;
	//cv::imshow("leftframe", img1);
	//cv::imshow("rightframme", img2);

	ImageRegistration imgreg = ImageRegistration();
	/*std::pair<std::pair<int,int>, double> result = imgreg.Norm_CrossCorr(*ref,*target,1.0,1.0, cv::Size(0,0));
	cv::Mat corresult =imgreg.stitch(*ref,*target, result.first.first, result.first.second);
	std::ostringstream stream;
	stream << "rawdata\\result\\";
	stream << "nccstitch.jpg";
	cv::String filename = stream.str();
	cv::imwrite(filename, corresult);*/
	depthPlaneDetector plane_detector(&depth1, 0, 0);
	cv::Mat depthgraph = plane_detector.displayDepthGraph(depth1, 0, 0);
	//once you have the depth visualization, convert it into a 1d pointcloud to perform ransac


	pcl::PointCloud<pcl::PointXYZ>::Ptr oridepth = imgreg.cvtMat2Cloud(depth1, 0, 0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr testcloud = imgreg.cvtMat2Cloud(&oridepth);

	//pcl::PointCloud<pcl::PointXYZ>::Ptr post_testcloud = imgreg.RANSAC(testcloud);
	pcl::visualization::CloudViewer viewer("cloud");
	viewer.showCloud(testcloud, "testcloud");
	cv::waitKey(30);
	while(!viewer.wasStopped()){
		if(cv::waitKey(0)){
			break;
		}
	}
	cv::Mat result = imgreg.cvCloud2Mat(testcloud);
	printf("result cols : %d, rows : %d\n", result.cols, result.rows);
	cv::imshow("result", result);
	cv::waitKey(30);
	
//	cv::Mat inlier_depth = plane_detector.displayDepthGraph(result, 0, 0);
	cv::imshow("oridepth", ori_depth);
	//cv::imshow("inlier_depth", inlier_depth);
	cv::waitKey(0);
	

	timer = clock() - timer;
	printf("Total program runtime %f\n",((double) timer/CLOCKS_PER_SEC) );
	cv::waitKey(0);
}