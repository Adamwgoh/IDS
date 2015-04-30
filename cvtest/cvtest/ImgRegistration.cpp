#include "ImgRegistration.h"

//constructor
ImageRegistration::ImageRegistration(){
	//default window size is (61,30);
	planedetector = depthPlaneDetector(61,30);
}

ImageRegistration::ImageRegistration(cv::Size winsize){
	planedetector = depthPlaneDetector(winsize.width, winsize.height);
}



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
	int neg_stitchy = stitchy;
	int pos_stitchy = stitchy;
	int roiy = 0;
	if(stitchy < 0){
		neg_stitchy = std::abs(stitchy);
		roiy = neg_stitchy;
		pos_stitchy = 0;
	}

	cv::Mat final = cv::Mat(cv::Size(img1.cols + img2.cols - (img2.cols - stitchx), img1.rows + neg_stitchy),CV_8UC1);
	cv::Mat roi1 = cv::Mat(final, cv::Rect(0, roiy,  img1.cols, img1.rows));
	cv::Mat roi2 = cv::Mat();
	img1.copyTo(roi1);

	//if one image is larget than the other, then assumed as a stitched image stitching with another frame,
	//hence offset is calculated from the right and not left.
	if(img1.cols > img2.cols && img1.rows > img2.rows){
		//check if vertical offset is upwards or downwards
		int bottom_val = img1.at<uchar>(img1.rows-1,img1.cols-1);
		int top_val = img1.at<uchar>(0,img1.cols-1);
		int prev_offy = 0;

		if(bottom_val != 205 && top_val == 205){
			//blank spaces is below the image, does not affect stitching
			prev_offy = pos_stitchy;
		}else if(bottom_val == 205 && top_val != 205){
			//blank spaces is above the image, affects stitching. Needs to be take away
			prev_offy = pos_stitchy +  (img1.rows - img2.rows);
		}

		roi2 = cv::Mat(final, cv::Rect(img1.cols - (img2.cols - stitchx),
			prev_offy, img2.cols, img2.rows));
	}else{
		roi2 = cv::Mat(final, cv::Rect(stitchx, pos_stitchy, img2.cols, img2.rows));
	}

	img2.copyTo(roi2);
	
	return final;
}

/**
*	Calculates normalized cross-correlation between two image with the given window coverage in percentage,
*	that is, the value must be between 0 to 100
*	Output is the highest cross-correlation value within the given window.
**/
std::pair<std::pair<int,int>,double> ImageRegistration::Norm_CrossCorr(cv::Mat L_src, cv::Mat R_src, int startx=0, int starty=0, cv::Size window=cv::Size(0,0)){
	
	double corr = 0;
	int finalx, finaly = 0;
	int start_offsetx = L_src.cols-1;
	int end_offsetx = 0;
	int start_offsety = 0;
	int end_offsety = L_src.rows;
	assert(startx+window.width < L_src.cols && startx+window.width < R_src.cols);
	assert(starty+window.height < L_src.rows && starty+window.height < R_src.rows);
	std::vector<double> corr_list = std::vector<double>();
	std::vector<std::pair<int,int>> coords = std::vector<std::pair<int,int>>();
	std::pair<int,int> coord = std::pair<int,int>();
	cv::Size search_window = window;
	std::vector<std::pair<std::pair<int,int>,double>> result = std::vector<std::pair<std::pair<int,int>,double> >();
	
	//search only in this window if specified
	if(search_window.height != 0 && search_window.width != 0){
		printf("wcoverage : %d, hcoverage: %d\n", search_window.width, search_window.height);
		start_offsety = -search_window.height;
		end_offsety = search_window.height;
		assert(end_offsety <= R_src.rows );
		//start_offsetx = L_src.cols-1;
		//end_offsetx = L_src.cols-search_window.width;
		start_offsetx = startx + search_window.width;
		end_offsetx = startx;
	}else{
		printf("wcoverage : %d, hcoverage : %d\n", (int) (((double) L_src.cols)), (int) (((double) L_src.rows)));
	}


	//calculate correlation
	for(int offsety = start_offsety; offsety < end_offsety; offsety++){
		for(int offsetx = start_offsetx; offsetx > end_offsetx; offsetx--){
			
			//printf("offsetx : %d, offsety : %d\n", offsetx, offsety);
			//stitch(L_src, R_src,offsetx, offsety);
			//cv::waitKey(40);
			//cv::waitKey(0);
			double corrval = calcCrossVal(L_src, R_src, offsetx, offsety,search_window);
			//printf("corrval : %f\n", corrval);
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
	//assert(offx + window.width < img1.cols && offx + window.width < img1.cols);
	//assert(offy + window.height< img2.rows && offy + window.height< img2.rows);

	int offsetx = offx;
	int offsety = offy;
	int neg_offy = 0;
	if(offsety < 0){
		neg_offy = offsety;
		offsety = 0;
	}	

	double ref_val = 0, targ_val = 0;
	double corr_val = 0;
	double ref_nval = 0, targ_nval = 0;
	double sq_ref_variance = 0, sq_targ_variance = 0;

	//calculate mean of the window patch, the moving window is the target image
	for(int j = 0; j < targ.cols-offsetx; j++){
		for(int i = 0+std::abs(neg_offy); i < targ.rows-offsety; i++){


			targ_nval += (int) targ.at<uchar>(i,j);
		}
	}
	targ_nval = targ_nval / ((targ.cols-offsetx)*(targ.rows-offsety+neg_offy));

	//mean for reference image, the image that stay stationary
	for(int j = offsetx; j < ref.cols; j++){
		for(int i = offsety; i < ref.rows+neg_offy; i++){

			ref_nval += (int) ref.at<uchar>(i,j);
		}
	}
	ref_nval = ref_nval / ((ref.cols-offsetx)*(ref.rows-offsety));

	//calculate Ecc with the given mean for each image
	for(int j = offsetx; j < ref.cols; j++){
		for(int i = offsety; i < ref.rows+neg_offy; i++){

			int ref_val = 0;
			int targ_val = 0;
			ref_val = (int) ref.at<uchar>(i,j) - ref_nval;
			targ_val = (int) targ.at<uchar>(i-offsety+std::abs(neg_offy), j-offsetx) - targ_nval;
			double reftarg_val = ref_val*targ_val;
			corr_val += reftarg_val;

			sq_ref_variance += std::pow(((double) ref_val - ref_nval), 2);
			sq_targ_variance += std::pow(((double) targ_val - targ_nval), 2);
		}
	}

	double denom = 0;
	denom = std::sqrt( sq_ref_variance*sq_targ_variance );

	//adding a 0.0 as a work around to having -0.0 as a value for double)
	if((0.0 + denom) == 0){
		//printf("denom is 0\n");
		return 0;//denom is 0. Is returning 0 correct?
	}else if((0.0 + corr_val) == 0){
		//printf("corr_val is 0\n");
		return 0;
	}
	
	assert(denom != 0 && corr_val != 0);
	double Ecc = corr_val / denom;
	//printf("Ecc val : %f\n", Ecc);
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
				//printf("val : %d\n", (int) img.at<uchar>(i, j));
				if((int)img.at<uchar>(i, j) > 5){

					// less than 5 depth is considered as noise and thrown away
					pcl::PointXYZ point(j, i,(int) img.at<uchar>(i,j));
					cloud->at(j, i) = point;
				}
			}
		}		
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

//finds the translational coordinate between two images using normalized cross correlation.
//Returns coordinate in the first image of VGA size
std::pair<int,int>	ImageRegistration::getColorOffset(cv::Mat img1, cv::Mat img2, int start_winx, int start_winy, cv::Size window_size){
	assert(img2.cols == VGA_WIDTH && img2.rows == VGA_HEIGHT);
	assert(img2.rows == VGA_HEIGHT && img1.cols == VGA_WIDTH);
	cv::Mat L_src, R_src;
	img1.copyTo(L_src);
	img2.copyTo(R_src);

	//downsample both images to half its sizes
	cv::pyrDown(L_src, L_src, cv::Size(L_src.cols/2, L_src.rows/2));
	cv::pyrDown(R_src, R_src, cv::Size(R_src.cols/2, R_src.rows/2));
	
	//blurring to smoothen any noises available
	if(L_src.cols > VGA_WIDTH && L_src.rows > VGA_HEIGHT){
	
		cv::GaussianBlur(L_src, L_src, cv::Size(17, 17), 0, 0);
	}else{
		cv::GaussianBlur(L_src, L_src, cv::Size(11, 11), 0, 0);
	}

	cv::GaussianBlur(R_src, R_src, cv::Size(11,11), 0,0);
	std::pair<int,int> cvt_offset = convertOffset(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(VGA_WIDTH/2, VGA_HEIGHT/2),start_winx, start_winy);
	std::pair<std::pair<int, int>, double> corr_values = Norm_CrossCorr(L_src, R_src, cvt_offset.first, cvt_offset.second, window_size);
	std::pair<int,int> highest_offset = corr_values.first;//highest corr value offset

	//convert to original resolution offset
	std::pair<int, int> result = convertOffset(cv::Size(L_src.cols, R_src.rows), cv::Size(VGA_WIDTH, VGA_HEIGHT), highest_offset.first, highest_offset.second);
	
	printf("before convert coord : (%d,%d), after convert to VGA coord : (%d,%d)\n", highest_offset.first, highest_offset.second, result.first, result.second);
	return result;
}

//converts the offset coordinates in the first image to the offset coordinates in the second offset
std::pair<int, int>	ImageRegistration::convertOffset(cv::Size src, cv::Size targ, int offsetx, int offsety){
	//find scale factor between two sizes
	double width_ratio = (double) targ.width/ (double) src.width;
	double height_ratio = (double) targ.height/(double) src.height;

	int new_offx = std::floor(offsetx*width_ratio);
	int new_offy = std::floor(offsety*height_ratio);
	std::pair<int,int> new_off = std::pair<int, int>(new_offx, new_offy);

	return new_off;
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