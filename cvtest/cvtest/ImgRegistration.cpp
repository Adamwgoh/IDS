#include "ImgRegistration.h"
const int LEFT2RIGHT = 19;
const int RIGHT2LEFT = 91;

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
	//printf("total peaks : %d\n", corrs.size());
	//printf("highest corr val : %f, x : %d, y : %d\n", corr, x, y);
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
	cv::Mat final = cv::Mat();
	if(stitchx < img2.cols){
		final = cv::Mat(cv::Size(img1.cols + img2.cols- (img2.cols - stitchx), img1.rows + neg_stitchy),img1.type());
	}else{
		final = cv::Mat(cv::Size(img1.cols + img2.cols - (img1.cols - stitchx), img1.rows + neg_stitchy),img1.type());
	}
	//cv::Mat final = cv::Mat(cv::Size(img1.cols + img2.cols - (img2.cols - stitchx), img1.rows + neg_stitchy),img1.type());
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
		if(stitchx > img2.cols){
			
			roi2 = cv::Mat(final, cv::Rect(stitchx, pos_stitchy, img2.cols, img2.rows));
		}else{
			roi2 = cv::Mat(final, cv::Rect(img1.cols - (img2.cols - stitchx),
				prev_offy, img2.cols, img2.rows));
		}
	}else{
		roi2 = cv::Mat(final, cv::Rect(stitchx, pos_stitchy, img2.cols, img2.rows));
	}

	img2.copyTo(roi2);
	cv::imshow("final", final);
	cv::waitKey(40);
	cv::waitKey(0);
	cv::destroyAllWindows();
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
		printf("calculating normalized cross-correlation value..\n");
		start_offsety = -search_window.height;
		end_offsety = search_window.height;
		assert(end_offsety <= R_src.rows );
		//start_offsetx = L_src.cols-1;
		//end_offsetx = L_src.cols-search_window.width;
		start_offsetx = startx + search_window.width;
		end_offsetx = startx;
	}else{
		
		printf("calculating normalized cross-correlation value..\n");
	}

	//calculate correlation
	for(int offsety = start_offsety; offsety < end_offsety; offsety++){
		for(int offsetx = start_offsetx; offsetx > end_offsetx; offsetx--){
			
			printf("offsetx : %d, offsety : %d\n", offsetx, offsety);
			cv::Mat stitchroi = stitch(L_src, R_src,offsetx, offsety);
			//cv::imshow("stitchroi", stitchroi);
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


/**
*	Calculates srmalized cross-correlation between two image with the given window coverage in percentage,
*	that is, the value must be between 0 to 100
*	Output is the highest cross-correlation value within the given window.
**/
std::pair<std::pair<int,int>,double> ImageRegistration::Norm_CrossCorr2(cv::Mat L_src, cv::Mat R_src, cv::Rect left_window, cv::Rect right_window){
	
	double corr = 0;
	int finalx = 0;int finaly = 0;
	int start_offsetx= L_src.cols;
	int end_offsetx = 0;
	int start_offsety = 0;
	int end_offsety = L_src.rows;
	std::vector<double> corr_list = std::vector<double>();
	std::vector<std::pair<int,int>> coords = std::vector<std::pair<int,int>>();
	std::pair<int,int> coord = std::pair<int,int>();
	//cv::Size search_window = window;
	std::vector<std::pair<std::pair<int,int>,double>> result = std::vector<std::pair<std::pair<int,int>,double> >();

	//search only in this window if specified
	if(left_window.height != 0 && right_window.width != 0 && right_window.width != 0 && right_window.height != 0){
		printf("wcoverage : %d, hcoverage: %d\n", left_window.width, right_window.height);
		start_offsety = -left_window.height;
		end_offsety = left_window.height;
		assert(end_offsety <= R_src.rows );
		//start_offsetx = L_src.cols-1;
		//end_offsetx = L_src.cols-search_window.width;
		start_offsetx = left_window.x + left_window.width - right_window.x;
		end_offsetx = left_window.x - right_window.x;
	}else{
		printf("wcoverage : %d, hcoverage : %d\n", (int) (((double) L_src.cols)), (int) (((double) L_src.rows)));
	}

	cv::Mat right_roi = cv::Mat(R_src, right_window);
	
	//calculate correlation
	for(int offsety = start_offsety; offsety < end_offsety; offsety++){
		for(int offsetx = start_offsetx; offsetx > end_offsetx; offsetx--){
			
			//printf("offsetx : %d, offsety : %d\n", offsetx, offsety);
			//cv::Mat offset_roi = stitch(L_src, R_src,offsetx, offsety);
			//cv::imshow("offset_roi", offset_roi);
			//cv::waitKey(40);
			//cv::waitKey(0);
			double corrval = calcCrossVal(L_src, R_src, offsetx, offsety,cv::Size(left_window.width,left_window.height));
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
	img1.convertTo(ref, CV_8UC1);
	img2.convertTo(targ, CV_8UC1);
	
	cv::cvtColor(img2, targ, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img1, ref, cv::COLOR_BGR2GRAY);

	//make sure both images are of the same size and channels
	
	assert(ref.channels() ==1 && targ.channels() == 1);
	assert(ref.rows == targ.rows && ref.cols == targ.cols);
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
std::pair<int,int>	ImageRegistration::getColorOffset(cv::Mat img1, cv::Mat img2,int leftstart_winx, int leftstart_winy, int rightstart_winx,int rightstart_winy, cv::Size window_size){
	assert(img2.cols == VGA_WIDTH && img2.rows == VGA_HEIGHT);
	assert(img2.rows == VGA_HEIGHT && img2.cols == VGA_WIDTH);
	cv::Mat L_src, R_src;
	img1.copyTo(L_src);
	img2.copyTo(R_src);
	L_src.convertTo(L_src, CV_8UC1);
	R_src.convertTo(R_src, CV_8UC1);
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

	std::pair<int,int> cvt_left_offset = convertOffset(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(VGA_WIDTH/2, VGA_HEIGHT/2),leftstart_winx, leftstart_winy);
	std::pair<std::pair<int, int>, double> corr_values = Norm_CrossCorr(L_src, R_src, cvt_left_offset.first, cvt_left_offset.second, window_size);
	std::pair<int,int> highest_offset = corr_values.first;//highest corr value offset

	//convert to original resolution offset
	std::pair<int, int> result = convertOffset(cv::Size(L_src.cols, R_src.rows), cv::Size(VGA_WIDTH, VGA_HEIGHT), highest_offset.first, highest_offset.second);
	
	printf("before convert coord : (%d,%d), after convert to VGA coord : (%d,%d)\n", highest_offset.first, highest_offset.second, result.first, result.second);
	return result;
}

//finds the translational coordinate between two images using normalized cross correlation.
//Returns coordinate in the first image of VGA size
std::pair<int,int>	ImageRegistration::getColorOffset2(cv::Mat img1, cv::Mat img2,cv::Rect left_frame, cv::Rect right_frame){
	assert(img2.cols == VGA_WIDTH && img2.rows == VGA_HEIGHT);
	assert(img2.rows == VGA_HEIGHT && img1.cols == VGA_WIDTH);
	cv::Mat L_src, R_src;
	img1.copyTo(L_src);
	img2.copyTo(R_src);
	L_src.convertTo(L_src, CV_8UC1);
	R_src.convertTo(R_src, CV_8UC1);
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
	std::pair<int,int> left_VGAcoords = convertOffset(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(VGA_WIDTH/2, VGA_HEIGHT/2),left_frame.x, left_frame.y);
	std::pair<int,int> right_VGAcoords = convertOffset(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(VGA_WIDTH/2,VGA_HEIGHT/2),right_frame.x,right_frame.y);
	std::pair<int,int> left_VGASize = convertOffset(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(VGA_WIDTH/2, VGA_HEIGHT/2),left_frame.width, left_frame.height);
	std::pair<int,int> right_VGASize = convertOffset(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(VGA_WIDTH/2, VGA_HEIGHT/2),right_frame.width, right_frame.height);
	
	cv::Rect left_window = cv::Rect(left_VGAcoords.first, left_VGAcoords.second, left_VGASize.first, left_VGASize.second);
	cv::Rect right_window = cv::Rect(right_VGAcoords.first, right_VGAcoords.second, right_VGASize.first, right_VGASize.second);
	std::pair<std::pair<int,int>, double> corr_values = Norm_CrossCorr2(L_src, R_src, left_window, right_window);

	//std::pair<int,int> cvt_left_offset = convertOffset(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(VGA_WIDTH/2, VGA_HEIGHT/2),leftstart_winx, leftstart_winy);
	std::pair<int,int> highest_offset = corr_values.first;//highest corr value offset

	//convert to original resolution offset
	std::pair<int, int> result = convertOffset(cv::Size(L_src.cols, R_src.rows), cv::Size(VGA_WIDTH, VGA_HEIGHT), highest_offset.first, highest_offset.second);
	
	//printf("before convert coord : (%d,%d), after convert to VGA coord : (%d,%d)\n", highest_offset.first, highest_offset.second, result.first, result.second);
	return result;
}

//given a window size and starting coordinate, calculate the average depth value in the window
double ImageRegistration::getWindowDepthValue(cv::Mat depth_src, cv::Size windowsize, int startx, int starty){
	double average_val = 0;
	cv::Mat roi(depth_src, cv::Rect(startx, starty, windowsize.width, windowsize.height));

	//calculate the average val in it
	for(int row = 0; row < roi.rows; row++){
		for(int col = 0; col < roi.cols; col++){
			double val = roi.at<uchar>(row,col);
			average_val += val;
		}
	}

	average_val /= (roi.rows*roi.cols);

	return average_val;
}

double ImageRegistration::compareColours(cv::Vec3b L_lastcolour,cv::Vec3b R_firstcolour){
	double diff = 0;
	diff = cv::norm(L_lastcolour - R_firstcolour);

	return diff;
}

double ImageRegistration::getWindowDepthValue(cv::Mat src_roi){
	double average_val = 0;

	//calculate average val in that roi
	for(int row = 0; row < src_roi.rows; row++){
		for(int col = 0; col < src_roi.cols; col++){
			double val = src_roi.at<uchar>(row,col);
			average_val += val;
		}
	}

	average_val /= (src_roi.rows*src_roi.cols);

	return average_val;
}

//given the found depth deviations and colors found, decide what are the window area for stitching
std::pair<cv::Rect, cv::Rect> ImageRegistration::findWindowOfInterest(Frame prev_frame, Frame curr_frame){
	std::pair<cv::Rect, cv::Rect> windows;
	cv::Mat left_cimg, left_dimg, right_cimg, right_dimg;
	std::vector<cv::Vec3b> prev_colours, curr_colours;
	std::vector<cv::Rect> prev_cwinx, curr_cwinx, prev_dwinx, curr_dwinx;
	double colourdiff_thresh = 50;
	int prev_width, curr_width;

	//assign data from input to local variables
	prev_colours = prev_frame.getColours();curr_colours = curr_frame.getColours();
	left_cimg = prev_frame.getColourImage(); right_cimg = curr_frame.getColourImage();
	left_dimg = prev_frame.getDepthImage(); right_dimg = curr_frame.getDepthImage();
	prev_cwinx = prev_frame.getColorDeviations(); prev_dwinx = prev_frame.getDepthDeviations();
	curr_cwinx = curr_frame.getColorDeviations(); curr_dwinx = curr_frame.getDepthDeviations();
	prev_width = prev_frame.getWindowWidth(); curr_width = curr_frame.getWindowWidth();

	/***
	 * As deviation found are from left to right of image
	 * , first check on the last deviation of the left image
	 * and the first deviation of the right image, and if the image color matches.
	 * If the color matches, the probable stitch is between the right side of Limage and 
	 * left side of Rimage.
	 ***/
	if(prev_dwinx.size() != 0 && curr_dwinx.size() == 0){

		//there is a change in deviation in the first frame but not on the second frame
		for(int prev_dev = prev_dwinx.size(); prev_dev > 0; prev_dev--){
			//cut the window into half, and get its deviation value
			cv::Rect prev_rect = prev_dwinx.at(prev_dev);
			cv::Mat left_prevroi = cv::Mat(left_dimg, cv::Rect(prev_rect.x, prev_rect.y, prev_rect.width/2, prev_rect.height));
			cv::Mat right_prevroi = cv::Mat(left_dimg, cv::Rect(prev_rect.width/2, prev_rect.y, (prev_rect.width/2)-1, prev_rect.height));

			//get the right frame's average depth value at the left of the frame
			cv::Rect curr_rect = cv::Rect(prev_rect);//doesn't matter where, there's no huge std deviation anyway
			cv::Mat curr_roi = cv::Mat(right_dimg, curr_rect);
			double left_prevdval = getWindowDepthValue(left_prevroi);double right_prevdval = getWindowDepthValue(right_prevroi);
			double right_currdval = getWindowDepthValue(curr_roi);
			double leftdval_diff, rightdval_diff;
			leftdval_diff = std::abs(left_prevdval - right_currdval);
			rightdval_diff = std::abs(right_prevdval - right_currdval);
			
			if(leftdval_diff > rightdval_diff && (leftdval_diff-rightdval_diff)/(leftdval_diff+rightdval_diff) > 0.4){
				//rightdval is more similar, right marker will be chosen with left marker of current frame
				windows = std::pair<cv::Rect,cv::Rect>(prev_frame.getRightMarker(), curr_frame.getLeftMarker());
				return windows;
			}else{
				//leftdval is more similar, left marker will be chosen with right marker of current frame
				windows = std::pair<cv::Rect,cv::Rect>(curr_frame.getRightMarker(),prev_frame.getLeftMarker()); 
				return windows;
			}
		}
	}else if(prev_dwinx.size() == 0 && curr_dwinx.size() == 0){
		//there are no change in deviation in neither the first frame nor the second frame
		//stitch right marker of prev frame with left marker of current frame
		windows = std::pair<cv::Rect,cv::Rect>(prev_frame.getRightMarker(), curr_frame.getLeftMarker());

		return windows;
	}else{
	
		for(int prev_dev = prev_dwinx.size(); prev_dev > 0; prev_dev--){
			for(int curr_dev = 0; curr_dev < curr_cwinx.size(); curr_dev+=2){

				cv::Vec3b L_lastcolour = prev_colours.at(prev_dev);
				cv::Vec3b R_firstcolour = curr_colours.at(curr_dev);
				double colourdiff = compareColours(L_lastcolour,R_firstcolour);
				//printf("colour diff : %f\n", colourdiff);
				if(compareColours(L_lastcolour,R_firstcolour) < colourdiff_thresh){
					//no significant change of color between these two region
					//now check if the two sides are of similar depth values
					//get corresponding depth deviation windows, split them and calculate the corresponding depth values
					cv::Rect prev_window = prev_dwinx.back(); cv::Rect curr_window = curr_dwinx.back();
					//get the right side of the prev frame window
					cv::Mat prev_roi = cv::Mat(left_dimg, cv::Rect(prev_window.x + (prev_window.width/2), prev_window.y, (prev_window.width/2)-1, prev_window.height));
					cv::Mat curr_roi = cv::Mat(right_dimg, cv::Rect(curr_window.x , curr_window.y, (curr_window.width/2), curr_window.height));
	
					double prev_dval = getWindowDepthValue(prev_roi); double curr_dval = getWindowDepthValue(curr_roi);
					//printf("prev_dval %f, curr_dval : %f\n", prev_dval, curr_dval);
					double dval_diff = std::abs(prev_dval - curr_dval);
					//printf("dval_diff :%f\n", dval_diff);
					double diff_percentage = (dval_diff/(prev_dval + curr_dval)) * 100;
					if(diff_percentage < 45){
						//printf("diff percentage : %f\n", diff_percentage);
						//no significant differences between the two average depth values
						//possible stitching area found for that side
						cv::Rect left_window, right_window;

						if(prev_window.x > left_dimg.cols/2){
							left_window = prev_frame.getRightMarker();
							//left_window = std::pair<int,int>(prev_frame.getRightMarker().x, prev_frame.getRightMarker().y);
						}else if(prev_window.x < left_dimg.cols/2){
							left_window = prev_frame.getLeftMarker();
							//left_window = std::pair<int,int>(prev_frame.getLeftMarker().x, prev_frame.getLeftMarker().y);
						}

						if(curr_window.x > right_dimg.cols/2){
							right_window = curr_frame.getRightMarker();
							//right_window = std::pair<int,int>(curr_frame.getRightMarker().x, curr_frame.getRightMarker().y);
						}else if(curr_window.x < right_dimg.cols/2){
							right_window = curr_frame.getLeftMarker();
							//right_window = std::pair<int,int>(curr_frame.getRightMarker().x, curr_frame.getRightMarker().y);
						}
						windows = std::pair<cv::Rect,cv::Rect>(left_window, right_window);
						//windows = std::pair<std::pair<int, int>,std::pair<int, int>>(left_window, right_window);
			
						return windows;
					}
				}else{
					//significant changes in colour. check the second one deviation
				}
			}//end inner for
		}//end outer for
	}//end else
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

std::pair<cv::Mat,cv::Mat> ImageRegistration::StitchFrames(std::vector<cv::Mat> cframes, std::vector<cv::Mat> dframes){
	std::pair<cv::Mat, cv::Mat> result = std::pair<cv::Mat, cv::Mat>();
	assert(cframes.size() == dframes.size());
	int direction = LEFT2RIGHT; //by default left to right stitching
	cv::Mat L_cimg, R_cimg;
	cv::Mat L_dimg, R_dimg;
	cv::Mat prev_cimg, prev_dimg;
	cv::Mat prev_color, curr_color;
	Frame prev_frame, curr_frame;
	std::vector<std::pair<int,int>> corr_offsets = std::vector<std::pair<int,int>>();
	std::pair<cv::Rect, cv::Rect> window;
	std::vector<cv::Mat> cstitches = std::vector<cv::Mat>();
	std::vector<cv::Mat> dstitches = std::vector<cv::Mat>();
	depthPlaneDetector detector(61,30);
	ColorProcessor cProc = ColorProcessor();
	cv::Mat marker = cv::imread("rawdata\\marker.jpg");

	for(int i = 0; i < cframes.size()-1; i++){
		L_cimg = cframes.at(i);
		R_cimg = cframes.at(i+1);
		L_dimg = dframes.at(i);
		R_dimg = dframes.at(i+1);
		prev_dimg = dframes.at(i);
		prev_cimg = cframes.at(i);
		prev_frame = Frame(prev_cimg, prev_dimg, cv::Size(61, 30));
		curr_frame = Frame(R_cimg, R_dimg, cv::Size(61, 30));
		cv::medianBlur(L_dimg, L_dimg, 3);
		cv::medianBlur(R_dimg, R_dimg, 3);
		
		//if there's an available stitch already, take that as the reference image
		if(!cstitches.empty()){	
			L_cimg = cstitches.back();
		}		

		if(!dstitches.empty()){
			L_dimg = dstitches.back();
			cv::resize(prev_dimg, prev_dimg, cv::Size(R_cimg.cols, R_cimg.rows), CV_INTER_NN);
		}else{
			cv::resize(L_dimg, L_dimg, cv::Size(R_cimg.cols, R_cimg.rows), CV_INTER_NN);
			prev_dimg = L_dimg;
		}

		cv::resize(R_dimg, R_dimg, cv::Size(R_cimg.cols, R_cimg.rows), CV_INTER_NN);
		//get the frame's dominant colours
		cv::Vec3b leftdominant, rightdominant;
		leftdominant = cProc.GetClassesColor(prev_cimg, 2, 5).front();
		rightdominant = cProc.GetClassesColor(R_cimg, 2, 5).front();
		cv::Mat ldominant = cProc.displayColor(leftdominant);
		cv::Mat rdominant = cProc.displayColor(rightdominant);

		prev_frame.storeDominantColour(leftdominant);
		curr_frame.storeDominantColour(rightdominant);

		//locate the markers and store them in the frames
		std::pair<cv::Rect,cv::Rect> prev_markers = cProc.getMarkers(marker, prev_cimg);
		std::pair<cv::Rect,cv::Rect> curr_markers = cProc.getMarkers(marker, R_cimg);
	
		prev_frame.storeLeftMarker(prev_markers.first);
		prev_frame.storeRightMarker(prev_markers.second);
		curr_frame.storeLeftMarker(curr_markers.first);
		curr_frame.storeRightMarker(curr_markers.second);
		
		//check for direction if it is the first and second frame
		if( i == 0){
			//finds deviation excerpts in depth image and gets its corresponding color image excerpts
			std::vector<cv::Mat>  excerpts = detector.searchDeviationDx(prev_dimg, prev_cimg);
			std::vector<cv::Rect> excerpt_xs = detector.getExcerptWindow();
			prev_frame.storeColourdev_window(excerpt_xs);
			prev_frame.storeDepthdev_window(excerpt_xs);
			//printf("nof excerpts :%d\n", excerpts.size());
			for(int k = 0; k < excerpts.size(); k++){

				cv::Mat window = excerpts.at(k);
				//split the excerpt down the middle
				cv::Mat L_window = cv::Mat(window, cv::Rect(0, 0, window.cols/2, window.rows));
				cv::Mat R_window = cv::Mat(window, cv::Rect((window.cols/2)-1, 0, window.cols/2, window.rows));
				//cv::imshow("leftwindow", L_window);
				//cv::imshow("rightwindow", R_window);
				//cv::waitKey(40);
				//printf("extracting color\n");

				//split the excerpt into two majority colors
				std::vector<cv::Vec3b> roi_classesL = cProc.GetClassesColor(L_window, 5, 2);
				std::vector<cv::Vec3b> roi_classesR = cProc.GetClassesColor(R_window, 5, 2);
				prev_frame.storeColour(roi_classesL.at(0));
				prev_frame.storeColour(roi_classesR.at(0));

				//display the two dominant colors for both sides
				cv::Mat leftcolor = cProc.displayColor(roi_classesL.at(0));
				cv::Mat rightcolor = cProc.displayColor(roi_classesR.at(0));
			}

			//do the same thing for the right frame
			std::vector<cv::Mat>  Rexcerpts = detector.searchDeviationDx(R_dimg, R_cimg);
			std::vector<cv::Rect> Rexcerpt_xs = detector.getExcerptWindow();
			curr_frame.storeColourdev_window(Rexcerpt_xs);
			curr_frame.storeDepthdev_window(Rexcerpt_xs);

			for(int k = 0; k < Rexcerpts.size(); k++){

				cv::Mat window = Rexcerpts.at(k);	
				//split the excerpt down the middle
				cv::Mat L_window = cv::Mat(window, cv::Rect(0, 0, window.cols/2, window.rows));
				cv::Mat R_window = cv::Mat(window, cv::Rect((window.cols/2)-1, 0, window.cols/2, window.rows));
				//cv::imshow("leftwindow", L_window);
				//cv::imshow("rightwindow", R_window);
				//cv::waitKey(40);
				//cv::waitKey(0);
				//cv::destroyAllWindows();
				//printf("extracting color\n");
				cv::GaussianBlur(L_window, L_window, cv::Size(7,7),5);
				cv::GaussianBlur(L_window, L_window, cv::Size(7,7),5);
				//split the excerpt into two majority colors
				std::vector<cv::Vec3b> roi_classesL = cProc.GetClassesColor(L_window, 5, 2);
				std::vector<cv::Vec3b> roi_classesR = cProc.GetClassesColor(R_window, 5, 2);
				curr_frame.storeColour(roi_classesL.at(0));
				curr_frame.storeColour(roi_classesR.at(0));

				//display the two dominant colors for both sides
				cv::Mat leftcolor = cProc.displayColor(roi_classesL.at(0));
				cv::Mat rightcolor = cProc.displayColor(roi_classesR.at(0));
			}

			//with the given information, find for the suitable area to look for correlation stitching
			window = findWindowOfInterest(prev_frame, curr_frame);

			//determine the direction
			bool prev_left = false;bool curr_left;
			if(window.first.x < prev_frame.getColourImage().cols/2){
				//left marker
				prev_left = true;
			}else{
				prev_left = false;
			}
			
			if(window.second.x < curr_frame.getColourImage().cols/2){
				//left marker
				curr_left = true;
			}else{
				curr_left = false;
			}

			if(prev_left && !curr_left){
				direction = RIGHT2LEFT;
				printf("stitching right to left\n");
			}else if(!prev_left && curr_left){
				direction = LEFT2RIGHT;
				printf("stitching left to right\n");
			}
		}else{//end of choosing direction

			if(direction == RIGHT2LEFT){
				//stitch left window of prev frame with right window of current frame
				window = std::pair<cv::Rect,cv::Rect>(curr_frame.getRightMarker(),prev_frame.getLeftMarker());
			}else if(direction == LEFT2RIGHT){
				//stitch right window of prev frame with left window of current frame
				window = std::pair<cv::Rect,cv::Rect>(prev_frame.getRightMarker(), curr_frame.getLeftMarker());
			}
		}


		//find offset of VGA resolution with given window offset
		std::pair<int, int> corr_values = getColorOffset2(prev_cimg, R_cimg, window.first, window.second);
		corr_offsets.push_back(corr_values);
		if(i > 0 && L_cimg.cols > VGA_WIDTH){
			for(int k = 0; k < corr_offsets.size()-1; k++){
				
				corr_values.first = corr_values.first + corr_offsets.at(k).first;
				corr_values.second = corr_values.second + corr_offsets.at(k).second;
			}
		}
		//stitch result together with found offset in VGA resolution

		cv::Mat left_cimg, right_cimg, left_dimg, right_dimg;
		L_cimg.convertTo(left_cimg, CV_8UC1);
		L_dimg.convertTo(left_dimg, CV_8UC1);
		R_cimg.convertTo(right_cimg, CV_8UC1);
		R_dimg.convertTo(right_dimg, CV_8UC1);

 		cv::imshow("stitching thesee", left_cimg); 
		cv::imshow("stitching these", right_cimg);
		cv::waitKey(30);
		cv::waitKey(0);
		cv::Mat corresult = stitch(left_cimg, right_cimg, corr_values.first, corr_values.second);
		cv::Mat depthstitch = stitch(left_dimg, right_dimg, corr_values.first, corr_values.second);
		std::ostringstream stream,dstream;
		stream << "rawdata\\result\\";
		stream << "colorstitch.jpg";
		dstream << "rawdata\\result\\";
		dstream << "depthstitch.jpg";
		cv::String dfilename = dstream.str();
		cv::String filename = stream.str();
		if(cv::imwrite(filename, corresult)){
			printf("Image saved. ");
		}else{
			printf("Error saving Image. ");
		}

		if(cv::imwrite(dfilename, depthstitch)){
			printf("Depth Image saved. ");
		}else{
			printf("Error saving Image. ");
		}

		cv::destroyAllWindows();
		cstitches.push_back(corresult);
		dstitches.push_back(depthstitch);
	}

	result = std::pair<cv::Mat,cv::Mat>(cstitches.back(), dstitches.back());

	return result;
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