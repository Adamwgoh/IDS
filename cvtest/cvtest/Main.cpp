#include "ImgRegistration.h"
#include "ColorProcessor.h"
//#include "DeviceRetriever.h"
#include "depthPlaneDetector.h"
#include "stdafx.h"


std::vector<cv::Mat> getCFrames(){
	std::vector<cv::Mat> frames = std::vector<cv::Mat>();
	//this is a messy temporary. Will fix as soon as i'm sure i can stitch all together
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\3cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\4cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\5cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\6cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\7cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\8cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\9cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\10cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\11cframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	
	return frames;
}

std::vector<cv::Mat> getDFrames(){
	std::vector<cv::Mat> frames = std::vector<cv::Mat>();
	//this is a messy temporary. Will fix as soon as i'm sure i can stitch all together
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\3depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\4depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\5depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\6depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\7depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\8depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\93depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\10depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\11depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	
	return frames;
}

//stitches all frames based on able to find keypoints. Stops when keypoint not found. FOR NOW
cv::Mat StitchFrames(std::vector<cv::Mat> cframes, std::vector<cv::Mat> dframes){
	assert(cframes.size() == dframes.size());
	cv::Mat L_cimg, R_cimg;
	cv::Mat L_dimg, R_dimg;
	cv::Mat prev_cimg, prev_dimg;
	std::vector<cv::Mat> cstitches = std::vector<cv::Mat>();
	std::vector<cv::Mat> dstitches = std::vector<cv::Mat>();
	depthPlaneDetector detector(61,30);
	ImageRegistration imgreg = ImageRegistration();
	
	for(int i = 0; i < cframes.size()-1; i = i++){
		L_cimg = cframes.at(i);
		R_cimg = cframes.at(i+1);
		L_dimg = dframes.at(i);
		prev_dimg = dframes.at(i);
		prev_cimg = cframes.at(i);
		R_dimg = dframes.at(i+1);
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

		cv::resize(R_dimg, R_dimg, cv::Size(R_cimg.cols, R_cimg.rows), CV_INTER_NN);\
		detector.searchDeviationDx(prev_dimg);
		//found window offset is in VGA resolution
		int x = detector.x; int y = detector.y;
		printf("x : %d, y : %d\n", x, y);
		assert(x != 0 && y != 0);

		//find offset of VGA resolution with given window offset
		std::pair<int, int> corr_values = imgreg.getColorOffset(prev_cimg, R_cimg, x, y, detector.getWindowsize());
		//stitch result together with found offset in VGA resolution


		cv::Mat corresult = imgreg.stitch(L_cimg, R_cimg, corr_values.first, corr_values.second);
		cv::Mat depthstitch = imgreg.stitch(L_dimg, R_dimg, corr_values.first, corr_values.second);
		std::ostringstream stream;
		stream << "rawdata\\result\\";
		stream << "ttttcolornccstitch.jpg";
		cv::String filename = stream.str();
		if(cv::imwrite(filename, corresult)){
			printf("Image saved. ");
		}else{
			printf("Error saving Image. ");
		}
		cv::imshow("corresult", corresult);
		cv::imshow("depthstitch", depthstitch);
		cv::waitKey(40);
		cv::waitKey(0);

		cstitches.push_back(corresult);
		dstitches.push_back(depthstitch);
	}

	return cstitches.back();
}




int main(){
	//DeviceRetriever* retrieve = new DeviceRetriever();

	//retrieve->runNodes();
	std::vector<cv::Mat> cframes = std::vector<cv::Mat>();
	std::vector<cv::Mat> dframes = std::vector<cv::Mat>();
	//cframes = getCFrames();
	//dframes = getDFrames();
	//StitchFrames(cframes, dframes);
	clock_t timer;
	ColorProcessor cProc = ColorProcessor();

	timer = clock();
	cv::Mat cframe1 = cv::imread("rawdata\\setthree_with_markers\\5cframe.jpg");
	cv::Mat cframe2 = cv::imread("rawdata\\setthree_with_markers\\11cframe.jpg");
	cv::Mat cframe3 = cv::imread("rawdata\\set_five\\6cframe.jpg");
	cv::Mat original, result;
	cframe1.copyTo(original);

	cv::GaussianBlur(original, result, cv::Size(3,3),0,0);
	cv::medianBlur(result, result, 9);

	cv::Mat hsv;
	cv::cvtColor(result, hsv, CV_RGB2HSV_FULL);
	cv::Mat hue = cProc.getHueImage(hsv);
	cv::Mat sat = cProc.getSaturationImage(hsv);
	cv::Mat val = cProc.getValueImage(hsv);

	cv::destroyAllWindows();

	cv::Mat hist = cProc.getHueHistogram(hue);
	cv::Mat thresh_sat,thresh_hue,thresh_val;

	cv::threshold(sat, thresh_sat, 0, 255, CV_THRESH_OTSU);
	
	cv::Mat dilated_thresh_sat = cProc.DilateRegionFilling(thresh_sat);

	//convert to u8bit single channel
	dilated_thresh_sat.convertTo(dilated_thresh_sat,CV_8UC1);
	cv::Mat satgraph = cProc.displayDepthGraph(dilated_thresh_sat);

	std::vector<cv::Mat> windows = cProc.findDx(satgraph, 40);

	for(int i = 0; i < windows.size(); i++){
		std::ostringstream stream;
		stream << i;
		stream << "_deviation window";
		cv::String windowname = stream.str();
		cv::imshow(windowname, windows.at(i));
	}

	//cv::imshow("dilated_thresh_sat", dilated_thresh_sat);
	cv::imshow("satgraph", satgraph);
	//cv::imshow("original", original);
	cv::waitKey(40);
	cv::waitKey(0);
	cv::destroyAllWindows();
	timer = clock() - timer;
	printf("Total program runtime %f\n",((double) timer/CLOCKS_PER_SEC) );
	cv::waitKey(0);

	return 0;
}