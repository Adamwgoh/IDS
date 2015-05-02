#include "ImgRegistration.h"
#include "ColorProcessor.h"
//#include "DeviceRetriever.h"
#include "depthPlaneDetector.h"
#include "Frame.h"
#include "DoorDetector.h"
#include "stdafx.h"

const int LEFT2RIGHT = 19;
const int RIGHT2LEFT = 91;

const int VGA_WIDTH = 640;
const int VGA_HEIGHT = 480;
const int QQVGA_WIDTH = 160;
const int QQVGA_HEIGHT = 120;

std::vector<cv::Mat> newsetoneColour(){
	std::vector<cv::Mat> frames = std::vector<cv::Mat>();
	frames.push_back(cv::imread("rawdata\\newsetone\\2cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\newsetone\\3cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\newsetone\\4cframe.jpg"));

	return frames;
}

std::vector<cv::Mat> newsetoneDepth(){
	std::vector<cv::Mat> frames = std::vector<cv::Mat>();
	frames.push_back(cv::imread("rawdata\\newsetone\\2depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\newsetone\\3depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(cv::imread("rawdata\\newsetone\\4depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	return frames;
}

std::vector<cv::Mat> getCFrames(){
	std::vector<cv::Mat> frames = std::vector<cv::Mat>();
	//this is a messy temporary. Will fix as soon as i'm sure i can stitch all together
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\3cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\4cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\5cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\6cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\7cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\8cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\9cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\10cframe.jpg"));
	frames.push_back(cv::imread("rawdata\\setthree_with_markers\\11cframe.jpg"));
	
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
	ImageRegistration imgreg = ImageRegistration();
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
			printf("nof excerpts :%d\n", excerpts.size());
			for(int k = 0; k < excerpts.size(); k++){

				cv::Mat window = excerpts.at(k);
				//split the excerpt down the middle
				cv::Mat L_window = cv::Mat(window, cv::Rect(0, 0, window.cols/2, window.rows));
				cv::Mat R_window = cv::Mat(window, cv::Rect((window.cols/2)-1, 0, window.cols/2, window.rows));
				cv::imshow("leftwindow", L_window);
				cv::imshow("rightwindow", R_window);
				cv::waitKey(40);
				printf("extracting color\n");

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
				//cv::String windowname = k + "_excerpt";
				//cv::imshow(windowname, window);

				//cv::waitKey(40);
				//cv::waitKey(0);
			
				//split the excerpt down the middle
				cv::Mat L_window = cv::Mat(window, cv::Rect(0, 0, window.cols/2, window.rows));
				cv::Mat R_window = cv::Mat(window, cv::Rect((window.cols/2)-1, 0, window.cols/2, window.rows));
				//cv::imshow("leftwindow", L_window);
				//cv::imshow("rightwindow", R_window);
				//cv::waitKey(40);
				//cv::waitKey(0);
				//cv::destroyAllWindows();
				printf("extracting color\n");
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
			window = imgreg.findWindowOfInterest(prev_frame, curr_frame);

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
		std::pair<int, int> corr_values = imgreg.getColorOffset2(prev_cimg, R_cimg, window.first, window.second);
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
		cv::Mat corresult = imgreg.stitch(left_cimg, right_cimg, corr_values.first, corr_values.second);
		cv::Mat depthstitch = imgreg.stitch(left_dimg, right_dimg, corr_values.first, corr_values.second);
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
		cv::imshow("corresult", corresult);
		cv::imshow("depthstitch", depthstitch);
		cv::waitKey(40);
		cstitches.push_back(corresult);
		dstitches.push_back(depthstitch);
	}

	return cstitches.back();
}

cv::Mat showHueSatHist(cv::Mat src){
	cv::Mat hsv;
	cv::cvtColor(src, hsv, CV_BGR2HSV);

	//quantize the hue to 30 levels and sat to 32 levels
	 int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    cv::MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    calcHist( &hsv, 1, channels, cv::Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
    double maxVal=0;
	cv::minMaxLoc(hist, 0, &maxVal, 0, 0);

    int scale = 10;
    cv::Mat histImg = cv::Mat::zeros(sbins*scale, hbins*10, CV_8UC3);

    for( int h = 0; h < hbins; h++ ){
        for( int s = 0; s < sbins; s++ )
        {
            float binVal = hist.at<float>(h, s);
            int intensity = cvRound(binVal*255/maxVal);
            cv::rectangle( histImg, cv::Point(h*scale, s*scale),
                        cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                        cv::Scalar::all(intensity),
                        CV_FILLED );
        }
	}
	cv::imshow("src", src);
	cv::imshow("histogram", histImg);
	cv::waitKey(40);
	cv::waitKey(0);

	return histImg;
}


int main(){
	//DeviceRetriever* retrieve = new DeviceRetriever();
	//retrieve->runNodes();
	cv::waitKey(0);
	clock_t timer;
	ColorProcessor cProc = ColorProcessor();
	depthPlaneDetector depth_detector = depthPlaneDetector(61,30);

	timer = clock();
	std::vector<cv::Mat> cframes = std::vector<cv::Mat>();
	std::vector<cv::Mat> dframes = std::vector<cv::Mat>();
	//cframes = newsetoneColour();
	//dframes = newsetoneDepth();
	//cv::Mat final  = StitchFrames(cframes, dframes);
	cv::Mat colorpano = cv::imread("rawdata\\colorstitch.jpg");
	cv::Mat depthpano = cv::imread("rawdata\\depthstitch.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	DoorDetector door_detector = DoorDetector();
	cv::Mat depthgraph = depth_detector.displayDepthGraph(depthpano,0,0);

	std::ostringstream stream;
		stream << "rawdata\\result\\";
		stream << "depthgraph.jpg";
		cv::String filename = stream.str();
		if(cv::imwrite(filename, depthgraph)){
			printf("Image saved. ");
		}else{
			printf("Error saving Image. ");
		}
	cv::destroyAllWindows();	
	std::vector<cv::Rect> keypoints = depth_detector.searchDepthDeviation(depthpano, depthgraph);
	std::vector<Line> line_models = door_detector.findDepthLines(depthpano, depthgraph);
	std::vector<cv::Rect> segments = door_detector.getSegments(depthgraph, keypoints);
	//match the vertical line segments with line models to get a candidate door
	std::vector<std::pair<cv::Rect, Line>> excerpts = door_detector.getExcerpts(depthgraph, segments, line_models);
	cv::Mat panoline;depthgraph.copyTo(panoline);
	for(int i = 0; i < excerpts.size(); i++){
		panoline = excerpts.at(i).second.drawLine(panoline,
			excerpts.at(i).second, excerpts.at(i).first);
	}

	cv::imshow("panoline", panoline);
	cv::waitKey(30);
	cv::waitKey(0);
	std::ostringstream steam;
	steam<< "rawdata\\result\\";
	steam << "panoline.jpg";
	cv::String fileaname = steam.str();
	if(cv::imwrite(fileaname, panoline)){
		printf("Image saved. ");
	}else{
		printf("Error saving Image. ");
	}

	timer = clock() - timer;
	printf("Total program runtime %f\n",((double) timer/CLOCKS_PER_SEC) );

	cv::waitKey(40);
	cv::waitKey(0);

	return 0;
}