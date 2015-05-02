#include "DoorDetector.h"

#include <baseapi.h>
#include <allheaders.h>

DoorDetector::DoorDetector(){
	cProc = ColorProcessor();
	dDetector = depthPlaneDetector(61,30);

}
//given a stitched colour and depth image, figure out if there's a door
//if there is one, highlight it in the colour and show it
DoorCandidate DoorDetector::hasDoor(cv::Mat colour_src, cv::Mat depth_src){
	//A door is determined by having two edges of the door, changes in hue,
	//and changes in depth indention.
	DoorCandidate possible_door;


	return possible_door;
}

std::vector<Line> DoorDetector::findDepthLines(cv::Mat depth_src, cv::Mat depth_graph){
	std::vector<Line> models = std::vector<Line>();
	std::vector<Line> temp_lines = std::vector<Line>();
	cv::Mat final_graph;depth_graph.copyTo(final_graph);
	//Line average_line = Line(depth_graph);
	std::vector<cv::Rect> keypoints = dDetector.searchDepthDeviation(depth_src, depth_graph);
	std::vector<cv::Rect> segments = getSegments(depth_graph, keypoints);

	//show the keypoints
	//for(int k = 0; k < keypoints.size(); k++){
	//	cv::Mat roi = cv::Mat(depth_graph, keypoints.at(k));
	//	cv::String roiname = k + "roiiii";
	//	cv::imshow(roiname, roi);
	//	cv::waitKey(40);
	//}

	//show the segments
	for(int i = 0; i < segments.size(); i++){
		cv::Mat roi = cv::Mat(depth_graph, segments.at(i));
		Line roi_line = Line(roi);
		temp_lines.push_back(roi_line);
		roi = roi_line.drawLine(roi, roi_line);
		cv::String roiname = i + "segment";
		printf("line model : gradient : %f, y-intercept : %f\n", roi_line.gradient, roi_line.yintercept);
		cv::imshow(roiname, roi);
		cv::waitKey(40);
		//cv::waitKey(0);
	}
	std::vector<Line> temp_models = std::vector<Line>();
	models.push_back(temp_lines.at(0));
	Line initial = models.at(0);

	//get the odd ones out
	for(int k = 0; k < temp_lines.size(); k++){
			//check if it has significant difference between.
			//if yes, add into the model.
			double potential_y = temp_lines.at(k).yintercept;
			double model_y = initial.yintercept;
			double potential_grad = temp_lines.at(k).gradient;
			double model_grad = initial.gradient;

			if(std::abs(potential_y - model_y)/(potential_y + model_y) > 0.1){
				models.push_back(temp_lines.at(k));
			}
	}

	//retrieve the best gradient out
	for(int idx = 0; idx < models.size(); idx++){
		for(int k = 0; k < temp_lines.size(); k++){
			double potential_y = temp_lines.at(k).yintercept;
			double model_y = models.at(idx).yintercept;
			double potential_grad = temp_lines.at(k).gradient;
			double model_grad = models.at(idx).gradient;

			//related to the model
			if(std::abs(potential_y - model_y)/(potential_y + model_y) < 0.1){
				if(std::abs(0 - potential_grad) < std::abs(0 - model_grad)){
					models.at(idx).gradient = potential_grad;
					printf("new gradient is : %f\n", potential_grad);
				}
			}
		}
	}


	printf("nof models found : %d\n", models.size());
	//draw these lines
	for(int i = 0; i < models.size(); i++){
		final_graph = models.at(i).drawLine(final_graph, models.at(i));
	}

	//printf("drawing line..\n");
	//cv::Mat linemat = average_line.drawLine(depth_graph, average_line);
	printf("showing line..\n");
//	cv::imshow("final_graph", final_graph);
//	cv::waitKey(40);
//	cv::waitKey(0);
//	cv::destroyAllWindows();
	return models;
}

std::vector<cv::Rect> DoorDetector::getSegments(cv::Mat depth_graph, std::vector<cv::Rect> rois){
	std::vector<cv::Rect> result = std::vector<cv::Rect>();
	//first segment
	if(rois.size() != 0){
		cv::Rect first_roi = rois.at(0);
		result.push_back(cv::Rect(0,0,first_roi.x, depth_graph.rows));
		//result.push_back(first_segment);
	}

	//middle segments
	for(int k = 0; k < rois.size(); k++){
		if(k == rois.size()-1){
			//last segment
			result.push_back(cv::Rect(rois.at(rois.size()-1).x+rois.at(rois.size()-1).width, 0, depth_graph.cols - (rois.at(rois.size()-1).x+rois.at(rois.size()-1).width), depth_graph.rows));
			//result.push_back(last_segment);
		}else{

			cv::Rect roi = rois.at(k);
			cv::Rect next_roi = rois.at(k+1);
			result.push_back(cv::Rect(roi.x+roi.width, 0, next_roi.x-(roi.x+roi.width), depth_graph.rows));
			//result.push_back(segment);
		}
	}

	return result;
}

//retrieve excerpts based on the given line model
std::vector<std::pair<cv::Rect, Line>> DoorDetector::getExcerpts(cv::Mat depthgraph, std::vector<cv::Rect> segments, std::vector<Line> models){
	std::vector<std::pair<cv::Rect, Line>> excerpts = std::vector<std::pair<cv::Rect, Line>>();
	cv::Rect prev_segment, curr_segment, temp_segment;
	Line prev_model, curr_model, temp_model;
	//initial setup
	prev_segment = segments.at(0);
	cv::Mat prev_roi = cv::Mat(depthgraph, prev_segment);
	prev_model = Line(prev_roi);
	temp_segment = prev_segment;

	for(int i = 1; i < segments.size(); i++){
		//fit a line and see which model does the segment belongs to
		curr_segment = segments.at(i);
		cv::Mat curr_roi = cv::Mat(depthgraph, curr_segment);
		curr_model  = Line(curr_roi);
		if(std::abs(curr_model.yintercept-prev_model.yintercept)/(curr_model.yintercept+prev_model.yintercept) < 0.1){
			//no significant difference. join the two rois
			temp_segment = cv::Rect(temp_segment.x, temp_segment.y, 
				(curr_segment.x-temp_segment.x) + curr_segment.width, (curr_segment.y-temp_segment.y) + curr_segment.height);
			
		}else{
			//significant difference. break the current roi 
			printf("pushing segment\n");
			cv::Mat seg = cv::Mat(depthgraph, temp_segment);
			excerpts.push_back(std::pair<cv::Rect,Line>(temp_segment, prev_model));
			temp_segment = curr_segment;
		}
		prev_segment = curr_segment;
		prev_model = curr_model;
	}

	//push the rest as an segment and its own line model
	cv::Mat last_segment = cv::Mat(depthgraph, temp_segment);
	Line last_model = Line(last_segment);
	//cv::Mat draw = last_model.drawLine(last_segment, last_model);
	//cv::imshow("last_segment", draw);
	//cv::waitKey(40);
	//cv::waitKey(0);
	
	excerpts.push_back(std::pair<cv::Rect,Line>(temp_segment, last_model) );

	return excerpts;
}

cv::String DoorDetector::DetectTexts(cv::Mat colorpano){
	cv::Mat sharpened_pano;
	cv::String answer;
	cv::GaussianBlur(colorpano, sharpened_pano, cv::Size(0, 0), 3);
	cv::addWeighted(colorpano, 1.5, sharpened_pano, -0.5, 0, sharpened_pano);
	cv::imshow("colorpanp", sharpened_pano);
	cv::waitKey(40);
	cv::waitKey(0);

	tesseract::TessBaseAPI tess;
	tess.Init(NULL, "eng");
	tess.SetImage((uchar*)sharpened_pano.data, colorpano.cols, colorpano.rows,
		colorpano.channels(), colorpano.step1());
	tess.Recognize(0);
	const char* text = tess.GetUTF8Text();
	printf("text found : %s\n", text);
	answer = text;

	return answer;
}