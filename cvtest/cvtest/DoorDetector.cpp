#include "DoorDetector.h"

#include <baseapi.h>
#include <allheaders.h>

DoorDetector::DoorDetector(){
	cProc = ColorProcessor();
	dDetector = depthPlaneDetector(61,30);

}
//given a stitched colour and depth image, figure out if there's a door
//if there is one, highlight it in the colour and show it
//supports only one door candidate
DoorCandidate DoorDetector::hasDoor(cv::Mat colorpano, cv::Mat depthpano){
	//A door is determined by having two edges of the door, changes in hue,
	//and changes in depth indention.
	/**
	 * 	std::vector<Line> depth_lines;
	 *	cv::Mat door;
	 *	cv::Mat door_graph;
	 *	cv::Vec3b door_colour;
	 **/
	DoorCandidate possible_door;
	depthPlaneDetector depth_detector = depthPlaneDetector(61,30);
	ColorProcessor cProc = ColorProcessor();
	cv::Mat depthgraph = depth_detector.displayDepthGraph(depthpano,0,0);

	std::vector<cv::Rect> keypoints = depth_detector.searchDepthDeviation(depthpano, depthgraph);
	std::vector<Line> line_models  = findDepthLines(depthpano, depthgraph);
	std::vector<cv::Rect> segments = getSegments(depthgraph, keypoints);
	//match the vertical line segments with line models to get a candidate door
	std::vector<std::pair<cv::Rect, Line>> excerpts = getExcerpts(depthgraph, segments, line_models);
	std::vector<cv::Mat> color_excerpts = std::vector<cv::Mat>();

	cv::Mat panoline;depthgraph.copyTo(panoline);
	possible_door.storeDoorGraph(panoline);

	for(int i = 0; i < excerpts.size(); i++){
		cv::Mat colorexcerpt = cv::Mat(colorpano, excerpts.at(i).first);
		possible_door.storeLine(excerpts.at(i).second);
		color_excerpts.push_back(colorexcerpt);
		panoline = excerpts.at(i).second.drawLine(panoline,
			excerpts.at(i).second, excerpts.at(i).first);
	}

	//cv::imshow("panoline", panoline);
	//cv::waitKey(30);
	//cv::waitKey(0);
	std::ostringstream steam;
	steam<< "rawdata\\result\\";
	steam << "panoline.jpg";
	cv::String fileaname = steam.str();
	if(cv::imwrite(fileaname, panoline)){
		printf("Image saved. \n");
	}else{
		printf("Error saving Image. \n");
	}

	//get their colours and look for change of colour between twon segments
	cv::Vec3b prev_color, curr_color;
	cv::Vec3b* temp_color = NULL;
	int doorcandidate_excerpt = 0;
	std::vector<int> doorcandidates = std::vector<int>();
	prev_color = cProc.GetClassesColor(color_excerpts.at(0),2,5).at(0);

	for(int k = 1; k < color_excerpts.size(); k++){
		curr_color = cProc.GetClassesColor(color_excerpts.at(k),2,5).at(0);

		double colourdiff = cProc.compareColours(prev_color,curr_color);
		if(colourdiff > 50){
			//there's a change in color, check if the next color 
			//is similar to the changed one 
			if(temp_color == NULL){
				//mark excerpt as potential
				temp_color = &curr_color;
				doorcandidate_excerpt = k;
			}else{
				//there's change in colour between previous and now colour
				//check if there's difference in colour between temp and prev?
				colourdiff = cProc.compareColours(prev_color, *temp_color);
				if(colourdiff > 50){
					possible_door.storeDoorColour(*temp_color);
					possible_door.setHasDoor(true);
					doorcandidates.push_back(doorcandidate_excerpt);
					break;
					
				}else{
				
					temp_color = &prev_color;
					doorcandidate_excerpt = k;
				}
			}
		}
		prev_color = curr_color;
	}

	//extract doorexcerpt and display it
	cv::Mat candidate = cv::Mat(colorpano, excerpts.at(doorcandidate_excerpt).first);
	possible_door.storeDoorExtract(candidate);

	return possible_door;
}

std::vector<Line> DoorDetector::findDepthLines(cv::Mat depth_src, cv::Mat depth_graph){
	std::vector<Line> models = std::vector<Line>();
	std::vector<Line> temp_lines = std::vector<Line>();
	cv::Mat final_graph;depth_graph.copyTo(final_graph);
	//Line average_line = Line(depth_graph);
	std::vector<cv::Rect> keypoints = dDetector.searchDepthDeviation(depth_src, depth_graph);
	std::vector<cv::Rect> segments = getSegments(depth_graph, keypoints);

	//show the segments
	for(int i = 0; i < segments.size(); i++){
		cv::Mat roi = cv::Mat(depth_graph, segments.at(i));
		Line roi_line = Line(roi);
		temp_lines.push_back(roi_line);
		roi = roi_line.drawLine(roi, roi_line);
		//cv::String roiname = i + "segment";
		printf("line model : gradient : %f, y-intercept : %f\n", roi_line.gradient, roi_line.yintercept);
		//cv::imshow(roiname, roi);
		//cv::waitKey(40);
		//cv::waitKey(0);
	}
	cv::waitKey(0);
	cv::waitKey(40);
	cv::destroyAllWindows();
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
	
	for(int i = 0; i < models.size(); i++){
		final_graph = models.at(i).drawLine(final_graph, models.at(i));
	}


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

const char* DoorDetector::DetectTexts(cv::Mat colorpano){
	cv::Mat sharpened_pano;
	const char* answer;
	cv::GaussianBlur(colorpano, sharpened_pano, cv::Size(0, 0), 3);
	cv::addWeighted(colorpano, 1.5, sharpened_pano, -0.5, 0, sharpened_pano);
	cv::imshow("colorpano", sharpened_pano);
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