#include "DoorCandidate.h"

DoorCandidate::DoorCandidate(){

}

void DoorCandidate::printLines(){
	printf("total lines : %d\n", depth_lines.size());
	for(int i = 0; i < depth_lines.size();i++){
		printf("Line %d, gradient :%f, yintercept :%f\n", i,depth_lines.at(i).gradient,depth_lines.at(i).yintercept);
	}
}

void DoorCandidate::storeLine(Line line_model){
	depth_lines.push_back(line_model);
}

void DoorCandidate::storeDoorExtract(cv::Mat roi){
	door = roi;
}

void DoorCandidate::setHasDoor(bool hasdoor){
	doorfound = hasdoor;
}

bool DoorCandidate::hasDoor(){
	return doorfound;
}

cv::Mat DoorCandidate::getDoorGraph(){
	return door_graph;
}

void DoorCandidate::storeDoorGraph(cv::Mat graph){
	door_graph = graph;
}

void DoorCandidate::storeDoorColour(cv::Vec3b colour){
	door_colour = colour;
}
	
std::vector<Line> DoorCandidate::getLineModels(){
	return depth_lines;
}

cv::Mat DoorCandidate::getDoorExtract(){
	return door;
}

cv::Vec3b DoorCandidate::getDoorColour(){
	return door_colour;
}