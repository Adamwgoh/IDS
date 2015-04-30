#include <vector>
#include <exception>
#include <DepthSense.hxx>
#include <cv.h>
#include <highgui.h>

using namespace cv;

//event listeners for audio, color and depth
void onNewColorSample(DepthSense::ColorNode node, DepthSense::ColorNode::NewSampleReceivedData data);
void onNewDepthSample(DepthSense::DepthNode node, DepthSense::DepthNode::NewSampleReceivedData data);

//functions to be called when node is connected or disconnected. When connected, call appriopriate configuration set
void onNodeDisconnected(DepthSense::Device device, DepthSense::Device::NodeRemovedData data);
void onNodeConnected(DepthSense::Device device, DepthSense::Device::NodeAddedData data);

void onDeviceConnected(DepthSense::Context context,DepthSense:: Context::DeviceAddedData data);
void onDeviceDisconnected(DepthSense::Context context, DepthSense::Context::DeviceRemovedData data);

void configureDepthNode();
void configureColorNode();
void configureNode(DepthSense::Node node);

int main(int arc, char* argv[]);

//when context finds device, connect nodes