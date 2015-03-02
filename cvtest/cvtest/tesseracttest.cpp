#include <baseapi.h>
#include <allheaders.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>

int main(){
	cv::Mat image = cv::imread("ocrtest.jpg");
	if(!image.empty()){
		cv::imshow("ocrtest", image);
		
	}else{
		printf("image is empty!\n");
	}

	tesseract::TessBaseAPI tess;
	tess.Init(NULL, "eng");
	tess.SetImage((uchar*)image.data, image.size().width, image.size().height,
		image.channels(), image.step1());
	tess.Recognize(0);
	const char* text = tess.GetUTF8Text();

	printf("%s\n", text);
	cv::waitKey(0);
}