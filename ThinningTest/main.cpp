/**
 * 画像を、DilationとThinningを使って、シンプル化する。
 * 以下の論文で説明されている手法を使っている。
 * Microsoft Tech Report 2012 - Segmentaion of Urban Areas Using Road Networks
 *
 * @author Gen Nishida
 * @version 1.0
 */

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <road/GraphUtil.h>
#include "ThinningUtil.h"

int main(int argc, char *argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: RoadSimplification <image file>" << std::endl;
		return 1;
	}
	
	cv::Mat result;

	result = cv::imread(argv[1], 0);
	result.convertTo(result, CV_8UC1);

	// Thinning
	//ThinningUtil::thinning(result, result);
	ThinningUtil::thinningGuoHall(result, result);
	//ThinningUtil::thinningGen(result, result);
	
	cv::imwrite("result.bmp", result);

	// Display
	cv::namedWindow("result", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	cv::imshow("result", result);

	cv::waitKey(0);
}