/**
 * This thinning utility implements the following website code.
 * https://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/#more-675
 * 元になる論文は、以下。
 * Z. Guo and R. Hall. Parallel thinning with two-subiteration algorithms. Communications of the ACM, 32(3):359-373, 1989.
 *
 */

#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "RoadGraph.h"

class ThinningUtil {
protected:
	ThinningUtil() {}
	~ThinningUtil() {}

private:
	static void thinningIteration(cv::Mat& im, int iter);
	static void thinningGuoHallIteration(cv::Mat& im, int iter);
	static void thinningGenIteration(cv::Mat& im, int iter);

public:
	static void thinning(const cv::Mat& src, cv::Mat& dst);
	static void thinningGuoHall(const cv::Mat& src, cv::Mat& dst);
	static void thinningGen(const cv::Mat& src, cv::Mat& dst);
	static void findIntersection(const cv::Mat& src, RoadGraph& roads);
	static void simplifyRoad(const cv::Mat& img, RoadGraph& srcRoad, RoadGraph& dstRoad);
	static bool findNearestCell(const cv::Mat& mat, int r, int c, int max_dist, int& nearestRow, int& nearestCol);
};

