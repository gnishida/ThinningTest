#include "ThinningUtil.h"
#include "GraphUtil.h"

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void ThinningUtil::thinningIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows - 1; i++) {
        for (int j = 1; j < im.cols - 1; j++) {
            uchar p1 = im.at<uchar>(i-1, j-1);
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p1 == 1) + (p1 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p1;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
* Perform one thinning iteration.
* Normally you wouldn't call this function directly from your code.
*
* Parameters:
*                 im Binary image with range = [0,1]
*                 iter 0=even, 1=odd
*/
void ThinningUtil::thinningGuoHallIteration(cv::Mat& im, int iter) {
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1); 

    for (int i = 1; i < im.rows - 1; i++) {
        for (int j = 1; j < im.cols - 1; j++) {
            uchar p1 = im.at<uchar>(i-1, j-1);
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                     (!p6 & (p7 | p8)) + (!p8 & (p1 | p2));
            int N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p1) & p8) : ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) & m == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
* Perform one thinning iteration.
* Parameters:
*                 im Binary image with range = [0,1]
*                 iter 0=even, 1=odd
*/
void ThinningUtil::thinningGenIteration(cv::Mat& im, int iter) {
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1); 

    for (int i = 1; i < im.rows - 1; i++) {
        for (int j = 1; j < im.cols - 1; j++) {
            uchar p1 = im.at<uchar>(i-1, j-1);
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);

			int C  = (!p1 & (p2 | p3)) + (p1 & !p2 & p3) + (!p3 & (p4 | p5)) + (p3 & !p4 & p5) + (!p5 & (p6 | p7)) + (p5 & !p6 & p7) + (!p7 & (p8 | p1)) + (p7 & !p8 & p1);
            int N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1);
            int N  = N1 < N2 ? N1 : N2;
			int m;
			if (iter == 0) {
				m = p2;
			} else if (iter == 1) {
				m = p4;
			} else if (iter == 2) {
				m = p6;
			} else if (iter == 3) {
				m = p8;
			}

            if (C == 1 && N >= 2 && N <= 3 && m == 0) {
                marker.at<uchar>(i,j) = 1;
			}
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void ThinningUtil::thinning(const cv::Mat& src, cv::Mat& dst) {
	dst = src.clone();
	dst /= 255;

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningGuoHallIteration(dst, 0);
        thinningGuoHallIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

/**
* Function for thinning the given binary image
*
* Parameters:
*                 src The source image, binary with range = [0,255]
*                 dst The destination image
*/
void ThinningUtil::thinningGuoHall(const cv::Mat& src, cv::Mat& dst) {
	dst = src.clone();
	dst /= 255;

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningGuoHallIteration(dst, 0);
        thinningGuoHallIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

/**
* Function for thinning the given binary image
*
* Parameters:
*                 src The source image, binary with range = [0,255]
*                 dst The destination image
*/
void ThinningUtil::thinningGen(const cv::Mat& src, cv::Mat& dst) {
	dst = src.clone();
	dst /= 255;

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningGenIteration(dst, 0);
        thinningGenIteration(dst, 1);
        thinningGenIteration(dst, 2);
        thinningGenIteration(dst, 3);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

/**
 * 細線化された画像を使って、交差点を決定し、それをroadsに登録する。
 * 画像の輝度値は、0または255という前提。
 */
void ThinningUtil::findIntersection(const cv::Mat& src, RoadGraph& roads) {
	roads.clear();

    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == 0) continue;

			int count = 0;
			for (int v = -1; v <= 1; v++) {
				for (int u = -1; u <= 1; u++) {
					if (u == 0 && v == 0) continue;

					count += src.at<uchar>(i + v, j + u);
				}
			}

			// 8近傍のセルのうち、3つ以上が白色、または1つのみ白色の時、当該セルを交差点とする。
			if (count >= 765 || count == 255) {
				// 近くに頂点があるか？
				RoadVertexDesc v;
				if (!GraphUtil::getVertex(roads, QVector2D(j, i), 10.0f, v)) {
					// なければ、頂点を追加する
					v = boost::add_vertex(roads.graph);
					roads.graph[v] = RoadVertexPtr(new RoadVertex(QVector2D(j, i)));
				}
			}
		}
	}
}

/**
 * 細線化された画像の行列を使って、道路網をsimplifyする。
 * 画像の輝度値は、0または255という前提。
 * アルゴリズム：
 *   1. 細線化された画像から、交差点の候補を抽出する。
 *   2. 各エッジについて、両端頂点を近傍の交差点に移動する。近くに交差点がない場合は、近くの白色セルに移動して交差点とする。
 */
void ThinningUtil::simplifyRoad(const cv::Mat& img, RoadGraph& srcRoad, RoadGraph& dstRoad) {
	int row, col;

	// まず、交差点候補を抽出する
	findIntersection(img, dstRoad);

	// dstRoadの画像
	cv::Mat dstRoadMat = cv::Mat::zeros(img.size(), CV_8U);

	QHash<RoadVertexDesc, RoadVertexDesc> conv;	// srcRoadの頂点 ⇒ dstRoadの頂点への変換テーブル

	RoadEdgeIter ei, eend;
	for (boost::tie(ei, eend) = boost::edges(srcRoad.graph); ei != eend; ++ei) {
		if (!srcRoad.graph[*ei]->valid) continue;

		RoadVertexDesc v1 = boost::source(*ei, srcRoad.graph);
		RoadVertexDesc v2 = boost::target(*ei, srcRoad.graph);

		RoadVertexDesc new_v1, new_v2;

		// v1に対応する新しい頂点new_v1を探す
		if (conv.contains(v1)) {
			new_v1 = conv[v1];
		} else {
			if (!GraphUtil::getVertex(dstRoad, srcRoad.graph[v1]->pt, 40.0f, new_v1)) {
				findNearestCell(img, srcRoad.graph[v1]->pt.y(), srcRoad.graph[v1]->pt.x(), 1000, row, col);
				new_v1 = boost::add_vertex(dstRoad.graph);
				dstRoad.graph[new_v1] = RoadVertexPtr(new RoadVertex(QVector2D(col, row)));
			}
			conv[v1] = new_v1;
		}

		// v2に対応する新しい頂点new_v2を探す
		if (conv.contains(v2)) {
			new_v2 = conv[v2];
		} else {
			if (!GraphUtil::getVertex(dstRoad, srcRoad.graph[v2]->pt, 40.0f, new_v2)) {
				findNearestCell(img, srcRoad.graph[v2]->pt.y(), srcRoad.graph[v2]->pt.x(), 1000, row, col);
				new_v2 = boost::add_vertex(dstRoad.graph);
				dstRoad.graph[new_v2] = RoadVertexPtr(new RoadVertex(QVector2D(col, row)));
			}
			conv[v2] = new_v2;
		}

		if (new_v1 == new_v2) continue;

		// エッジを追加
		RoadEdgeDesc e = GraphUtil::addEdge(dstRoad, new_v1, new_v2, srcRoad.graph[*ei]);
		GraphUtil::moveEdge(dstRoad, e, dstRoad.graph[new_v1]->pt, dstRoad.graph[new_v2]->pt);

		// 追加したエッジが、白色の線の上をだいたい通っていることを確認する
		int onCount = 0;
		int totalCount = 0;
		int r, c;
		std::vector<QVector2D> line = GraphUtil::finerEdge(dstRoad, e, 10.0f);
		for (int i = 0; i < line.size(); i++, totalCount++) {
			if (findNearestCell(dstRoadMat, line[i].y(), line[i].x(), 5, r, c)) onCount++;
		}
		if (onCount > (float)totalCount * 0.5f) dstRoad.graph[e]->valid = false;

		// 追加したエッジを、画像に反映する
		GraphUtil::drawRoadSegmentOnMat(dstRoad, e, dstRoadMat);
	}

	GraphUtil::reduce(dstRoad);
	GraphUtil::clean(dstRoad);
}

bool ThinningUtil::findNearestCell(const cv::Mat& mat, int r, int c, int max_dist, int& nearestRow, int& nearestCol) {
	for (int dist = 0; dist <= max_dist; dist++) {
		// 上辺を左から右へスキャン
		int row = r - dist;
		if (row >= 0 && row < mat.rows) {
			for (int col = c - dist; col <= c + dist; col++) {
				if (col < 0) continue;
				if (col >= mat.cols) break;

				if (mat.at<uchar>(row, col) > 0) {
					nearestRow = row;
					nearestCol = col;
					return true;
				}
			}
		}

		// 下辺を左から右へスキャン
		row = r + dist;
		if (row >= 0 && row < mat.rows) {
			for (int col = c - dist; col <= c + dist; col++) {
				if (col < 0) continue;
				if (col >= mat.cols) break;

				if (mat.at<uchar>(row, col) > 0) {
					nearestRow = row;
					nearestCol = col;
					return true;
				}
			}
		}

		// 左辺を上から下へスキャン
		int col = c - dist;
		if (col >= 0 && col < mat.cols) {
			for (int row = r - dist + 1; row < r + dist; row++) {
				if (row < 0) continue;
				if (row >= mat.rows) break;

				if (mat.at<uchar>(row, col) > 0) {
					nearestRow = row;
					nearestCol = col;
					return true;
				}
			}
		}

		// 右辺を上から下へスキャン
		col = c + dist;
		if (col >= 0 && col < mat.cols) {
			for (int row = r - dist + 1; row < r + dist; row++) {
				if (row < 0) continue;
				if (row >= mat.rows) break;

				if (mat.at<uchar>(row, col) > 0) {
					nearestRow = row;
					nearestCol = col;
					return true;
				}
			}
		}
	}

	return false;
}
