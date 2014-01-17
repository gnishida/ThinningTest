﻿#include "RoadGraph.h"
#include "Util.h"
#include "GraphUtil.h"
#include <QGLWidget>

#define _USE_MATH_DEFINES
#include <math.h>

#define SQR(x)		((x) * (x))

RoadGraph::RoadGraph() {
	modified = true;
}

RoadGraph::~RoadGraph() {
}

void RoadGraph::generateMesh() {
	if (!modified) return;

	renderables.clear();

	renderables.push_back(RenderablePtr(new Renderable(GL_TRIANGLES)));

	// road edge
	RoadEdgeIter ei, eend;
	for (boost::tie(ei, eend) = boost::edges(graph); ei != eend; ++ei) {
		if (!graph[*ei]->valid) continue;

		RoadEdgePtr edge = graph[*ei];

		QColor color, bColor;
		float height;
		switch (edge->type) {
		case 3:	// high way
			/*
			color = QColor(255, 225, 104);
			bColor = QColor(229, 153, 21);
			height = highwayHeight;
			break;
			*/
		case 2: // avenue
		case 1: // street
			color = QColor(255, 255, 255);
			bColor = QColor(217, 209, 201);
			height = avenueHeight;
			break;
		}

		// draw the border of the road segment
		if (!showLocalStreets && edge->type == 1) {
			// If this is the local street and it should be drawn in gray color, it should be a little narrow line.
			addMeshFromEdge(renderables[0], edge, widthBase * 0.6f, bColor, 0.0f);
		} else {
			addMeshFromEdge(renderables[0], edge, widthBase * (1.0f + curbRatio), bColor, 0.0f);
			addMeshFromEdge(renderables[0], edge, widthBase, color, height);
		}
	}

	// road vertices
	Vertex v;
	v.normal[0] = 0.0f;
	v.normal[1] = 0.0f;
	v.normal[2] = 0.0f;
	v.color[0] = 0.0f;
	v.color[1] = 0.0f;
	v.color[2] = 1.0f;
	renderables.push_back(RenderablePtr(new Renderable(GL_POINTS, 10.0f)));
	RoadVertexIter vi, vend;
	for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi) {
		if (!graph[*vi]->valid) continue;

		v.location[0] = graph[*vi]->pt.x();
		v.location[1] = graph[*vi]->pt.y();
		v.location[2] = avenueHeight;
		renderables[1]->vertices.push_back(v);
	}

	modified = false;
}

/**
 * Add a mesh for the specified edge.
 */
void RoadGraph::addMeshFromEdge(RenderablePtr renderable, RoadEdgePtr edge, float widthBase, QColor color, float height) {
	Vertex v;

	// define the width of the road segment
	float width;
	switch (edge->type) {
	case 3: // high way
		width = widthBase * 2.0f;
		break;
	case 2: // avenue
		width = widthBase * 1.5f;
		break;
	case 1: // local street
		width = widthBase * 1.0f;
		break;
	}

	int num = edge->polyLine.size();

	// draw the edge
	for (int i = 0; i < num - 1; ++i) {
		QVector2D pt1 = edge->polyLine[i];
		QVector2D pt2 = edge->polyLine[i + 1];
		QVector2D vec = pt2 - pt1;
		vec = QVector2D(-vec.y(), vec.x());
		vec.normalize();

		QVector2D p0 = pt1 + vec * width * 0.5f;
		QVector2D p1 = pt1 - vec * width * 0.5f;
		QVector2D p2 = pt2 - vec * width * 0.5f;
		QVector2D p3 = pt2 + vec * width * 0.5f;

		v.color[0] = color.redF();
		v.color[1] = color.greenF();
		v.color[2] = color.blueF();
		v.color[3] = color.alphaF();
		v.normal[0] = 0.0f;
		v.normal[1] = 0.0f;
		v.normal[2] = 1.0f;

		v.location[2] = height;

		v.location[0] = p0.x();
		v.location[1] = p0.y();
		renderable->vertices.push_back(v);

		v.location[0] = p1.x();
		v.location[1] = p1.y();
		renderable->vertices.push_back(v);

		v.location[0] = p2.x();
		v.location[1] = p2.y();
		renderable->vertices.push_back(v);

		v.location[0] = p0.x();
		v.location[1] = p0.y();
		renderable->vertices.push_back(v);

		v.location[0] = p2.x();
		v.location[1] = p2.y();
		renderable->vertices.push_back(v);

		v.location[0] = p3.x();
		v.location[1] = p3.y();
		renderable->vertices.push_back(v);
	}
}

bool RoadGraph::getModified() {
	return modified;
}

void RoadGraph::setModified() {
	modified = true;
}

void RoadGraph::clear() {
	graph.clear();

	modified = true;
}

void RoadGraph::setZ(float z) {
	// define the width per lane
	float widthBase2;
	if (z < 300.0f) {
		widthBase2 = 2.0f;
	} else if (z < 600.0f) {
		widthBase2 = 4.0f;
	} else if (z < 1080.0f) {
		widthBase2 = 10.0f;
	} else if (z < 5760.0f) {
		widthBase2 = 12.0f;
	} else {
		widthBase2 = 24.0f;
	}
	if (widthBase != widthBase2) {
		widthBase = widthBase2;
		modified = true;
	}

	// define the curb ratio
	float curbRatio2;
	if (z < 2880.0f) {
		curbRatio2 = 0.4f;
	} else {
		curbRatio2 = 0.8f;
	}
	if (curbRatio != curbRatio2) {
		curbRatio = curbRatio2;
		modified = true;
	}

	// define whether to draw local street
	bool showLocalStreets2;
	if (z < 5760.0f) {
		showLocalStreets2 = true;
	} else {
		showLocalStreets2 = false;
	}
	if (showLocalStreets != showLocalStreets2) {
		showLocalStreets = showLocalStreets2;
		modified = true;
	}

	// define the height
	float highwayHeight2 = (float)((int)(z * 0.012f)) * 0.1f;
	if (highwayHeight != highwayHeight2) {
		highwayHeight = highwayHeight2;
		avenueHeight = highwayHeight2 * 0.66f;
		modified = true;
	}
}

/**
 * Importanceに基づいて、エッジを並べて返却する。
 */
QList<RoadEdgeDesc> RoadGraph::getOrderedEdgesByImportance() {
	std::vector<RoadEdgeDesc> data;

	RoadEdgeIter ei, eend;
	int count = 0;
	for (boost::tie(ei, eend) = boost::edges(graph); ei != eend; ++ei) {
		if (!graph[*ei]->valid) continue;

		data.push_back(*ei);
		//data.push_back(count);
		count++;
	}

	std::sort(data.begin(), data.end(), MoreImportantEdge(this));

	QList<RoadEdgeDesc> ret;
	for (int i = 0; i < data.size(); i++) {
		//ret.push_back(GraphUtil::getEdge(this, data[i]));
		ret.push_back(data[i]);
	}

	return ret;
}

LessWeight::LessWeight(RoadGraph* roads) {
	this->roads = roads;
}

bool LessWeight::operator()(const RoadEdgeDesc& left, const RoadEdgeDesc& right) const {
	return roads->graph[left]->lanes < roads->graph[right]->lanes;
}

MoreImportantEdge::MoreImportantEdge(RoadGraph* roads) {
	this->roads = roads;
}

//bool MoreImportantEdge::operator()(const int& left, const int& right) const {
bool MoreImportantEdge::operator()(const RoadEdgeDesc& left, const RoadEdgeDesc& right) const {
	RoadEdgePtr e1 = roads->graph[left];
	RoadEdgePtr e2 = roads->graph[right];

	return e1->importance > e2->importance;
}

/*void RoadGraph::load(FILE* fp, int roadType) {
	clear();

	QMap<uint, RoadVertexDesc> idToDesc;

	// Read the number of vertices
	unsigned int nVertices;
	fread(&nVertices, sizeof(unsigned int), 1, fp);

	// Read each vertex's information: desc, x, and y.
	for (int i = 0; i < nVertices; i++) {
		RoadVertexDesc id;
		float x, y;
		fread(&id, sizeof(RoadVertexDesc), 1, fp);
		fread(&x, sizeof(float), 1, fp);
		fread(&y, sizeof(float), 1, fp);

		RoadVertexPtr vertex = new RoadVertex(QVector2D(x, y));

		RoadVertexDesc desc = boost::add_vertex(graph);
		graph[desc] = vertex;

		idToDesc[id] = desc;
	}

	// Read the number of edges
	unsigned int nEdges;
	fread(&nEdges, sizeof(unsigned int), 1, fp);

	// Read each edge's information: the descs of two vertices, road type, the number of lanes, the number of points along the polyline, and the coordinate of each point along the polyline.
	for (int i = 0; i < nEdges; i++) {
		RoadEdge* edge = new RoadEdge(1, 1, false);

		RoadVertexDesc id1, id2;
		fread(&id1, sizeof(RoadVertexDesc), 1, fp);
		fread(&id2, sizeof(RoadVertexDesc), 1, fp);

		RoadVertexDesc src = idToDesc[id1];
		RoadVertexDesc tgt = idToDesc[id2];

		fread(&edge->type, sizeof(unsigned int), 1, fp);
		fread(&edge->lanes, sizeof(unsigned int), 1, fp);
		fread(&edge->oneWay, sizeof(unsigned int), 1, fp);

		unsigned int nPoints;
		fread(&nPoints, sizeof(unsigned int), 1, fp);

		for (int j = 0; j < nPoints; j++) {
			float x, y;
			fread(&x, sizeof(float), 1, fp);
			fread(&y, sizeof(float), 1, fp);

			edge->addPoint(QVector2D(x, y));
		}

		// 指定されたタイプの道路エッジのみを読み込む
		if (((int)powf(2, (edge->type - 1)) & roadType)) {
			std::pair<RoadEdgeDesc, bool> edge_pair = boost::add_edge(src, tgt, graph);
			graph[edge_pair.first] = edge;
		} else {
			delete edge;
		}
	}

	modified = true;
}*/