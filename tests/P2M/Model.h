#ifndef P2M_MODEL_H
#define P2M_MODEL_H

#include "KDTree.h"
#include "RTree.h"
#include "VoronoiTetgen.h"

struct P2M_Result
{
	double dis;
	// mark the type of primitive where the closest point is located
	// vertex: 0, edge: 1, face: 2
	int primitive_type;
	Eigen::Vector3d cp;  // closest point
	Eigen::Vector3d nn;  // nearest vertex found by KDT search
	int kdt_nodes_visited;
	int intercepted_edges, intercepted_faces;  // number of primitives intercepted by nn
};

class Model
{
public:
	struct Edge
	{
		int s, t;  // endpoints
		Eigen::Vector3d dir;  // unit direction vector
		std::vector<Eigen::Vector4d> vp;  // vertical planes
		Edge(int S, int T) :s(S), t(T) {}
	};
	struct Face
	{
		Eigen::Vector3i verts;
		Eigen::Vector3d n;  // unit normal vector
		std::vector<Eigen::Vector4d> vp;  // vertical planes
		Face(Eigen::Vector3i Verts, Eigen::Vector3d N) :verts(Verts), n(N) {}
	};

	// construct with model info 
	// limit_cube_len is a limit value used in Voronoi diagram construction
	// make sure the coordinate values of mesh points and query points are within the limit 
	Model(const std::vector<Eigen::Vector3d>& Points, const std::vector<std::pair<int, int> >& Segments,
		  const std::vector<Eigen::Vector3i>& Faces, const double& Limit_cube_len = 1e3);
	~Model();
	int num_verts() const { return (int)points.size(); }
	int num_edges() const { return (int)edges.size(); }
	int num_faces() const { return (int)faces.size(); }

	// point-mesh distance query
	void p2m(const Eigen::Vector3d& p, P2M_Result& res) const;

public:
	// record the intercepted primitives of a vertex as well as 
	// their corresponding interception regions and acceleration structures
	struct Region
	{
		std::vector<int> edges, faces;
		std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> box_e, box_f;
		std::vector<std::pair<int, double> > x_max;  
		RTree* T;
		~Region()
		{
			if (edges.size()+ faces.size() > 200) delete T;
		}
	};

	double limit_cube_len;
	std::vector<Edge> edges;
	std::vector<Face> faces;
	std::vector<Eigen::Vector3d> points;
	KDTree* kdt;
	std::vector<Region> regions;
	VoronoiTetgen* vor;

	// construction
	void get_mesh(const std::vector<Eigen::Vector3d>& Points, const std::vector<std::pair<int, int> >& Segments, 
		          const std::vector<Eigen::Vector3i>& Faces);
	void init();
	void cal_info();
	void cal_table();
};

#endif //P2M_MODEL_H


