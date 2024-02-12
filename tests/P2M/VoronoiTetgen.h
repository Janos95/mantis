#ifndef P2M_VORONOITETGEN_H
#define P2M_VORONOITETGEN_H

#include <Eigen/Dense>
#include "tetgen.h"

class VoronoiTetgen
{
public:
	struct CEdge
	{
		int e, f1, f2;  // f1, f2 are the connected faces in a certain Voronoi cell
		CEdge(){}
		CEdge(int E,int F1,int F2):e(E),f1(F1),f2(F2){}
	};
	struct Cell
	{
		std::vector<int> nei;
		std::vector<CEdge> edges;
	};
	Cell *cells;

	VoronoiTetgen(const std::vector<Eigen::Vector3d> & pts, const double& limit_cube_len);
	~VoronoiTetgen();

	// clip a Voronoi cell by several planes
	bool clip(int cell_id, const std::vector<Eigen::Vector4d>& ps, 
		      std::vector<Eigen::Vector3d>& res_pts, std::vector<std::vector<int>>& to_pts);
	
private:
	int num_cells;
	tetgenmesh m;
	Eigen::Vector3d *_verts; 
	std::pair<int, int> *_edges; 

	void Delaunay(const std::vector<Eigen::Vector3d>& pts, const double& limit_cube_len);
	void Voronoi();
};

#endif //P2M_VORONOITETGEN_H
