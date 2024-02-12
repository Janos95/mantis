#include <unordered_map>
#include "VoronoiTetgen.h"
using namespace std;

VoronoiTetgen::VoronoiTetgen(const vector<Eigen::Vector3d>& pts, const double& limit_cube_len)
{
	Delaunay(pts, limit_cube_len);
	Voronoi();
}

VoronoiTetgen::~VoronoiTetgen()
{
	delete[] _verts;
	delete[] _edges;
	delete[] cells;
}

// construction of Delaunay tetrahedralization by tetgen
inline void VoronoiTetgen::Delaunay(const std::vector<Eigen::Vector3d>& pts, const double& limit_cube_len)
{
	char paras[] = "";
	tetgenbehavior arguments;
	arguments.parse_commandline(paras);
	m.b = &arguments;
	tetgenio in;
	m.in = &in;
	vector<Eigen::Vector3d> points = pts;
	// add 8 virtual points in the construction of Delaunay tetrahedralization
	// the last 4 points will be used to create the initial Delaunay tetrahedralization
	double l = limit_cube_len * 2;
	points.emplace_back(l, l, l);
	points.emplace_back(-l, l, l);
	points.emplace_back(l, -l, l);
	points.emplace_back(l, l, -l);
	points.emplace_back(-l, -l, l);
	points.emplace_back(-l, l, -l);
	points.emplace_back(l, -l, -l);
	points.emplace_back(-l, -l, -l);
	// construct Delaunay by tetgen
	// the results may have duplication
	m.in->load_sample_pts(points);
	m.initializepools();
	m.transfernodes();
	m.incrementaldelaunay();
}

// calculate the Voronoi diagram
inline void VoronoiTetgen::Voronoi()
{
	tetgenmesh::triface tetloop, worktet, spintet, firsttet;
	tetgenmesh::point pt[4];
	REAL ccent[3];
	long ntets, faces;
	int* indexarray, * fidxs;
	int vpointcount, vedgecount, vfacecount;
	int ishullface, end1, end2, i(0), t1ver;
	indexarray = new int[m.tetrahedrons->items * 10];
	m.tetrahedrons->traversalinit();
	tetloop.tet = m.alltetrahedrontraverse();
	while (tetloop.tet != NULL) {
		tetloop.tet[11] = (tetgenmesh::tetrahedron) & (indexarray[i * 10]);
		i++;
		tetloop.tet = m.alltetrahedrontraverse();
	}
	// the number of tetrahedra
	ntets = m.tetrahedrons->items - m.hullsize;
	// the number of Delaunay faces (Voronoi edges)
	faces = (4l * ntets + m.hullsize) / 2l;
	// get Voronoi vertices
	_verts = new Eigen::Vector3d[ntets];
	m.tetrahedrons->traversalinit();
	tetloop.tet = m.tetrahedrontraverse();
	vpointcount = 0;
	while (tetloop.tet != (tetgenmesh::tetrahedron*)NULL) {
		for (i = 0; i < 4; i++) {
			pt[i] = (tetgenmesh::point)tetloop.tet[4 + i];
			m.setpoint2tet(pt[i], m.encode(tetloop));
		}
		if (!m.circumsphere(pt[0], pt[1], pt[2], pt[3], ccent, NULL))
		{
			printf("\nException: A degenerate tetrahedron is found after Delaunay tetrahedralization. Program stopped!\n");
			exit(0);
		}
		_verts[vpointcount] = Eigen::Vector3d(ccent[0], ccent[1], ccent[2]);
		m.setelemindex(tetloop.tet, vpointcount);
		vpointcount++;
		tetloop.tet = m.tetrahedrontraverse();
	}
	// get Voronoi edges
	_edges = new pair<int, int>[faces]; 
	m.tetrahedrons->traversalinit();
	tetloop.tet = m.tetrahedrontraverse();
	vedgecount = 0; 
	while (tetloop.tet != (tetgenmesh::tetrahedron*)NULL) {
		end1 = m.elemindex(tetloop.tet);
		for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++) {
			m.fsym(tetloop, worktet);
			if (m.ishulltet(worktet) ||
				(m.elemindex(tetloop.tet) < m.elemindex(worktet.tet))) {
				if (!m.ishulltet(worktet)) end2 = m.elemindex(worktet.tet);
				else end2 = -1;
				_edges[vedgecount].first = end1, _edges[vedgecount].second = end2;
				fidxs = (int*)(tetloop.tet[11]);
				fidxs[tetloop.ver] = vedgecount;
				fidxs = (int*)(worktet.tet[11]);
				fidxs[worktet.ver & 3] = vedgecount;
				vedgecount++;
			}
		} 
		tetloop.tet = m.tetrahedrontraverse();
	}
	// traverse all facets and construct cells
	int nsites = m.points->items - m.unuverts - m.dupverts;
	unordered_map<int, pair<int, int>>* tmp_edges = new unordered_map<int, pair<int, int>>[nsites];
	cells = new Cell[nsites];
	num_cells = nsites - 8;
	m.tetrahedrons->traversalinit();
	tetloop.tet = m.tetrahedrontraverse();
	vfacecount = 0; 
	while (tetloop.tet != (tetgenmesh::tetrahedron*)NULL) {
		worktet.tet = tetloop.tet;
		for (i = 0; i < 6; i++) {
			worktet.ver = m.edge2ver[i];
			firsttet = worktet;
			spintet = worktet;
			while (1) {
				t1ver = (spintet).ver;
				m.decode((spintet).tet[m.facepivot1[(spintet).ver]], (spintet));
				(spintet).ver = m.facepivot2[t1ver][(spintet).ver];
				if (spintet.tet == worktet.tet) break;
				if (!m.ishulltet(spintet)) {
					if (m.elemindex(spintet.tet) < m.elemindex(worktet.tet)) break;
				}
				else {
					if (m.apex(spintet) == m.dummypoint) {
						m.fnext(spintet, firsttet);
					}
				}
			} 
			if (spintet.tet == worktet.tet) {
				pt[0] = m.org(worktet);
				pt[1] = m.dest(worktet);
				end1 = m.pointmark(pt[0]) - m.in->firstnumber;
				end2 = m.pointmark(pt[1]) - m.in->firstnumber;
				if (end1 < num_cells && end2 < num_cells)
				{
					cells[end2].nei.emplace_back(end1);
					cells[end1].nei.emplace_back(end2);
				}
				spintet = firsttet;
				bool is_finite = true;
				while (1) {
					fidxs = (int*)(spintet.tet[11]);
					if (m.apex(spintet) != m.dummypoint) {
						vedgecount = fidxs[spintet.ver & 3];
						ishullface = 0;
					}
					else {
						ishullface = 1; 
						is_finite = false;
					}
					// add this face to its two sites
					int e_id = !ishullface ? vedgecount : -1;
					if (tmp_edges[end1].find(e_id) == tmp_edges[end1].end()) tmp_edges[end1][e_id].first = vfacecount;
					else tmp_edges[end1][e_id].second = vfacecount;
					if (tmp_edges[end2].find(e_id) == tmp_edges[end2].end()) tmp_edges[end2][e_id].first = vfacecount;
					else tmp_edges[end2][e_id].second = vfacecount;

					t1ver = (spintet).ver;
					m.decode((spintet).tet[m.facepivot1[(spintet).ver]], (spintet));
					(spintet).ver = m.facepivot2[t1ver][(spintet).ver];
					if (spintet.tet == firsttet.tet) break;
				} 
				vfacecount++;
			}
		} 
		tetloop.tet = m.tetrahedrontraverse();
	}
	delete[] indexarray;
	for (int i = 0;i < num_cells;i++)
	{
		int j = 0;
		cells[i].edges.resize(tmp_edges[i].size());
		for (auto it = tmp_edges[i].begin();it != tmp_edges[i].end();++it)
		{
			cells[i].edges[j].e = it->first;
			cells[i].edges[j].f1 = it->second.first;
			cells[i].edges[j++].f2 = it->second.second;
		}
	}
	delete[] tmp_edges;
}

bool VoronoiTetgen::clip(int cell_id, const vector<Eigen::Vector4d>& ps, 
	                            vector<Eigen::Vector3d>& res_pts, vector<vector<int>>& to_pts)
{
	vector<Eigen::Vector3d> new_verts;
	vector<pair<int, int>> new_edges;
	vector<pair<CEdge, bool>> edges;  // (e_info,survive)
	new_verts.emplace_back(Eigen::Vector3d());
	new_edges.emplace_back(pair<int, int>());
	int siz = (int)cells[cell_id].edges.size();
	edges.resize(siz);
	int f_cnt = 0;
	for (int i = 0;i < siz;i++) edges[i] = make_pair(cells[cell_id].edges[i], true);
	vector<pair<int, pair<int, int>>> cut_pts;  // p_id,(f1,f2)
	pair<int, int> e;
	Eigen::Vector3d p1, p2, cut;
	unordered_map<int, int> f_id;
	for (auto plane : ps)
	{
		bool flag = false;
		cut_pts.clear();
		// clip each edge by the plane
		for (auto& cur_edge : edges)
		{
			if (!cur_edge.second) continue;
			auto e_info = &cur_edge.first;
			flag = true;
			e = e_info->e >= 0 ? _edges[e_info->e] : new_edges[-e_info->e];
			p1 = e.first >= 0 ? _verts[e.first] : new_verts[-e.first];
			p2 = e.second >= 0 ? _verts[e.second] : new_verts[-e.second];
			double f1 = plane[0] * p1[0] + plane[1] * p1[1] + plane[2] * p1[2] + plane[3];
			double f2 = plane[0] * p2[0] + plane[1] * p2[1] + plane[2] * p2[2] + plane[3];
			const double tol = 0;
			if (f1 < tol)
			{
				if (f2 < tol) continue;
				else  // f2 >= tol
				{
					int cut_id;
					cut = p1 + (p2 - p1) * (fabs(f1) / (fabs(f1) + fabs(f2))); 
					cut_id = -(int)new_verts.size();
					new_verts.emplace_back(cut);
					cut_pts.emplace_back(cut_id, make_pair(e_info->f1, e_info->f2));

					if (e_info->e >= 0)
					{
						e_info->e = -(int)new_edges.size();
						new_edges.emplace_back(e.first, cut_id);
					}
					else new_edges[-e_info->e].second = cut_id;
				}
			}
			else  // f1 >= tol
			{
				if (f2 >= tol)
				{
					cur_edge.second = false;
					continue;
				}
				else // f2 < tol
				{
					int cut_id;
					cut = p1 + (p2 - p1) * (fabs(f1) / (fabs(f1) + fabs(f2))); 
					cut_id = -(int)new_verts.size();
					new_verts.emplace_back(cut);
					cut_pts.emplace_back(cut_id, make_pair(e_info->f1, e_info->f2));

					if (e_info->e >= 0)
					{
						e_info->e = -(int)new_edges.size();
						new_edges.emplace_back(e.second, cut_id);
					}
					else new_edges[-e_info->e].first = cut_id;
				}
			}
		}
		if (!flag) return false;
		// get new edges by intersection points
		{
			f_id.clear();
			f_cnt++;
			for (auto it = cut_pts.begin();it != cut_pts.end();++it)
			{
				auto ite = f_id.find(it->second.first);
				if (ite == f_id.end()) f_id[it->second.first] = it->first;
				else
				{
					edges.emplace_back(CEdge(-(int)new_edges.size(), -f_cnt, it->second.first), true);
					new_edges.emplace_back(ite->second, it->first);
				}
				ite = f_id.find(it->second.second);
				if (ite == f_id.end()) f_id[it->second.second] = it->first;
				else
				{
					edges.emplace_back(CEdge(-(int)new_edges.size(), -f_cnt, it->second.second), true);
					new_edges.emplace_back(ite->second, it->first);
				}
			}
		}
	}
	// get the result vertices and their adjacency relation
	unordered_map<int, int> mp;
	for (auto cur_edge : edges)
	{
		if (!cur_edge.second) continue;
		auto e_info = cur_edge.first;
		e = e_info.e >= 0 ? _edges[e_info.e] : new_edges[-e_info.e];
		p1 = e.first >= 0 ? _verts[e.first] : new_verts[-e.first];
		p2 = e.second >= 0 ? _verts[e.second] : new_verts[-e.second];
		if (mp.find(e.first) == mp.end()) mp[e.first] = (int)res_pts.size(), res_pts.emplace_back(p1), to_pts.emplace_back(vector<int>());
		if (mp.find(e.second) == mp.end()) mp[e.second] = (int)res_pts.size(), res_pts.emplace_back(p2), to_pts.emplace_back(vector<int>());
		int id1 = mp[e.first], id2 = mp[e.second];
		to_pts[id1].emplace_back(id2);
		to_pts[id2].emplace_back(id1);
	}
	if (res_pts.empty()) return false;
	return true;
}
