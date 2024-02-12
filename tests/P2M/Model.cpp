#include <set>
#include <map>
#include <queue>
#include <algorithm>
#include <cfloat>
#include <thread>
#include "Model.h"
using namespace std;

constexpr int kMaxInterceptionListLength = 200;

inline bool same(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
	const double tol = 1e-14;
	if (fabs(a[0] - b[0]) < tol && fabs(a[1] - b[1]) < tol && fabs(a[2] - b[2]) < tol) return true;
	return false;
}

template<class F>
void parallel_for(size_t begin, size_t end, F f) {

    // serial implementation
    //for(size_t i = begin; i < end; ++i) {
    //    f(i);
    //}


    size_t num_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk_size = (end - begin + num_threads - 1) / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t thread_begin = begin + i * chunk_size;
        size_t thread_end = std::min(thread_begin + chunk_size, end);

        threads.emplace_back([thread_begin, thread_end, &f]() {
            for (size_t j = thread_begin; j < thread_end; ++j) {
                f(j);
            }
        });
    }

    for (auto &thread: threads) {
        thread.join();
    }
}

Model::Model(const vector<Eigen::Vector3d>& Points, const vector<pair<int, int> >& Segments,
	         const vector<Eigen::Vector3i>& Faces, const double& Limit_cube_len)
	:limit_cube_len(Limit_cube_len)
{
	get_mesh(Points, Segments, Faces);
	init();
}

Model::~Model()
{
	delete kdt;
}

// distance between two points
inline double dis_p2p(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2)
{
	return (p2 - p1).norm();
}

// distance between a point and the straight line of an edge
inline double dis_p2e(const Eigen::Vector3d& p, const Model::Edge& e, const vector<Eigen::Vector3d>& points)
{
	return (points[e.s] - p).cross(e.dir).norm();
}

// distance between a point and the plane of a face
inline double dis_p2f(const Eigen::Vector3d& p, const Model::Face& f, const vector<Eigen::Vector3d>& points)
{
	return fabs(f.n.dot(points[f.verts.x()] - p));
}

// squared distance between two points
inline double dis2_p2p(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2)
{
	return (p2 - p1).squaredNorm();
}

// squared distance between a point and the straight line of an edge
inline double dis2_p2e(const Eigen::Vector3d& p, const Model::Edge& e, const vector<Eigen::Vector3d>& points)
{
	return (points[e.s] - p).cross(e.dir).squaredNorm();
}

// squared distance between a point and the plane of a face
inline double dis2_p2f(const Eigen::Vector3d& p, const Model::Face& f, const vector<Eigen::Vector3d>& points)
{
	double d = f.n.dot(points[f.verts.x()] - p);
	return d * d;
}

// projection of a point onto the straight line of an edge
inline void proj_p2e(const Eigen::Vector3d& p, const Model::Edge& e,
	const vector<Eigen::Vector3d>& points, Eigen::Vector3d& res)
{
	Eigen::Vector3d a(points[e.s]), ap(p - a), ab(e.dir);
	res = a + ab.dot(ap) * ab;
}

// projection of a point onto the plane of a face
inline void proj_p2f(const Eigen::Vector3d& p, const Model::Face& f,
	const vector<Eigen::Vector3d>& points, Eigen::Vector3d& res)
{
	double t = f.n.dot(p - points[f.verts[0]]);
	res = p - f.n * t;
}

// check whether a point is inside the vertical space of an edge
// here we only check the vertical planes rooted at the endpoints of the edge
inline bool inside_space_e(const Eigen::Vector3d& p, const vector<Eigen::Vector4d>& vp)
{
	if (vp[0][0] * p[0] + vp[0][1] * p[1] + vp[0][2] * p[2] + vp[0][3] > 0) return false;
	if (vp[1][0] * p[0] + vp[1][1] * p[1] + vp[1][2] * p[2] + vp[1][3] > 0) return false;
	return true;
}

// check whether a point is inside the vertical space of a face
inline bool inside_space_f(const Eigen::Vector3d& p, const vector<Eigen::Vector4d>& vp)
{
	if (vp[0][0] * p[0] + vp[0][1] * p[1] + vp[0][2] * p[2] + vp[0][3] > 0) return false;
	if (vp[1][0] * p[0] + vp[1][1] * p[1] + vp[1][2] * p[2] + vp[1][3] > 0) return false;
	if (vp[2][0] * p[0] + vp[2][1] * p[1] + vp[2][2] * p[2] + vp[2][3] > 0) return false;
	return true;
}

// one of the condition is already checked before calling
inline bool inside_bbox(const Eigen::Vector3d& p, const pair<Eigen::Vector3d, Eigen::Vector3d>& bbox)
{
	if (p[0] < bbox.first[0]) return false;
	if (p[1] < bbox.first[1]) return false;
	if (p[1] > bbox.second[1]) return false;
	if (p[2] < bbox.first[2]) return false;
	if (p[2] > bbox.second[2]) return false;
	return true;
}

// check whether the coordinate values of a point is within the limit
inline bool in_limit_cube(const Eigen::Vector3d& p, const double& limit_cube_len)
{
	for (int i = 0;i < 3;i++) if (fabs(p[i]) > limit_cube_len) return false;
	return true;
}

inline bool cmp_x_max(const pair<int, double>& a, const pair<int, double>& b)
{
	return a.second > b.second;
}

void Model::p2m(const Eigen::Vector3d& p, P2M_Result& res)const
{
	if (!in_limit_cube(p, limit_cube_len))
	{
		printf("\nException: The query point is out of the limit cube. Program stopped!\n");
		exit(0);
	}
	// find the nearest vertex
	int site_id = kdt->NNS(p, res.kdt_nodes_visited);
	res.nn = points[site_id];
	res.intercepted_edges = (int)regions[site_id].edges.size();
	res.intercepted_faces = (int)regions[site_id].faces.size();
	res.primitive_type = 0;
	double ans2 = dis2_p2p(p, points[site_id]), ans = DBL_MAX;

	// check the intercepted primitives
	// If the number of primitives is over 200, we search the R-Tree to locate the query point.
	// Otherwise, we just do a binary search according to the max value of the bounding boxes 
	// in the x-dimension to quickly remove those useless primitives. 
	int min_e_id(-1), min_f_id(-1);
	if (res.intercepted_edges + res.intercepted_faces > kMaxInterceptionListLength)
	{
		double dis, dis2;
		vector<int> list;
		regions[site_id].T->locate(p, list);
		int siz = (int)list.size(), e_id, f_id;
		for (int i = 0;i < siz;i++)
		{
			if (list[i] < res.intercepted_edges)  // this primitive is an edge
			{
				e_id = regions[site_id].edges[list[i]];
				if (!inside_space_e(p, edges[e_id].vp)) continue;
				dis2 = dis2_p2e(p, edges[e_id], points);
				if (dis2 < ans2) ans2 = dis2, min_e_id = e_id;
			}
			else  // this primitive is a face
			{
				f_id = regions[site_id].faces[list[i] - res.intercepted_edges];
				if (!inside_space_f(p, faces[f_id].vp)) continue;
				dis = dis_p2f(p, faces[f_id], points);
				if (dis < ans) ans = dis, min_f_id = f_id;
			}
		}
	}
	else
	{
		int e_id, f_id;
		double dis, dis2;
		auto last = upper_bound(regions[site_id].x_max.begin(), regions[site_id].x_max.end(), make_pair(0, p[0]), cmp_x_max);
		for (auto it = regions[site_id].x_max.begin();it != last;++it)
		{
			if (it->first < res.intercepted_edges)  // this primitive is an edge
			{
				if (!inside_bbox(p, regions[site_id].box_e[it->first])) continue;
				e_id = regions[site_id].edges[it->first];
				if (!inside_space_e(p, edges[e_id].vp)) continue;
				dis2 = dis2_p2e(p, edges[e_id], points);
				if (dis2 < ans2) ans2 = dis2, min_e_id = e_id;
			}
			else  // this primitive is a face
			{
				if (!inside_bbox(p, regions[site_id].box_f[it->first - res.intercepted_edges])) continue;
				f_id = regions[site_id].faces[it->first - res.intercepted_edges];
				if (!inside_space_f(p, faces[f_id].vp)) continue;
				dis = dis_p2f(p, faces[f_id], points);
				if (dis < ans) ans = dis, min_f_id = f_id;
			}
		}
	}
	ans2 = sqrt(ans2);
	if (ans < ans2)  // the closest point is on a face
	{
		res.dis = ans;
		res.primitive_type = 2;
		proj_p2f(p, faces[min_f_id], points, res.cp);
	}
	else
	{
		res.dis = ans2;
		if (min_e_id != -1)  // the closest point is on an edge
		{
			res.primitive_type = 1;
			proj_p2e(p, edges[min_e_id], points, res.cp);
		}
		else res.primitive_type = 0, res.cp = res.nn;  // the closest point is a vertex
	}
}

// check whether the input mesh has duplicated points 
// and check whether the points are within the limit
bool point_deduplication(const vector<Eigen::Vector3d>& pts, int* new_id, 
	                     const double& limit_cube_len, vector<Eigen::Vector3d>& points)
{
	struct CmpPoint
	{
		bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const
		{
			if (a[1] == b[1] && a[2] == b[2]) return a[0] < b[0];
			return a[2] == b[2] ? a[1] < b[1] : a[2] < b[2];
		}
	};
	int siz = (int)pts.size();
	int num_dup(0);
	int* dup = new int[siz];
	memset(dup, -1, siz * sizeof(int));
	map<Eigen::Vector3d, int, CmpPoint> mp;
	for (int i = 0;i < siz;i++)
	{
		if (!in_limit_cube(pts[i], limit_cube_len))
		{
			printf("\nException: The input model is out of the limit cube. Program stopped!\n");
			exit(0);
		}
		auto res = mp.find(pts[i]);
		if (res != mp.end())
		{
			int j = res->second;
			num_dup++;
			dup[i] = j;
		}
		else mp[pts[i]] = i;
	}
	if (!num_dup)
	{
		points = pts;
		delete[] dup;
		return false;
	}
	points.resize(siz - num_dup);
	int cnt(0);
	for (int i = 0;i < siz;i++)
	{
		if (dup[i] == -1)
		{
			points[cnt] = pts[i];
			new_id[i] = cnt++;
		}
		else new_id[i] = new_id[dup[i]];
	}
	delete[] dup;
	return true;
}

void Model::get_mesh(const vector<Eigen::Vector3d>& Points, const vector<pair<int, int> >& Segments, const vector<Eigen::Vector3i>& Faces)
{
	struct CmpFace
	{
		bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const
		{
			if (a[1] == b[1] && a[2] == b[2]) return a[0] < b[0];
			return a[2] == b[2] ? a[1] < b[1] : a[2] < b[2];
		}
	};
	set<Eigen::Vector3i, CmpFace> in_faces;  // avoid duplication of faces
	set<pair<int, int>> in_edges;  // avoid duplication of edges
	int* new_id = new int[Points.size()];
	bool has_dup = point_deduplication(Points, new_id, limit_cube_len, points);
	const double eps = 1e-13;
	int siz = (int)Faces.size();
	int x, y, z, dim;
	pair<int, int> e;
	Eigen::Vector3d n;
	for (int i = 0;i < siz;i++)
	{
		x = Faces[i].x(), y = Faces[i].y(), z = Faces[i].z();
		if (has_dup) x = new_id[x], y = new_id[y], z = new_id[z];
		if (y > z) swap(y, z);
		if (x > y) swap(x, y);
		if (y > z) swap(y, z);
		Eigen::Vector3i f(x, y, z);
		if (in_faces.find(f) == in_faces.end())
		{
			n = (points[y] - points[x]).cross(points[z] - points[x]).normalized();
			if (n.norm() < eps)  // degenarate into segment
			{
				dim = 0;
				if (fabs(points[x][1] - points[y][1]) > eps) dim = 1;
				if (fabs(points[x][2] - points[y][2]) > eps) dim = 2;
				int min_v(x), max_v(x);
				double min_val(points[x][dim]), max_val(points[x][dim]);
				if (points[y][dim] < min_val) min_val = points[y][dim], min_v = y;
				if (points[z][dim] < min_val) min_val = points[z][dim], min_v = z;
				if (points[y][dim] > max_val) max_val = points[y][dim], max_v = y;
				if (points[z][dim] > max_val) max_val = points[z][dim], max_v = z;
				e = make_pair(min(min_v, max_v), max(min_v, max_v));
				if (in_edges.find(e) == in_edges.end()) edges.emplace_back(e.first, e.second), in_edges.insert(e);
				continue;
			}
			faces.emplace_back(f, n);
			in_faces.insert(f);
			pair<int, int> e1(x, y), e2(x, z), e3(y, z);
			if (in_edges.find(e1) == in_edges.end()) edges.emplace_back(x, y), in_edges.insert(e1);
			if (in_edges.find(e2) == in_edges.end()) edges.emplace_back(x, z), in_edges.insert(e2);
			if (in_edges.find(e3) == in_edges.end()) edges.emplace_back(y, z), in_edges.insert(e3);
		}
	}

	siz = (int)Segments.size();
	for (int i = 0;i < siz;i++)
	{
		x = Segments[i].first, y = Segments[i].second;
		if (has_dup) x = new_id[x], y = new_id[y];
		if (x == y) continue;
		if (x > y) swap(x, y);
		e = make_pair(x, y);
		if (in_edges.find(e) == in_edges.end()) edges.emplace_back(e.first, e.second), in_edges.insert(e);
	}
	delete[] new_id;
}

void Model::init()
{
	cal_info();
	vor = new VoronoiTetgen(points, limit_cube_len);
	cal_table();
	kdt = new KDTree(points);
}

void Model::cal_info()
{
	map<pair<int, int>, int> edge_id;
	int siz = (int)edges.size();
	for (int i = 0;i < siz;i++)
	{
		edge_id[make_pair(edges[i].s, edges[i].t)] = i;
		Eigen::Vector3d st = points[edges[i].t] - points[edges[i].s];
		edges[i].dir = st.normalized();

		// add vertical planes at the endpoints
		edges[i].vp.emplace_back(-st.x(), -st.y(), -st.z(), st.dot(points[edges[i].s]));
		edges[i].vp.emplace_back(st.x(), st.y(), st.z(), -st.dot(points[edges[i].t]));
	}
	siz = (int)faces.size();
	for (int i = 0;i < siz;i++)
	{
		int p1(faces[i].verts.x()), p2(faces[i].verts.y()), p3(faces[i].verts.z());
		Eigen::Vector3d a(points[p1]), b(points[p2]), c(points[p3]);
		Eigen::Vector3d n1((b - a).cross(faces[i].n)), n2((c - b).cross(faces[i].n)), n3((a - c).cross(faces[i].n));
		int e1(edge_id[make_pair(p1, p2)]), e2(edge_id[make_pair(p2, p3)]), e3(edge_id[make_pair(p1, p3)]);

		// add vertical planes between edges and faces 
		Eigen::Vector4d plane1(n1.x(), n1.y(), n1.z(), -n1.dot(a));
		Eigen::Vector4d plane2(n2.x(), n2.y(), n2.z(), -n2.dot(b));
		Eigen::Vector4d plane3(n3.x(), n3.y(), n3.z(), -n3.dot(c));
		faces[i].vp.emplace_back(plane1);
		faces[i].vp.emplace_back(plane2);
		faces[i].vp.emplace_back(plane3);
		// an edge may be connected to any number of faces, we record at most 8 vertical planes
		if (edges[e1].vp.size() < 8) edges[e1].vp.emplace_back(-plane1);
		if (edges[e2].vp.size() < 8) edges[e2].vp.emplace_back(-plane2);
		if (edges[e3].vp.size() < 8) edges[e3].vp.emplace_back(-plane3);
	}
}

// calculate the intersection point between the segment AB and the point-edge bisector surface
// Because of the precision issues, we apply an iterative method. 
inline void intersect_e(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& p,
	                    const Model::Edge& e, const vector<Eigen::Vector3d>& points, Eigen::Vector3d& res)
{
	const double tol = 1e-5; 
	double l(0), r(1), m;
	Eigen::Vector3d cur;
	int T = (int)log2((A - B).norm() / tol);
	if (T <= 0) T = 1;
	while (T--)
	{
		m = (l + r) / 2;
		cur = (1 - m) * B + m * A;
		if (dis2_p2p(cur, p) > dis2_p2e(cur, e, points)) r = m;
		else l = m;
	}
	res = (1 - l) * B + l * A;
}

// calculate the intersection point between the segment AB and the point-face bisector surface
// Because of the precision issues, we apply an iterative method. 
inline void intersect_f(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& p,
	                    const Model::Face& f, const vector<Eigen::Vector3d>& points, Eigen::Vector3d& res)
{
	const double tol = 1e-5;
	double l(0), r(1), m;
	Eigen::Vector3d cur;
	int T = (int)log2((A - B).norm() / tol);
	if (T <= 0) T = 1;
	while (T--)
	{
		m = (l + r) / 2;
		cur = (1 - m) * B + m * A;
		if (dis2_p2p(cur, p) > dis2_p2f(cur, f, points)) r = m;
		else l = m;
	}
	res = (1 - l) * B + l * A;
}

// calculate the interception table as well as the acceleration structures
void Model::cal_table()
{
	regions.resize(points.size());
	int siz = (int)edges.size();
	const int num_cells = (int)points.size();
	vector<vector<int> > tmp_edges;
	vector<vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> > tmp_box_e;
	tmp_edges.resize(siz);
	tmp_box_e.resize(siz);

	// for each edge, we flood from its endpoints
	parallel_for(0, siz,
		[&](int i)
		{
			{
				vector<Eigen::Vector3d> pts;
				vector<vector<int>> to_pts;
				queue<int> q;
				vector<bool> checked;
				vector<bool> out;
				Eigen::Vector3d intersection, box[2];
				checked.resize(points.size(), false);
				q.push(edges[i].s), checked[edges[i].s] = true;
				q.push(edges[i].t), checked[edges[i].t] = true;
				// flooding
				while (!q.empty())
				{
					int site_id = q.front();
					q.pop();
					pts.clear();
					to_pts.clear();
					// calculate the intersection between the Voronoi cell and the vertical space
					if (!vor->clip(site_id, edges[i].vp, pts, to_pts)) continue;
					// if the intersection region is not empty, calculate its bounding box
					bool intercept = false;
					box[0] = Eigen::Vector3d(DBL_MAX, DBL_MAX, DBL_MAX);
					box[1] = -box[0];
					out.clear();
					out.resize(pts.size(), false);
					// find the edges of the region which cross the bisector surface
					for (int j = 0;j < pts.size();j++)
					{
						double d2_p2p = dis2_p2p(pts[j], points[site_id]);
						if (d2_p2p > dis2_p2e(pts[j], edges[i], points))
						{
							intercept = true, out[j] = true;
							for (int w = 0;w < 3;w++) box[0][w] = min(box[0][w], pts[j][w]);
							for (int w = 0;w < 3;w++) box[1][w] = max(box[1][w], pts[j][w]);
						}
					}
					// calculate the intersection point between the region and the bisector surface
					if (!intercept) continue;
					for (int j = 0;j < pts.size();j++)
					{
						if (!out[j]) continue;
						for (int k = 0;k < to_pts[j].size();k++)
						{
							int id = to_pts[j][k];
							if (out[id]) continue;
							intersect_e(pts[j], pts[id], points[site_id], edges[i], points, intersection);
							for (int w = 0;w < 3;w++) box[0][w] = min(box[0][w], intersection[w]);
							for (int w = 0;w < 3;w++) box[1][w] = max(box[1][w], intersection[w]);
						}
					}
					// this edge is intercepted by the vertex
					tmp_edges[i].emplace_back(site_id);
					tmp_box_e[i].emplace_back(box[0], box[1]);
					for (int id : vor->cells[site_id].nei)
					{
						if (!checked[id] && id < num_cells)
							checked[id] = true, q.push(id);
					}
				}
			}
		});

	for (int i = 0;i < siz;i++) for (int j = 0;j < tmp_edges[i].size();j++)
	{
		int site_id(tmp_edges[i][j]);
		regions[site_id].edges.emplace_back(i);
		regions[site_id].box_e.emplace_back(tmp_box_e[i][j]);
	}

	siz = (int)faces.size();
	vector<vector<int> > tmp_faces;
	vector<vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> > tmp_box_f;
	tmp_faces.resize(siz);
	tmp_box_f.resize(siz);

	// for each face, we flood from its vertices
	parallel_for(0, siz,
		[&](int i)
		{
			{
				vector<Eigen::Vector3d> pts;
				vector<vector<int>> to_pts;
				queue<int> q;
				vector<bool> checked;
				vector<bool> out;
				Eigen::Vector3d intersection, box[2];
				checked.resize(points.size(), false);
				q.push(faces[i].verts.x()), checked[faces[i].verts.x()] = true;
				q.push(faces[i].verts.y()), checked[faces[i].verts.y()] = true;
				q.push(faces[i].verts.z()), checked[faces[i].verts.z()] = true;
				// flooding
				while (!q.empty())
				{
					int site_id = q.front();
					q.pop();
					pts.clear();
					to_pts.clear();
					// calculate the intersection between the Voronoi cell and the vertical space
					if (!vor->clip(site_id, faces[i].vp, pts, to_pts)) continue;
					// find the edges of the region which cross the bisector surface
					bool intercept = false;
					box[0] = Eigen::Vector3d(DBL_MAX, DBL_MAX, DBL_MAX);
					box[1] = -box[0];
					out.clear();
					out.resize(pts.size(), false);
					for (int j = 0;j < pts.size();j++)
					{
						double d2_p2p = dis2_p2p(pts[j], points[site_id]);
						if (d2_p2p > dis2_p2f(pts[j], faces[i], points))
						{
							intercept = true, out[j] = true;
							for (int w = 0;w < 3;w++) box[0][w] = min(box[0][w], pts[j][w]);
							for (int w = 0;w < 3;w++) box[1][w] = max(box[1][w], pts[j][w]);
						}
					}
					// calculate the intersection point between the region and the bisector surface
					if (!intercept) continue;
					for (int j = 0;j < pts.size();j++)
					{
						if (!out[j]) continue;
						for (int k = 0;k < to_pts[j].size();k++)
						{
							int id = to_pts[j][k];
							if (out[id]) continue;
							intersect_f(pts[j], pts[id], points[site_id], faces[i], points, intersection);
							for (int w = 0;w < 3;w++) box[0][w] = min(box[0][w], intersection[w]);
							for (int w = 0;w < 3;w++) box[1][w] = max(box[1][w], intersection[w]);
						}
					}
					// this face is intercepted by the vertex
					tmp_faces[i].emplace_back(site_id);
					tmp_box_f[i].emplace_back(box[0], box[1]);
					for (int id : vor->cells[site_id].nei)
					{
						if (!checked[id] && id < num_cells)
							checked[id] = true, q.push(id);
					}
				}
			}
		});

	for (int i = 0;i < siz;i++) for (int j = 0;j < tmp_faces[i].size();j++)
	{
		int site_id(tmp_faces[i][j]);
		regions[site_id].faces.emplace_back(i);
		regions[site_id].box_f.emplace_back(tmp_box_f[i][j]);
	}

	// For each point, we get an interception list including edges and faces.
	// If the length of the list is over 200, we construct an R-Tree 
	// to manage the bounding boxes of the interception regions.
	// Otherwise, we sort the bounding boxes according to the max value in the x-dimension.
	siz = (int)points.size();
	parallel_for(0, siz,
		[&](int i)
		{
			//for (int i = r.begin(); i < r.end(); i++)
			{
				int num_edges = (int)regions[i].edges.size();
				int tot = num_edges + (int)regions[i].faces.size();
				if (tot > kMaxInterceptionListLength)
				{
					vector<pair<Eigen::Vector3d, Eigen::Vector3d>> boxes;
					boxes.resize(tot);
					for (int j = 0;j < num_edges;j++) boxes[j] = regions[i].box_e[j];
					for (int j = num_edges;j < tot;j++) boxes[j] = regions[i].box_f[j - num_edges];
					regions[i].T = new RTree(boxes);
					vector<pair<Eigen::Vector3d, Eigen::Vector3d>>().swap(regions[i].box_e);
					vector<pair<Eigen::Vector3d, Eigen::Vector3d>>().swap(regions[i].box_f);
				}
				else
				{
					regions[i].x_max.resize(tot);
					for (int j = 0;j < num_edges;j++) regions[i].x_max[j] = make_pair(j, regions[i].box_e[j].second[0]);
					for (int j = num_edges;j < tot;j++) regions[i].x_max[j] = make_pair(j, regions[i].box_f[j - num_edges].second[0]);
					sort(regions[i].x_max.begin(), regions[i].x_max.end(), cmp_x_max);
				}
			}
		});
}
