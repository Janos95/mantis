#include <cfloat>
#include "KDTree.h"
using namespace std;
#define MAX3(a, b, c) max((a),max((b),(c)))

// compare the points according to the specified dimension
struct CmpPoint
{
	int dim;
	CmpPoint(int Dim) :dim(Dim) {}
	bool operator()(const pair<Eigen::Vector3d, int>& a, const pair<Eigen::Vector3d, int>& b) const
	{
		return a.first[dim] < b.first[dim];
	}
};

KDTree::KDTree(const vector<Eigen::Vector3d>& Points)
	:tot(0)
{
	int siz = (int)Points.size();
	nodes = new Node[siz + 1];
	pts = new pair<Eigen::Vector3d, int>[siz];
	for (int i = 0;i < siz;i++) pts[i]=make_pair(Points[i],i);
	build(0, siz - 1);
	delete[] pts;
}

KDTree::~KDTree()
{
	delete[] nodes;
}

int KDTree::NNS(const Eigen::Vector3d& p, int& nodes_visited)const
{
	nodes_visited = 0;
	int cur_min_point_id = -1;
	double cur_min_dist = DBL_MAX;
	find(nodes[1], p, cur_min_dist, cur_min_point_id, nodes_visited);
	return cur_min_point_id;
}

// find the dimension where the variance of the coordinates is the largest
inline int get_dim(int l, int r, const pair<Eigen::Vector3d, int>* pts)
{
	int ans = 0;
	double mx = 0, num = (double)(r - l + 1);
	Eigen::Vector3d sum(0, 0, 0), squ_sum(0, 0, 0), s(0, 0, 0);
	for (int i = l;i <= r;i++) for (int j = 0;j <= 2;j++) 
		sum[j] += pts[i].first[j], squ_sum[j] += pts[i].first[j] * pts[i].first[j];
	for (int j = 0;j <= 2;j++) s[j] = squ_sum[j] / num - (sum[j] / num) * (sum[j] / num);
	for (int i = 0;i <= 2;i++) if (s[i] > mx) mx = s[i], ans = i;
	return ans;
}

// calculate the bounding box of node k
inline void update(int k, KDTree::Node* nodes)
{
	int l = nodes[k].l, r = nodes[k].r;
	nodes[k].box[0] = nodes[k].box[1] = nodes[k].p;
	if (l)
	{
		for (int i = 0;i < 3;i++) nodes[k].box[0][i] = min(nodes[k].box[0][i], nodes[l].box[0][i]);
		for (int i = 0;i < 3;i++) nodes[k].box[1][i] = max(nodes[k].box[1][i], nodes[l].box[1][i]);
	}
	if (r)
	{
		for (int i = 0;i < 3;i++) nodes[k].box[0][i] = min(nodes[k].box[0][i], nodes[r].box[0][i]);
		for (int i = 0;i < 3;i++) nodes[k].box[1][i] = max(nodes[k].box[1][i], nodes[r].box[1][i]);
	}
}

int KDTree::build(int l, int r)
{
	if (l > r) return 0;
	int cur_dim = get_dim(l, r, pts), mid = (l + r) >> 1, k = ++tot;
	nth_element(pts + l, pts + mid, pts + r + 1, CmpPoint(cur_dim));
	nodes[k].p = pts[mid].first;
	nodes[k].mid_id = pts[mid].second;
	nodes[k].l = build(l, mid - 1);
	nodes[k].r = build(mid + 1, r);
	update(k, nodes);
	return k;
}

// calculate the minimum distance from a point to a bounding box
inline double p2bbox(const KDTree::Node& n, const Eigen::Vector3d& qp)
{
	Eigen::Vector3d dis;
	dis[0] = MAX3(n.box[0][0] - qp[0], 0.0, qp[0] - n.box[1][0]);
	dis[1] = MAX3(n.box[0][1] - qp[1], 0.0, qp[1] - n.box[1][1]);
	dis[2] = MAX3(n.box[0][2] - qp[2], 0.0, qp[2] - n.box[1][2]);
	return dis.squaredNorm();
}

void KDTree::find(const Node& n, const Eigen::Vector3d& qp, double& cur_min_dist, int& cur_min_point_id, int& nodes_visited)const
{
	nodes_visited++;
	double d = (qp - n.p).squaredNorm();
	if (d < cur_min_dist) cur_min_dist = d, cur_min_point_id = n.mid_id;
	double dl = n.l ? p2bbox(nodes[n.l], qp) : DBL_MAX;
	double dr = n.r ? p2bbox(nodes[n.r], qp) : DBL_MAX;
	// search the child node with more potential first
	if (dl < dr)
	{
		if (n.l && dl < cur_min_dist) find(nodes[n.l], qp, cur_min_dist, cur_min_point_id, nodes_visited);
		if (n.r && dr < cur_min_dist) find(nodes[n.r], qp, cur_min_dist, cur_min_point_id, nodes_visited);
	}
	else
	{
		if (n.r && dr < cur_min_dist) find(nodes[n.r], qp, cur_min_dist, cur_min_point_id, nodes_visited);
		if (n.l && dl < cur_min_dist) find(nodes[n.l], qp, cur_min_dist, cur_min_point_id, nodes_visited);
	}
}
