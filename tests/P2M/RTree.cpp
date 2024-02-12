#include "RTree.h"
using namespace std;

struct CmpRegion
{
	int dim;
	CmpRegion(int Dim) :dim(Dim) {}
	bool operator()(const RTree::Region& a, const RTree::Region& b) const
	{
		return a.mid[dim] < b.mid[dim];
	}
};

RTree::RTree(const vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes)
	:tot(0)
{
	const int num_boxes = (int)boxes.size();
	nodes = new Node[num_boxes * 2];
	regions = new Region[num_boxes];
	for (int i = 0;i < num_boxes;i++)
	{
		regions[i].id = i;
		regions[i].box[0] = boxes[i].first;
		regions[i].box[1] = boxes[i].second;
		regions[i].mid = regions[i].box[0] + regions[i].box[1];
	}
	build(0, num_boxes - 1);
	delete[] regions;
}

RTree::~RTree()
{
	delete[] nodes;
}

// check whether a point is inside a bounding box
inline bool inside(const Eigen::Vector3d& p, const RTree::Node& n)
{
	if (p[0]<n.box[0][0] || p[0]>n.box[1][0]) return false;
	if (p[1]<n.box[0][1] || p[1]>n.box[1][1]) return false;
	if (p[2]<n.box[0][2] || p[2]>n.box[1][2]) return false;
	return true;
}

void RTree::locate(const Eigen::Vector3d& p, vector<int>& res)const
{
	int* q = new int[tot];
	q[0] = 1;
	int i, l(0), r(0);
	while (l <= r)
	{
		i = q[l++];
		if (nodes[i].leaf)
		{
			if (inside(p, nodes[i])) res.emplace_back(nodes[i].id);
			continue;
		}
		if (inside(p, nodes[nodes[i].l])) q[++r] = nodes[i].l;
		if (inside(p, nodes[nodes[i].r])) q[++r] = nodes[i].r;
	}
	delete[] q;
}

// find the dimension where the variance of the box center coordinates is the largest
inline int get_dim(int l, int r,const RTree::Region* regions)
{
	int ans = 0;
	double mx = 0, num = (double)(r - l + 1);
	Eigen::Vector3d sum(0, 0, 0), squ_sum(0, 0, 0), s(0, 0, 0);
	for (int i = l;i <= r;i++) for (int j = 0;j <= 2;j++) 
		sum[j] += regions[i].mid[j], squ_sum[j] += regions[i].mid[j] * regions[i].mid[j];
	for (int j = 0;j <= 2;j++) s[j] = squ_sum[j] / num - (sum[j] / num) * (sum[j] / num);
	for (int i = 0;i <= 2;i++) if (s[i] > mx) mx = s[i], ans = i;
	return ans;
}

inline void upd(int k, RTree::Node* nodes)
{
	int l = nodes[k].l, r = nodes[k].r;
	nodes[k].box[0][0] = min(nodes[l].box[0][0], nodes[r].box[0][0]);
	nodes[k].box[0][1] = min(nodes[l].box[0][1], nodes[r].box[0][1]);
	nodes[k].box[0][2] = min(nodes[l].box[0][2], nodes[r].box[0][2]);
	nodes[k].box[1][0] = max(nodes[l].box[1][0], nodes[r].box[1][0]);
	nodes[k].box[1][1] = max(nodes[l].box[1][1], nodes[r].box[1][1]);
	nodes[k].box[1][2] = max(nodes[l].box[1][2], nodes[r].box[1][2]);
}

int RTree::build(int l, int r)
{
	if (l > r) return 0;
	int k = ++tot;
	if (l == r)
	{
		nodes[k].id = regions[l].id;
		nodes[k].leaf = true;
		nodes[k].box[0] = regions[l].box[0];
		nodes[k].box[1] = regions[l].box[1];
		return k;
	}
	int cur_dim = get_dim(l, r, regions), mid = (l + r) >> 1;
	nth_element(regions + l, regions + mid, regions + r + 1, CmpRegion(cur_dim));
	nodes[k].leaf = false;
	nodes[k].l = build(l, mid);
	nodes[k].r = build(mid + 1, r);
	upd(k,nodes);
	return k;
}
