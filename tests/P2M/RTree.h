#ifndef P2M_RTREE_H
#define P2M_RTREE_H

#include <vector>
#include <Eigen/Dense>

class RTree
{
public:
	struct Node
	{
		bool leaf;
		int id;  // in leaf
		int l, r;
		Eigen::Vector3d box[2];
	};
	struct Region
	{
		int id;
		Eigen::Vector3d mid,box[2];
	};
	
	RTree(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &boxes);
	~RTree();

	// find all the bounding boxes where the query point is located in
	void locate(const Eigen::Vector3d& p, std::vector<int> &res)const;

private:
	int tot;
	Node* nodes;
	Region* regions;  // only used in construction

	// R-Tree construction
	int build(int l, int r);
};

#endif //P2M_RTREE_H
