#ifndef P2M_KDTREE_H
#define P2M_KDTREE_H

#include <vector>
#include <Eigen/Dense>

class KDTree
{
public:
	struct Node
	{
		int l, r, mid_id;
		Eigen::Vector3d p,box[2];
	};

	KDTree(const std::vector<Eigen::Vector3d>& Points);
	~KDTree();

	// nearest neighbour search
	int NNS(const Eigen::Vector3d& p, int& nodes_visited) const;

private:
	int tot;
	Node* nodes;
	std::pair<Eigen::Vector3d, int>* pts;  // only used in KDT construction
	
	// KDT construction
	int build(int l, int r);

	// recursively traverse the tree to find the nearest point
	void find(const Node& n, const Eigen::Vector3d& qp,
			  double& cur_min_dist, int& cur_min_point_id, int& nodes_visited) const;
};

#endif //P2M_KDTREE_H
