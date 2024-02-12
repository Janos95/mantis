# P2M: A Fast Solver for Querying Distance from Point to Mesh Surface
This repository is the official code release for the paper "P2M: A Fast Solver for Querying Distance from Point to Mesh Surface". This paper is published in ACM Transactions on Graphics (SIGGRAPH 2023). 

Paper link: https://arxiv.org/abs/2308.16084

Doi: https://dl.acm.org/doi/10.1145/3592439

### Dependencies

- Eigen: https://eigen.tuxfamily.org/

  We use the template in Eigen as the basic data type of geometric information. Please download and configure this library. 

- TBB: https://github.com/wjakob/tbb

  We add CPU-based parallelism in the preprocessing stage of P2M.  Please download and configure this library. 

- Tetgen: https://wias-berlin.de/software/index.jsp?lang=1&id=TetGen

  We call tetgen in the construction of Delaunay tetrahedralization. We have already added the modified files (`tetgen.h`, `tetgen.cpp`, `predicates.h`) of tetgen in this repository. 

### Example

Here is an example. The code is tested on Windows 11 with Visual Studio 2019. 

```C++
#include "Model.h"

// specify the positions of the points
vector<Eigen::Vector3d> points;  

// specify the indexes of endpoints, leave this vector empty if there are no segments
vector<pair<int, int>> segments;   

// specify the indexes of vertices
vector<Eigen::Vector3i> faces;

// see the note below for details
double limit_cube_len = 1e3;

// construction
Model m(points, segments, faces, limit_cube_len);

// perform a closest point query
P2M_Result res;
Eigen::Vector3d query_point;
m.p2m(query_point, res);

```

Note: `limit_cube_len` is a limit value used in Voronoi diagram construction. Just ensure that the absolute coordinate values of mesh and query points are within `limit_cube_len`. If `limit_cube_len` is too large compared to the bounding box of the input mesh, there will be some precision problems in Delaunay tetrahedralization. If we find any degenerate tetrahedron, the program will stop and output an exception. Also, a smaller limit value will make the program slightly faster. 

In P2M, the input `points` are part of the mesh and will be considered in distant queries even if there exist isolated points, so remove the isolated ones if you don't want them. 

If you come across any problems with the code, please let me know.

### Citation

```
@article{Zong2023P2M,
author = {Zong, Chen and Xu, Jiacheng and Song, Jiantao and Chen, Shuangmin and Xin, Shiqing and Wang, Wenping and Tu, Changhe},
title = {P2M: A Fast Solver for Querying Distance from Point to Mesh Surface},
year = {2023},
issue_date = {August 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592439},
doi = {10.1145/3592439},
journal = {ACM Transactions on Graphics (TOG)},
month = {jul},
articleno = {147},
numpages = {13},
keywords = {distance query, bounding volume hierarchy (BVH), proximity query package (PQP), KD tree (KDT), convex polytope}
}
```

