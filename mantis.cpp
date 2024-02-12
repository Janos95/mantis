#include "mantis.h"
#include "Delaunay_psm.h"

#include <arm_neon.h>

#include <queue>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <thread>

//#define DEBUG_MANTIS

namespace mantis {

using index_t = GEO::index_t;

// ============================= MISC STRUCTS ===============================

struct PackedEdge {
    float32x4_t min_x;
    float32x4_t start[3];
    float32x4_t dir[3];
    float32x4_t dir_len_squared;
    int32x4_t primitive_idx;
};

struct PackedFace {
    float32x4_t min_x;
    float32x4_t face_plane[4];
    float32x4_t edge_plane0[4];
    float32x4_t edge_plane1[4];
    float32x4_t edge_plane2[4];
    int32x4_t primitive_idx;
};


struct FaceData {
    // Plane coefficients of the face plane. Normal is of unit length.
    GEO::vec4 face_plane;
    // Edge plane at index i is the plane that contains the edge opposite to vertex i.
    // Note that edge planes have to be oriented inwards, i.e. the normal is pointing to the
    // inside of the triangle.
    GEO::vec4 clipping_planes[3];

    GEO::vec3 pt_on_plane;
};

struct EdgeData {
    uint32_t start = uint32_t(-1), end = uint32_t(-1);
    uint32_t left_face = uint32_t(-1), right_face = uint32_t(-1);
    GEO::vec4 clipping_planes[4];
    int num_planes = 0;
};

struct BoundingBox {
    GEO::vec3 lower = GEO::vec3{std::numeric_limits<double>::max()};
    GEO::vec3 upper = GEO::vec3{-std::numeric_limits<double>::max()};

    void extend(const GEO::vec3 &pt) {
        lower = {std::min(lower.x, pt.x), std::min(lower.y, pt.y), std::min(lower.z, pt.z)};
        upper = {std::max(upper.x, pt.x), std::max(upper.y, pt.y), std::max(upper.z, pt.z)};
    }

    void extend(const BoundingBox &box) {
        extend(box.lower);
        extend(box.upper);
    }
};

struct Node {
    float32x4_t minCorners[3]; // x, y, z minimum corners for 4 boxes
    float32x4_t maxCorners[3]; // x, y, z maximum corners for 4 boxes
    int32x4_t children;
};

struct LeafNode {
    float32x4_t x_coords = vdupq_n_f32(FLT_MAX);
    float32x4_t y_coords = vdupq_n_f32(FLT_MAX);
    float32x4_t z_coords = vdupq_n_f32(FLT_MAX);
    int32x4_t indices = vdupq_n_u32(uint32_t(-1));
};

// ============================= SIMD ===============================

// a*b + c
float32x4_t fma(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vmlaq_f32(c, a, b);
}

float32x4_t min(float32x4_t a, float32x4_t b) {
    return vminq_f32(a, b);
}

uint32x4_t leq(float32x4_t a, float32x4_t b) {
    return vcleq_f32(a, b);
}

uint32x4_t logical_and(uint32x4_t a, uint32x4_t b) {
    return vandq_u32(a, b);
}

int32x4_t select(int32x4_t condition, int32x4_t trueValue, int32x4_t falseValue) {
    return vbslq_s32(condition, trueValue, falseValue);
}

float32x4_t dup_float(float x) {
    return vdupq_n_f32(x);
}

int32x4_t dup_int(int32_t x) {
    return vdupq_n_s32(x);
}

float32x4_t dot(float32x4_t ax, float32x4_t ay, float32x4_t az,
                float32x4_t bx, float32x4_t by, float32x4_t bz) {
    float32x4_t result = vmulq_f32(ax, bx);
    result = fma(ay, by, result);
    result = fma(az, bz, result);
    return result;
}

float32x4_t length_squared(float32x4_t x, float32x4_t y, float32x4_t z) {
    float32x4_t result = vmulq_f32(x, x);
    result = fma(y, y, result);
    result = fma(z, z, result);
    return result;
}

float32x4_t distance_squared(float32x4_t ax, float32x4_t ay, float32x4_t az,
                             float32x4_t bx, float32x4_t by, float32x4_t bz) {
    float32x4_t dx = vsubq_f32(ax, bx);
    float32x4_t dy = vsubq_f32(ay, by);
    float32x4_t dz = vsubq_f32(az, bz);
    return length_squared(dx, dy, dz);
}

float32x4_t eval_plane(float32x4_t px, float32x4_t py, float32x4_t pz, float32x4_t plane_x, float32x4_t plane_y,
                       float32x4_t plane_z, float32x4_t plane_w) {
    float32x4_t result = vmulq_f32(px, plane_x);
    result = fma(py, plane_y, result);
    result = fma(pz, plane_z, result);
    return vaddq_f32(result, plane_w);
}

inline float32x4_t p2bbox(const Node &node, const float32x4_t qx, const float32x4_t qy, const float32x4_t qz) {
    // Compute distances in x, y, z directions and clamp them to zero if they are negative
    float32x4_t dx = vmaxq_f32(vsubq_f32(node.minCorners[0], qx), vsubq_f32(qx, node.maxCorners[0]));
    dx = vmaxq_f32(dx, vdupq_n_f32(0.0f));
    float32x4_t dy = vmaxq_f32(vsubq_f32(node.minCorners[1], qy), vsubq_f32(qy, node.maxCorners[1]));
    dy = vmaxq_f32(dy, vdupq_n_f32(0.0f));
    float32x4_t dz = vmaxq_f32(vsubq_f32(node.minCorners[2], qz), vsubq_f32(qz, node.maxCorners[2]));
    dz = vmaxq_f32(dz, vdupq_n_f32(0.0f));
    // Compute squared distances for each box
    float32x4_t squaredDist = vaddq_f32(vmulq_f32(dx, dx), vaddq_f32(vmulq_f32(dy, dy), vmulq_f32(dz, dz)));
    return squaredDist;
}

// ============================= BVH ===============================

#define cmin(a, b) distances[a] > distances[b] ? b : a
#define cmax(a, b) distances[a] > distances[b] ? a : b

#define cswap(a, b)  \
    {int tmp = a;    \
    a = cmax(a,b);   \
    b = cmin(tmp, b);}

#define nsort4(a, b, c, d) \
    do                     \
    {                      \
        cswap(a, b);       \
        cswap(c, d);       \
        cswap(a, c);       \
        cswap(b, d);       \
        cswap(b, c);       \
    } while (0)

constexpr static long long NUM_PACKETS = 8;

class Bvh {
public:
    void updateClosestPoint(const float32x4_t &pt_x,
                            const float32x4_t &pt_y,
                            const float32x4_t &pt_z,
                            size_t firstPacket,
                            size_t numPackets,
                            float &bestDistSq,
                            int &bestIdx) const {
        float32x4_t minDist = vdupq_n_f32(bestDistSq);
        int32x4_t minIdx = vdupq_n_s32(bestIdx);

        for (size_t i = firstPacket; i < firstPacket + numPackets; ++i) {
            // Compute squared distances for a batch of 4 points
            const auto &leaf = m_leaves[i];
            float32x4_t dx = vsubq_f32(pt_x, leaf.x_coords);
            float32x4_t dy = vsubq_f32(pt_y, leaf.y_coords);
            float32x4_t dz = vsubq_f32(pt_z, leaf.z_coords);
            float32x4_t distSq = vmlaq_f32(vmlaq_f32(vmulq_f32(dx, dx), dy, dy), dz, dz);

            // Comparison mask for distances
            // if distSq >= minDist => keep minDist
            uint32x4_t keepMinDist = vcgeq_f32(distSq, minDist);
            minDist = vminq_f32(minDist, distSq);

            // Update the indices
            minIdx = vbslq_s32(keepMinDist, minIdx, leaf.indices);
        }

        // Find overall minimum distance and index
        for (int j = 0; j < 4; ++j) {
            if (minDist[j] < bestDistSq) {
                bestDistSq = minDist[j];
                bestIdx = minIdx[j];
            }
        }
    }

    explicit Bvh(const std::vector<GEO::vec3> &points) {
        // Initialize the original_points vector
        original_points.resize(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            original_points[i] = points[i];
        }

        // Create an index array for all points
        std::vector<int> indices(points.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Build the KD-tree
        BoundingBox box;
        int node_idx = constructTree(indices, 0, indices.size(), 0, box);
        assert(node_idx == 0 || node_idx < 0);
    }

    int closestPoint(const GEO::vec3 &q) const {
        constexpr int MAX_STACK_SIZE = 64;
        struct StackNode {
            int nodeIndex;
            float minDistSq;
        };
        StackNode stack[MAX_STACK_SIZE];
        int stackSize = 0;

        float bestDistSq = std::numeric_limits<float>::max();
        int bestIdx = -1;

        // Broadcast query point coordinates to SIMD size
        float32x4_t q_x = vdupq_n_f32(q.x);
        float32x4_t q_y = vdupq_n_f32(q.y);
        float32x4_t q_z = vdupq_n_f32(q.z);

        // Start with the root node
        stack[stackSize++] = {0, 0.0f};
        if (m_nodes.empty()) {
            stack[0].nodeIndex = -1;
        }

        while (stackSize > 0) {
            StackNode current = stack[--stackSize];
            if (current.minDistSq >= bestDistSq) {
                continue;  // Skip nodes that can't possibly contain a closer point
            }
            if (current.nodeIndex < 0) {
                auto [begin, numPackets] = m_leafRange[-(current.nodeIndex + 1)];
                updateClosestPoint(q_x, q_y, q_z, begin, numPackets, bestDistSq, bestIdx);
                continue;
            }

            const Node &node = m_nodes[current.nodeIndex];

            // Compute distances to each child
            float32x4_t distances = p2bbox(node, q_x, q_y, q_z);
            int childIndices[4] = {0, 1, 2, 3};

            // Sort children by distance
            nsort4(childIndices[0], childIndices[1], childIndices[2], childIndices[3]);

            // push children that are internal
            for (int idx: childIndices) {
                int childIdx = node.children[idx];
                float childDist = distances[idx];
                if (childDist < bestDistSq) {
                    assert(stackSize + 1 < MAX_STACK_SIZE);
                    stack[stackSize++] = {childIdx, childDist};
                }
            }
        }

        return bestIdx;
    }

private:
    std::vector<GEO::vec3> original_points;

    std::vector<Node> m_nodes;
    std::vector<LeafNode> m_leaves;
    std::vector<std::pair<int, int>> m_leafRange;

    int constructTree(std::vector<int> &indices, size_t begin, size_t end, size_t depth, BoundingBox &box) {
        if (end - begin <= NUM_PACKETS * 4) {
            // Update the bounding box for this leaf node
            box = BoundingBox();
            for (size_t i = begin; i < end; ++i) {
                int idx = indices[i];
                box.extend(original_points[idx]);
            }

            int leafIdx = int(m_leafRange.size());
            auto firstLeaf = int(m_leaves.size());
            auto numPackets = int((end - begin + 3) / 4);
            m_leafRange.emplace_back(firstLeaf, numPackets);

            for (int i = 0; i < numPackets; ++i) {
                LeafNode leaf{};
                for (size_t j = 0; j < 4; ++j) {
                    size_t k = i * 4 + j;
                    if (k < end - begin) {
                        leaf.x_coords[j] = (float) original_points[indices[begin + k]].x;
                        leaf.y_coords[j] = (float) original_points[indices[begin + k]].y;
                        leaf.z_coords[j] = (float) original_points[indices[begin + k]].z;
                        leaf.indices[j] = (int) (indices[begin + k]);
                    }
                }
                m_leaves.push_back(leaf);
            }

            // Return negative index to indicate leaf node
            return -(leafIdx + 1);
        }

        Node node{};

        // Split dimensions: Choose different dimensions for each split
        size_t primaryDim = depth % 3;
        size_t secondaryDim = (primaryDim + 1) % 3; // Choose next dimension for secondary split

        // Primary split
        size_t primarySplit = (begin + end) / 2;
        std::nth_element(indices.begin() + (long) begin, indices.begin() + (long) primarySplit,
                         indices.begin() + (long) end,
                         [primaryDim, this](int i1, int i2) {
                             return original_points[i1][primaryDim] < original_points[i2][primaryDim];
                         });

        // Secondary splits
        size_t secondarySplit1 = (begin + primarySplit) / 2;
        size_t secondarySplit2 = (primarySplit + end) / 2;

        std::nth_element(indices.begin() + (long) begin, indices.begin() + (long) secondarySplit1,
                         indices.begin() + (long) primarySplit,
                         [secondaryDim, this](int i1, int i2) {
                             return original_points[i1][secondaryDim] < original_points[i2][secondaryDim];
                         });

        std::nth_element(indices.begin() + (long) primarySplit, indices.begin() + (long) secondarySplit2,
                         indices.begin() + (long) end,
                         [secondaryDim, this](int i1, int i2) {
                             return original_points[i1][secondaryDim] < original_points[i2][secondaryDim];
                         });

        BoundingBox childBoxes[4] = {};

        auto node_idx = int(m_nodes.size());
        m_nodes.emplace_back();

        node.children[0] = constructTree(indices, begin, secondarySplit1, depth + 2, childBoxes[0]);
        node.children[1] = constructTree(indices, secondarySplit1, primarySplit, depth + 2, childBoxes[1]);
        node.children[2] = constructTree(indices, primarySplit, secondarySplit2, depth + 2, childBoxes[2]);
        node.children[3] = constructTree(indices, secondarySplit2, end, depth + 2, childBoxes[3]);

        // set bounding boxes of node
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                node.minCorners[i][j] = (float) childBoxes[j].lower[i];
                node.maxCorners[i][j] = (float) childBoxes[j].upper[i];
            }
        }

        // Combine bounding boxes from children
        box = childBoxes[0];
        for (int i = 1; i < 4; ++i) {
            box.extend(childBoxes[i]);
        }

        m_nodes[node_idx] = node;
        return node_idx;
    }
};

// ============================= UTILS ==================================

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

// ============================= GEOMETRY UTILS ===============================

GEO::vec4 to_vec4(GEO::vec3 v, double w) {
    return {v.x, v.y, v.z, w};
}

GEO::vec3 to_vec3(GEO::vec4 v) {
    return {v.x, v.y, v.z};
}

double eval_plane(GEO::vec4 plane, GEO::vec3 p) {
    return plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w;
}

double distance_to_line_squared(GEO::vec3 p, GEO::vec3 a, GEO::vec3 b) {
    GEO::vec3 ab = b - a;
    GEO::vec3 ap = p - a;
    // Project ap onto ab to find the projected point p'
    GEO::vec3 p_prime = a + ab * (GEO::dot(ap, ab) / GEO::dot(ab, ab));
    return GEO::distance2(p, p_prime);
}

// assumes the plane normal is unit length
double distance_to_plane_squared(GEO::vec3 p, GEO::vec4 plane) {
    assert(std::abs(to_vec3(plane).length() - 1.0) < 1e-8);
    double d = eval_plane(plane, p);
    return d * d;
}

GEO::vec3 project_plane(GEO::vec3 p, const FaceData &face) {
    GEO::vec3 normal = to_vec3(face.face_plane);
    GEO::vec3 pt_on_plane = face.pt_on_plane;
    return p - GEO::dot(normal, p - pt_on_plane) * normal;
}

GEO::vec3 project_line(GEO::vec3 p, GEO::vec3 a, GEO::vec3 b) {
    GEO::vec3 ab = b - a;
    GEO::vec3 ap = p - a;
    return a + ab * (GEO::dot(ap, ab) / GEO::dot(ab, ab));
}

template<class F>
inline GEO::vec3 intersect(GEO::vec3 A, GEO::vec3 B, GEO::vec3 p, F dist_to_element_squared) {
    const double tol = 1e-5;
    double l(0), r(1), m;
    GEO::vec3 cur;
    int T = (int) log2(GEO::length(A - B) / tol);
    if (T <= 0) T = 1;
    while (T--) {
        m = (l + r) / 2;
        cur = (1 - m) * B + m * A;
        if (GEO::distance2(cur, p) > dist_to_element_squared(cur)) r = m;
        else l = m;
    }
    return (1 - l) * B + l * A;
}

// squared distance between a point and the straight line of an edge
inline double dis2_p2e(const GEO::vec3 &p, const EdgeData &e, const std::vector<GEO::vec3> &points) {
    GEO::vec3 dir = GEO::normalize(points[e.end] - points[e.start]);
    return GEO::cross(points[e.start] - p, dir).length2();
}

// squared distance between a point and the plane of a face
inline double dis2_p2f(const GEO::vec3 &p, const FaceData &f) {
    //double d = f.n.dot(points[f.verts.x()] - p);
    double d = GEO::dot(to_vec3(f.face_plane), f.pt_on_plane - p);
    return d * d;
}

// returns true if the vertex corresponding to site_point is intercepting the element,
// otherwise returns false. If an interception is detected, the bounding box of the
// element region clipped with the bisector of the element and the intercepted vertex
// is returned in box.
template<class F>
bool check_and_create_bounding_box(
        const GEO::ConvexCell &C,
        GEO::vec3 site_point,
        F dist_to_element_squared,
        BoundingBox &box) {

    bool is_intercepting = false;
    for (index_t v = 1; v < C.nb_v(); ++v) {
        index_t t = C.vertex_triangle(v);

        // Happens if a clipping plane did not clip anything.
        if (t == VBW::END_OF_LIST) {
            continue;
        }

        int last_region = 0;
        int first_region = 0;
        GEO::vec3 last_pt;
        GEO::vec3 first_pt;
        bool first_pt_set = false;

        do {
            GEO::vec3 pt = C.triangle_point(VBW::ushort(t));

            int region = dist_to_element_squared(pt) < GEO::distance2(pt, site_point) ? -1 : 1;

            if (!first_pt_set) {
                first_pt_set = true;
                first_pt = pt;
                first_region = region;
            }

            if (region == -1) {
                box.extend(pt);
                is_intercepting = true;
            }

            // note that we traverse every edge twice, once from each side, but we only need to compute
            // the intersection point once
            if (last_region == -1 && region == 1) {
                GEO::vec3 intersection = intersect(last_pt, pt, site_point, dist_to_element_squared);
                box.extend(intersection);
            }

            last_pt = pt;
            last_region = region;
            index_t lv = C.triangle_find_vertex(t, v);
            t = C.triangle_adjacent(t, (lv + 1) % 3);
        } while (t != C.vertex_triangle(v));

        // Process the edge connecting the last and first points
        if (last_region == -1 && first_region == 1) {
            GEO::vec3 intersection = intersect(last_pt, first_pt, site_point, dist_to_element_squared);
            box.extend(intersection);
        }
    }

    return is_intercepting;
}

// ============================= DISTANCE TO MESH ===============================

struct Impl {

    Impl(const std::vector<GEO::vec3> &points, const std::vector<std::array<uint32_t, 3>> &triangles,
         double limit_cube_len);

    // for each voronoi cell, check every face of the mesh if the vertex corresponding to the cell
    // "intercepts" the face. This means that after trimming the cell by the face's edge planes, it is
    // contained in the convex region that is closer
    void compute_interception_list();

    Result calc_closest_point(GEO::vec3 q);

    Bvh bvh;

    std::vector<GEO::vec3> points;
    std::vector<std::array<uint32_t, 3>> triangles;

    double limit_cube_len = 0;

    std::vector<EdgeData> edges;
    std::vector<FaceData> faces;

    std::vector<std::vector<PackedEdge>> intercepted_edges_packed;
    std::vector<std::vector<PackedFace>> intercepted_faces_packed;

#ifdef DEBUG_MANTIS
    std::map<std::pair<index_t, index_t>, GEO::ConvexCell> vertex_edge_cells;
    std::map<std::pair<index_t, index_t>, GEO::ConvexCell> vertex_face_cells;
    std::map<index_t, GEO::ConvexCell> vor_cells;
    std::map<index_t, GEO::ConvexCell> edge_cells;
    std::map<index_t, GEO::ConvexCell> face_cells;
#endif
};

Impl::Impl(const std::vector<GEO::vec3> &points, const std::vector<std::array<index_t, 3>> &triangles,
           double limit_cube_len)
        : points(points), triangles(triangles), bvh(points), limit_cube_len(limit_cube_len) {

    static int init_geogram = [] {
        GEO::initialize();
        return 0;
    }();
    (void) init_geogram;

    std::map<std::pair<index_t, index_t>, EdgeData> edge_map;
    for (index_t f = 0; f < triangles.size(); ++f) {
        auto t = triangles[f];
        for (int i = 0; i < 3; ++i) {
            index_t v0 = t[i];
            index_t v1 = t[(i + 1) % 3];
            bool is_left = true;
            if (v0 > v1) {
                std::swap(v0, v1);
                is_left = false;
            }
            EdgeData e{v0, v1};
            auto [it, inserted] = edge_map.emplace(std::pair{v0, v1}, e);
            if (inserted) {
                // populate end planes of edge
                GEO::vec3 start_pt = points[v0];
                GEO::vec3 end_pt = points[v1];

                GEO::vec3 n1 = GEO::normalize(end_pt - start_pt);
                GEO::vec3 n2 = GEO::normalize(start_pt - end_pt);
                auto &ed = it->second;
                ed.clipping_planes[ed.num_planes++] = to_vec4(n1, -GEO::dot(n1, start_pt));
                ed.clipping_planes[ed.num_planes++] = to_vec4(n2, -GEO::dot(n2, end_pt));
            }
            if (is_left) {
                assert(it->second.left_face == index_t(-1));
                it->second.left_face = f;
            } else {
                assert(it->second.right_face == index_t(-1));
                it->second.right_face = f;
            }
        }
    }

    faces.resize(triangles.size());
    for (index_t f = 0; f < faces.size(); ++f) {
        auto [v0, v1, v2] = triangles[f];
        GEO::vec3 p0 = points[v0];
        GEO::vec3 p1 = points[v1];
        GEO::vec3 p2 = points[v2];

        GEO::vec3 n = GEO::normalize(GEO::cross(p1 - p0, p2 - p0));

        GEO::vec3 n0 = GEO::normalize(GEO::cross(p2 - p1, n));
        GEO::vec3 n1 = GEO::normalize(GEO::cross(p0 - p2, n));
        GEO::vec3 n2 = GEO::normalize(GEO::cross(p1 - p0, n));

        GEO::vec4 plane0 = to_vec4(-n0, GEO::dot(n0, p1));
        GEO::vec4 plane1 = to_vec4(-n1, GEO::dot(n1, p2));
        GEO::vec4 plane2 = to_vec4(-n2, GEO::dot(n2, p0));

        faces[f].face_plane = to_vec4(n, -GEO::dot(n, p0));
        faces[f].clipping_planes[0] = plane0;
        faces[f].clipping_planes[1] = plane1;
        faces[f].clipping_planes[2] = plane2;

        faces[f].pt_on_plane = p0;

        auto &ed0 = edge_map[std::minmax(v0, v1)];
        ed0.clipping_planes[ed0.num_planes++] = -plane2;

        auto &ed1 = edge_map[std::minmax(v1, v2)];
        ed1.clipping_planes[ed1.num_planes++] = -plane0;

        auto &ed2 = edge_map[std::minmax(v2, v0)];
        ed2.clipping_planes[ed2.num_planes++] = -plane1;

    }

    edges.reserve(edge_map.size());
    for (const auto &[key, edge]: edge_map) {
        assert(edge.left_face != index_t(-1));
        assert(edge.right_face != index_t(-1));
        assert(edge.num_planes == 4);
        edges.push_back(edge);
    }

    compute_interception_list();
}


// for each voronoi cell, check every face of the mesh if the vertex corresponding to the cell
// "intercepts" the face. This means that after trimming the cell by the face's edge planes, it is
// contained in the convex region that is closer
void Impl::compute_interception_list() {
    const index_t nb_points = points.size();
    const index_t nb_faces = triangles.size();
    const index_t nb_edges = edges.size();

    double l = limit_cube_len * 2;
    auto copy = points;
    copy.emplace_back(l, l, l);
    copy.emplace_back(-l, l, l);
    copy.emplace_back(l, -l, l);
    copy.emplace_back(l, l, -l);
    copy.emplace_back(-l, -l, l);
    copy.emplace_back(-l, l, -l);
    copy.emplace_back(l, -l, -l);
    copy.emplace_back(-l, -l, -l);

    GEO::SmartPointer<GEO::PeriodicDelaunay3d> delaunay = new GEO::PeriodicDelaunay3d(false, 1.0);
    delaunay->set_keeps_infinite(true);
    delaunay->set_vertices(copy.size(), (double *) copy.data());
    delaunay->set_stores_neighbors(true);
    delaunay->compute();

    GEO::PeriodicDelaunay3d::IncidentTetrahedra W;

#ifdef DEBUG_MANTIS
    for (index_t v = 0; v < nb_points; ++v) {
            GEO::vec3 site_p = points[v];
            delaunay->copy_Laguerre_cell_from_Delaunay(v, C, W);
            C.compute_geometry();
            vor_cells[v] = C;
        }

        for (index_t e = 0; e < nb_edges; ++e) {
            GEO::ConvexCell clipped_cell;
            clipped_cell.init_with_box(-l, -l, -l, l, l, l);
            for (const GEO::vec4 &clipping_plane: edges[e].clipping_planes) {
                clipped_cell.clip_by_plane(clipping_plane);
            }
            clipped_cell.compute_geometry();
            edge_cells[e] = clipped_cell;
        }

        for (index_t f = 0; f < nb_faces; ++f) {
            GEO::ConvexCell clipped_cell;
            clipped_cell.init_with_box(-l, -l, -l, l, l, l);
            for (const GEO::vec4 &clipping_plane: faces[f].clipping_planes) {
                clipped_cell.clip_by_plane(clipping_plane);
            }
            clipped_cell.compute_geometry();
            face_cells[f] = clipped_cell;
        }
#endif

    std::vector<GEO::ConvexCell> voronoi_cells(nb_points);

    for (index_t v = 0; v < nb_points; ++v) {
        delaunay->copy_Laguerre_cell_from_Delaunay(v, voronoi_cells[v], W);
        voronoi_cells[v].compute_geometry();
    }

    std::vector<std::vector<index_t>> face_vertex(nb_faces);
    std::vector<std::vector<BoundingBox>> face_vertex_bb(nb_faces);

    auto handle_face = [this, &voronoi_cells, delaunay, nb_points, &face_vertex, &face_vertex_bb](index_t f) {
        auto dist_squared = [plane = faces[f].face_plane](GEO::vec3 p) {
            return distance_to_plane_squared(p, plane);
        };

        std::unordered_set<index_t> visited = {triangles[f][0], triangles[f][1], triangles[f][2]};
        std::queue<index_t> queue;
        queue.push(triangles[f][0]);
        queue.push(triangles[f][1]);
        queue.push(triangles[f][2]);

        while (!queue.empty()) {
            index_t v = queue.front();
            queue.pop();

            //delaunay->copy_Laguerre_cell_from_Delaunay(v, C, W);
            GEO::ConvexCell C = voronoi_cells[v];
            for (const GEO::vec4 &clipping_plane: faces[f].clipping_planes) {
                C.clip_by_plane(clipping_plane);
            }
            if (C.empty()) {
                continue;
            }
            C.compute_geometry();
            BoundingBox box;
            if (check_and_create_bounding_box(C, points[v], dist_squared, box)) {
                face_vertex[f].push_back(v);
                face_vertex_bb[f].push_back(box);
#ifdef DEBUG_MANTIS
                vertex_face_cells[{v, f}] = C;
#endif
            } else {
                continue;
            }

            GEO::vector<index_t> neighbors;
            delaunay->get_neighbors(v, neighbors);
            for (index_t n: neighbors) {
                if (n < nb_points && !visited.count(n)) {
                    visited.insert(n);
                    queue.push(n);
                }
            }
        }
    };

    parallel_for(0, nb_faces, handle_face);

    std::vector<std::vector<index_t>> edge_vertex(nb_edges);
    std::vector<std::vector<BoundingBox>> edge_vertex_bb(nb_edges);

    auto handle_edge = [this, &voronoi_cells, delaunay, nb_points, &edge_vertex, &edge_vertex_bb](index_t e) {
        GEO::vec3 l0 = points[edges[e].start];
        GEO::vec3 l1 = points[edges[e].end];
        auto dist_squared = [l0, l1](GEO::vec3 p) {
            return distance_to_line_squared(p, l0, l1);
        };

        std::unordered_set<index_t> visited = {edges[e].start, edges[e].end};
        std::queue<index_t> queue;
        queue.push(edges[e].start);
        queue.push(edges[e].end);

        while (!queue.empty()) {
            index_t v = queue.front();
            queue.pop();

            //delaunay->copy_Laguerre_cell_from_Delaunay(v, C, W);
            GEO::ConvexCell C = voronoi_cells[v];
            for (const GEO::vec4 &clipping_plane: edges[e].clipping_planes) {
                C.clip_by_plane(clipping_plane);
            }
            if (C.empty()) {
                continue;
            }
            C.compute_geometry();
            BoundingBox box;
            if (check_and_create_bounding_box(C, points[v], dist_squared, box)) {
                edge_vertex[e].push_back(v);
                edge_vertex_bb[e].push_back(box);
#ifdef DEBUG_MANTIS
                vertex_edge_cells[{v, e}] = C;
#endif
            } else {
                continue;
            }

            GEO::vector<index_t> neighbors;
            delaunay->get_neighbors(v, neighbors);
            for (index_t n: neighbors) {
                if (n < nb_points && !visited.count(n)) {
                    visited.insert(n);
                    queue.push(n);
                }
            }
        }
    };

    parallel_for(0, nb_edges, handle_edge);

    std::vector<std::vector<index_t>> intercepted_edges(nb_points);
    std::vector<std::vector<BoundingBox>> intercepted_edges_bb(nb_points);
    std::vector<std::vector<index_t>> intercepted_faces(nb_points);
    std::vector<std::vector<BoundingBox>> intercepted_faces_bb(nb_points);

    // transpose edge_vertex and face_vertex arrays
    for (index_t e = 0; e < nb_edges; ++e) {
        for (size_t i = 0; i < edge_vertex[e].size(); ++i) {
            index_t v = edge_vertex[e][i];
            intercepted_edges[v].push_back(e);
            intercepted_edges_bb[v].push_back(edge_vertex_bb[e][i]);
        }
    }
    for (index_t f = 0; f < nb_faces; ++f) {
        for (size_t i = 0; i < face_vertex[f].size(); ++i) {
            index_t v = face_vertex[f][i];
            intercepted_faces[v].push_back(f);
            intercepted_faces_bb[v].push_back(face_vertex_bb[f][i]);
        }
    }

    intercepted_edges_packed.resize(nb_points);
    intercepted_faces_packed.resize(nb_points);

    // iterate over vertices and sort interception lists
    std::vector<int> order;
    for (index_t v = 0; v < nb_points; ++v) {
        // first reorder edges
        order.resize(intercepted_edges[v].size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
            return intercepted_edges_bb[v][a].lower.x < intercepted_edges_bb[v][b].lower.x;
        });

        std::vector<index_t> new_intercepted_edges(intercepted_edges[v].size());
        std::vector<BoundingBox> new_intercepted_edges_bb(intercepted_edges_bb[v].size());
        for (index_t i = 0; i < order.size(); ++i) {
            new_intercepted_edges[i] = intercepted_edges[v][order[i]];
            new_intercepted_edges_bb[i] = intercepted_edges_bb[v][order[i]];
        }

        intercepted_edges[v] = std::move(new_intercepted_edges);
        intercepted_edges_bb[v] = std::move(new_intercepted_edges_bb);

        // round up number of edge batches
        size_t num_edge_packed = (intercepted_edges[v].size() + 3) / 4;
        intercepted_edges_packed[v].resize(num_edge_packed);

        for (size_t i = 0; i < num_edge_packed; ++i) {
            PackedEdge packed{};
            for (size_t j = 0; j < 4; ++j) {
                if (i * 4 + j < intercepted_edges[v].size()) {
                    index_t e = intercepted_edges[v][i * 4 + j];
                    packed.min_x[j] = (float) intercepted_edges_bb[v][i * 4 + j].lower.x;
                    for (size_t d = 0; d < 3; ++d) {
                        packed.start[d][j] = (float) points[edges[e].start][d];
                        packed.dir[d][j] = float(points[edges[e].end][d] - points[edges[e].start][d]);
                    }
                    packed.dir_len_squared[j] = float(GEO::distance2(points[edges[e].end], points[edges[e].start]));
                    packed.primitive_idx[j] = int(e + nb_points);
                } else {
                    // duplicate last edge
                    assert(j > 0);
                    packed.min_x[j] = packed.min_x[j - 1];
                    for (size_t d = 0; d < 3; ++d) {
                        packed.start[d][j] = packed.start[d][j - 1];
                        packed.dir[d][j] = packed.dir[d][j - 1];
                    }
                    packed.dir_len_squared[j] = packed.dir_len_squared[j - 1];
                    packed.primitive_idx[j] = packed.primitive_idx[j - 1];
                }
            }
            intercepted_edges_packed[v][i] = packed;
        }

        // then reorder faces
        order.resize(intercepted_faces[v].size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
            return intercepted_faces_bb[v][a].lower.x < intercepted_faces_bb[v][b].lower.x;
        });
        std::vector<index_t> new_intercepted_faces(intercepted_faces[v].size());
        std::vector<BoundingBox> new_intercepted_faces_bb(intercepted_faces_bb[v].size());
        for (index_t i = 0; i < order.size(); ++i) {
            new_intercepted_faces[i] = intercepted_faces[v][order[i]];
            new_intercepted_faces_bb[i] = intercepted_faces_bb[v][order[i]];
        }

        intercepted_faces[v] = std::move(new_intercepted_faces);
        intercepted_faces_bb[v] = std::move(new_intercepted_faces_bb);

        // round up nb of face batches
        size_t num_face_packed = (intercepted_faces[v].size() + 3) / 4;
        intercepted_faces_packed[v].resize(num_face_packed);
        intercepted_faces_bb[v].resize(num_face_packed);

        for (size_t i = 0; i < num_face_packed; ++i) {
            PackedFace packed{};
            for (size_t j = 0; j < 4; ++j) {
                if (i * 4 + j < intercepted_faces[v].size()) {
                    index_t f = intercepted_faces[v][i * 4 + j];
                    packed.min_x[j] = (float) intercepted_faces_bb[v][i * 4 + j].lower.x;
                    for (size_t d = 0; d < 4; ++d) {
                        packed.face_plane[d][j] = (float) faces[f].face_plane[d];
                        packed.edge_plane0[d][j] = (float) faces[f].clipping_planes[0][d];
                        packed.edge_plane1[d][j] = (float) faces[f].clipping_planes[1][d];
                        packed.edge_plane2[d][j] = (float) faces[f].clipping_planes[2][d];
                    }
                    packed.primitive_idx[j] = int(f + nb_points + nb_edges);
                } else {
                    // duplicate last face
                    assert(j > 0);
                    packed.min_x[j] = packed.min_x[j - 1];
                    for (size_t d = 0; d < 4; ++d) {
                        packed.face_plane[d][j] = packed.face_plane[d][j - 1];
                        packed.edge_plane0[d][j] = packed.edge_plane0[d][j - 1];
                        packed.edge_plane1[d][j] = packed.edge_plane1[d][j - 1];
                        packed.edge_plane2[d][j] = packed.edge_plane2[d][j - 1];
                    }
                }
            }
            intercepted_faces_packed[v][i] = packed;
        }
    }
}

Result Impl::calc_closest_point(GEO::vec3 q) {
    int v = bvh.closestPoint(q);

    float32x4_t qx = dup_float((float) q.x);
    float32x4_t qy = dup_float((float) q.y);
    float32x4_t qz = dup_float((float) q.z);

    float32x4_t best_d2 = dup_float(float(GEO::distance2(q, points[v])));
    int32x4_t best_idx = dup_int(v);

    const auto &v_edges = intercepted_edges_packed[v];
    for (const PackedEdge &pack: v_edges) {
        if (q.x < pack.min_x[0]) {
            break;
        }

        float32x4_t apx = qx - pack.start[0];
        float32x4_t apy = qy - pack.start[1];
        float32x4_t apz = qz - pack.start[2];

        float32x4_t t = dot(apx, apy, apz, pack.dir[0], pack.dir[1], pack.dir[2]) / pack.dir_len_squared;

        // the result is only valid if t is in [0, 1]
        uint32x4_t mask = logical_and(leq(dup_float(0.0f), t), leq(t, dup_float(1.0f)));

        // project onto segment
        float32x4_t projectedx = fma(t, pack.dir[0], pack.start[0]);
        float32x4_t projectedy = fma(t, pack.dir[1], pack.start[1]);
        float32x4_t projectedz = fma(t, pack.dir[2], pack.start[2]);

        float32x4_t d2_line = distance_squared(qx, qy, qz, projectedx, projectedy, projectedz);

        mask = logical_and(mask, leq(d2_line, best_d2));
        best_d2 = select(mask, d2_line, best_d2);
        best_idx = select(mask, pack.primitive_idx, best_idx);
    }

    const auto &v_faces = intercepted_faces_packed[v];
    for (const PackedFace &pack: v_faces) {
        if (q.x < pack.min_x[0]) {
            break;
        }

        // point is inside face region if it is on the positive side of all three planes
        float32x4_t s0 = eval_plane(qx, qy, qz, pack.edge_plane0[0], pack.edge_plane0[1], pack.edge_plane0[2],
                                    pack.edge_plane0[3]);
        float32x4_t s1 = eval_plane(qx, qy, qz, pack.edge_plane1[0], pack.edge_plane1[1], pack.edge_plane1[2],
                                    pack.edge_plane1[3]);
        float32x4_t s2 = eval_plane(qx, qy, qz, pack.edge_plane2[0], pack.edge_plane2[1], pack.edge_plane2[2],
                                    pack.edge_plane2[3]);

        uint32x4_t mask = logical_and(logical_and(leq(dup_float(0.0f), s0), leq(dup_float(0.0f), s1)),
                                      leq(dup_float(0.0f), s2));

        float32x4_t d2 = eval_plane(qx, qy, qz, pack.face_plane[0], pack.face_plane[1], pack.face_plane[2],
                                    pack.face_plane[3]);
        d2 = d2 * d2;

        mask = logical_and(mask, leq(d2, best_d2));
        best_d2 = select(mask, d2, best_d2);
        best_idx = select(mask, pack.primitive_idx, best_idx);
    }

    Result result{best_d2[0], best_idx[0]};

    // Find overall minimum distance and index
    for (int j = 1; j < 4; ++j) {
        if (best_d2[j] < result.distance_squared) {
            result.distance_squared = best_d2[j];
            result.primitive_index = best_idx[j];
        }
    }

    GEO::vec3 cp;
    if (result.primitive_index < points.size()) {
        cp = points[result.primitive_index];
        result.type = PrimitiveType::Vertex;
    } else if (result.primitive_index < points.size() + edges.size()) {
        int offset = (int) points.size();
        const auto &e = edges[result.primitive_index - offset];
        cp = project_line(q, points[e.start], points[e.end]);
        result.primitive_index -= offset;
        result.type = PrimitiveType::Edge;
    } else {
        int offset = (int) points.size() + (int) edges.size();
        auto f = faces[result.primitive_index - offset];
        cp = project_plane(q, f);
        result.primitive_index -= offset;
        result.type = PrimitiveType::Face;
    }
    result.closest_point[0] = (float) cp.x;
    result.closest_point[1] = (float) cp.y;
    result.closest_point[2] = (float) cp.z;

    return result;
}

AccelerationStructure::AccelerationStructure(const float *points, size_t num_points, const uint32_t *indices,
                                             size_t num_faces,
                                             float limit_cube_len) {
    std::vector<GEO::vec3> points_vec(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        points_vec[i] = {points[3 * i], points[3 * i + 1], points[3 * i + 2]};
    }
    std::vector<std::array<uint32_t, 3>> faces_vec(num_faces);
    for (size_t i = 0; i < num_faces; ++i) {
        faces_vec[i] = {indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]};
    }
    impl = new Impl(points_vec, faces_vec, limit_cube_len);
}

AccelerationStructure::AccelerationStructure(const std::vector<std::array<float, 3>> &points,
                                             const std::vector<std::array<uint32_t, 3>> &triangles,
                                             float limit_cube_len) :
        AccelerationStructure((const float *) points.data(), points.size(), (const uint32_t *) triangles.data(),
                              triangles.size(), limit_cube_len) {}

AccelerationStructure::AccelerationStructure(AccelerationStructure &&other) noexcept {
    impl = other.impl;
    other.impl = nullptr;
}

AccelerationStructure &AccelerationStructure::operator=(AccelerationStructure &&other) noexcept {
    if (this != &other) {
        delete impl;
        impl = other.impl;
        other.impl = nullptr;
    }
    return *this;
}

Result AccelerationStructure::calc_closest_point(float x, float y, float z) const {
    return impl->calc_closest_point({x, y, z});
}

Result AccelerationStructure::calc_closest_point(std::array<float, 3> q) const {
    return impl->calc_closest_point({q[0], q[1], q[2]});
}

std::vector<std::array<uint32_t, 3>> AccelerationStructure::get_face_edges() const {
    std::vector<std::array<uint32_t, 3>> result(num_faces());
    std::vector<int> current_index(num_faces(), 0);
    for (size_t i = 0; i < num_edges(); ++i) {
        const auto &e = impl->edges[i];
        assert(e.left_face != -1);
        assert(e.right_face != -1);
        assert(current_index[e.left_face] < 3);
        assert(current_index[e.right_face] < 3);
        result[e.left_face][current_index[e.left_face]++] = i;
        result[e.right_face][current_index[e.right_face]++] = i;
    }
    return result;
}


std::vector<std::pair<uint32_t, uint32_t>> AccelerationStructure::get_edge_vertices() const {
    std::vector<std::pair<uint32_t, uint32_t>> result(num_edges());
    for (size_t i = 0; i < num_edges(); ++i) {
        const auto &e = impl->edges[i];
        result[i] = {e.start, e.end};
    }
    return result;
}

std::vector<std::array<uint32_t, 3>> AccelerationStructure::get_faces() const {
    return impl->triangles;
}

std::vector<std::array<float, 3>> AccelerationStructure::get_positions() const {
    std::vector<std::array<float, 3>> result(num_vertices());
    for (size_t i = 0; i < num_vertices(); ++i) {
        const auto &p = impl->points[i];
        result[i] = {float(p.x), float(p.y), float(p.z)};
    }
    return result;
}

std::pair<uint32_t, uint32_t> AccelerationStructure::get_edge(size_t index) const {
    const auto &e = impl->edges[index];
    return {e.start, e.end};
}

size_t AccelerationStructure::num_edges() const {
    return impl->edges.size();
}

size_t AccelerationStructure::num_faces() const {
    return impl->triangles.size();
}

size_t AccelerationStructure::num_vertices() const {
    return impl->points.size();
}

AccelerationStructure::~AccelerationStructure() {
    delete impl;
}

}