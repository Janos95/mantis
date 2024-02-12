#include "implicit_meshing.h"
#include "parallel.h"

#include <memory>

#include "probabilistic-quadrics.hh"
#include "hash_table7.hpp"

#define ENABLE_TIMING 0
using UNIT = std::chrono::milliseconds;
constexpr char UNIT_NAME[] = "ms";

#if ENABLE_TIMING
#define PROF(stage) printf("%s took: %d %s sec\n", stage, (int) std::chrono::duration_cast<UNIT>(end_t - start_t).count(), UNIT_NAME);
#else
#define PROF(stage) (void) start_t; (void) end_t;
#endif

using Math = pq::math<double, Vec3, Vec3, Mat3>;
using quadric = pq::quadric<Math>;

// Edge between two adjacent grid points a and b
// idx shows the axis of the edge (0 -> x, 1 -> y, 2 -> z)
struct Edge {
    Vec3i a;
    Vec3i b;
    int idx = -1;
};

struct GridHash {
    size_t operator()(const Vec3i &v) const {
        return (v.x * 73856093) ^ (v.y * 19349669) ^ (v.z * 83492791);
    }
};

// GridCell represents a range of grid points in 3D space.
// The first vector defines the inclusive lower-left front grid index,
// while the second vector defines the exclusive upper-right back grid index.
using GridCell = std::pair<Vec3i, Vec3i>;

// Split current grid cell intro 8 smaller cells
void generate_children(std::vector<GridCell> &grid_cells, const GridCell &current) {
    Vec3i min_coord = current.first;
    Vec3i max_coord = current.second;

    // Ensure the cell dimensions are valid
    if (max_coord.x <= min_coord.x || max_coord.y <= min_coord.y || max_coord.z <= min_coord.z) {
        throw std::invalid_argument("Invalid cell dimensions");
    }

    // Calculate the size of each child cell
    Vec3i cell_size = (max_coord - min_coord) / 2;

    // Ensure each child cell will have a positive size
    assert(cell_size.x > 0 && cell_size.y > 0 && cell_size.z > 0);

    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            for (int z = 0; z < 2; ++z) {
                Vec3i child_min = min_coord + cell_size * Vec3i(x, y, z);
                Vec3i child_max = child_min + cell_size;

                // Adjust the max boundary for the last cell in each dimension
                if (x == 1) child_max.x = max_coord.x;
                if (y == 1) child_max.y = max_coord.y;
                if (z == 1) child_max.z = max_coord.z;

                GridCell child_cell(child_min, child_max);
                grid_cells.push_back(child_cell);
            }
        }
    }
}

Vec3 interpolate(std::pair<Vec3, double> neg, std::pair<Vec3, double> pos) {
    double val_neg = neg.second;
    double val_pos = pos.second;
    assert(val_neg <= 0);
    assert(val_pos >= 0);
    auto pt_neg = neg.first;
    auto pt_pos = pos.first;
    double t = val_neg / (val_neg - val_pos);
    Vec3 p = pt_neg + (pt_pos - pt_neg) * t;
    return p;
}

Vec3 find_point_on_surface(std::pair<Vec3, double> neg, std::pair<Vec3, double> pos,
                           const std::function<double(Vec3)> &f,
                           int num_iterations) {
    assert(num_iterations > 0);
    const double threshold = 10e-5;
    for (int i = 0; i < num_iterations - 1; ++i) {
        Vec3 p = interpolate(neg, pos);
        double val = f(p);
        if (std::abs(val) < threshold) {
            return p;
        }
        if (val < 0) {
            neg = {p, val};
        } else {
            pos = {p, val};
        }
    }

    // Return the best approximation if the loop completes without finding the exact point
    return interpolate(neg, pos);
}

struct Item {
    double value;
    size_t index;
};


struct SharedData {
    ThreadSpecific<std::vector<Vec3i>> &grids;
    std::function<double(Vec3)> f;
    Vec3 lower, upper;
    int n;
};

struct Task;

size_t num_voxels(const GridCell &cell) {
    Vec3i grid_size = cell.second - cell.first;
    return size_t(grid_size.x) * size_t(grid_size.y) * size_t(grid_size.z);
}

struct Task {
    GridCell cell{};
    SharedData *data = nullptr;

    void operator()() const {
        Vec3i grid_size = cell.second - cell.first;
        // if the cell is too small to divide it again, just evaluate the function at the lower corner
        if (num_voxels(cell) <= 16 || grid_size.x == 1 || grid_size.y == 1 || grid_size.z == 1) {
            auto &grid = data->grids.local();
            for (int i = cell.first.x; i < cell.second.x; ++i) {
                for (int j = cell.first.y; j < cell.second.y; ++j) {
                    for (int k = cell.first.z; k < cell.second.z; ++k) {
                        grid.emplace_back(i, j, k);
                    }
                }
            }
            return;
        }
        // if the cell does not contain a zero-crossing, skip the cell
        Vec3 cell_lower = index_to_grid_point(cell.first);
        Vec3 cell_upper = index_to_grid_point(cell.second);
        double v = data->f((cell_upper + cell_lower) / 2.0);
        double dist_to_corner = length(cell_upper - cell_lower) / 2.0;
        constexpr double LIPSCHITZ = 2.0;
        if (abs(v) > LIPSCHITZ * dist_to_corner) {
            return;
        }

        // if the cell contains a zero-crossing, subdivide it into 8 smaller cells
        generate_children();
    }

    void generate_children() const {
        auto current = cell;
        Vec3i min_coord = current.first;
        Vec3i max_coord = current.second;

        assert(max_coord.x > min_coord.x && max_coord.y > min_coord.y && max_coord.z > min_coord.z);

        Vec3i cell_size = (max_coord - min_coord) / 2;

        assert(cell_size.x > 0 && cell_size.y > 0 && cell_size.z > 0);

        TaskGroup tg;
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    Vec3i child_min = min_coord + cell_size * Vec3i(x, y, z);
                    Vec3i child_max = child_min + cell_size;

                    // Adjust the max boundary for the last cell in each dimension
                    if (x == 1) child_max.x = max_coord.x;
                    if (y == 1) child_max.y = max_coord.y;
                    if (z == 1) child_max.z = max_coord.z;

                    GridCell child_cell(child_min, child_max);
                    tg.run(Task{child_cell, data});
                }
            }
        }
        tg.wait();
    }

    Vec3 index_to_grid_point(Vec3i index_) const {
        Vec3 index(double(index_.x), double(index_.y), double(index_.z));
        return data->lower + index / (double(data->n) - 1.0f) * (data->upper - data->lower);
    };
};

QuadMesh generate_mesh(std::function<double(Vec3)> f, int n) {
    Vec3 lower{-3.f};
    Vec3 upper{3.f};

    auto index_to_grid_point = [=](Vec3i index_) {
        Vec3 index(double(index_.x), double(index_.y), double(index_.z));
        return lower + index / (double(n) - 1.0f) * (upper - lower);
    };

    auto gradient_fwd = [=](Vec3 p) -> Vec3 {
        double eps = 10e-5;
        double v = f(p);
        double dx = (f({p.x + eps, p.y, p.z}) - v) / eps;
        double dy = (f({p.x, p.y + eps, p.z}) - v) / eps;
        double dz = (f({p.x, p.y, p.z + eps}) - v) / eps;
        return {dx, dy, dz};
    };

    auto start_t = std::chrono::high_resolution_clock::now();
    ThreadSpecific<std::vector<Vec3i>> grids;

    // subdivide cells that contain zero-crossings
    SharedData data{grids, f, lower, upper, n};
    Task root_task{{{0, 0, 0}, {n, n, n}}, &data};
    root_task();

    auto end_t = std::chrono::high_resolution_clock::now();

    PROF("Subdivision")

    using GridMap = emhash7::HashMap<Vec3i, Item, GridHash>;
    GridMap grid;
    std::vector<Vec3i> grid_index;
    std::vector<double> grid_values;

    auto initialize_grid = [&]() {
        size_t counter = 0;
        for (auto &local_grid: grids) {
            counter += local_grid.size();
        }

        grid_index.reserve(counter);
        for (const auto &local_grid: grids) {
            grid_index.insert(grid_index.end(), local_grid.begin(), local_grid.end());
        }

        grid_values.resize(grid_index.size());
        parallel_for(size_t(0), grid_index.size(), [&](size_t i) {
            Vec3 p = index_to_grid_point(grid_index[i]);
            grid_values[i] = f(p);
        });

        grid.reserve(counter);
        for (size_t i = 0; i < grid_index.size(); ++i) {
            grid.insert_unique(grid_index[i], Item{grid_values[i], i});
        }
        assert(counter == grid.size());
    };

    start_t = std::chrono::high_resolution_clock::now();
    initialize_grid();
    end_t = std::chrono::high_resolution_clock::now();

    PROF("Init global grid")

    start_t = std::chrono::high_resolution_clock::now();

    std::vector<int> edge_has_zero_crossing(grid.size() * 3, 0);
    std::vector<quadric> edge_quadrics(grid.size() * 3);

    parallel_for(size_t(0), grid_index.size(), [&](size_t i) {
        auto index1 = grid_index[i];
        auto v1 = grid_values[i];
        Vec3 p1 = index_to_grid_point(index1);
        for (size_t j = 0; j < 3; ++j) {
            Vec3i index2 = index1 + Vec3i(j == 0, j == 1, j == 2);
            auto it = grid.find(index2);
            if (it == grid.end()) {
                continue;
            }
            double v2 = it->second.value;
            if (v1 * v2 <= 0) {
                Vec3 p2 = index_to_grid_point(index2);
                std::pair p1v1 = {p1, v1};
                std::pair p2v2 = {p2, v2};

                if (v1 > 0) {
                    std::swap(p1v1, p2v2);
                }

                auto zero_crossing = find_point_on_surface(p1v1, p2v2, f, 5);
                auto eq = quadric::probabilistic_plane_quadric(zero_crossing,
                                                               normalize(gradient_fwd(zero_crossing)),
                                                               0.05f, 0.05f);

                edge_quadrics[i * 3 + j] = eq;
                edge_has_zero_crossing[i * 3 + j] = true;
            }
        }
    });

    end_t = std::chrono::high_resolution_clock::now();
    PROF("Computing edge quadrics")

    // generate vertex positions of the output mesh
    // for each voxel we compute a point if at least one of its edges contains a zero-crossing
    // the point is computed by minimizing a quadric error metric
    start_t = std::chrono::high_resolution_clock::now();

    std::vector<Vec3> voxel_point(grid.size());
    std::vector<int> voxel_has_point(grid.size(), 0);

    // all edges must point from smaller to larger
    const std::vector<std::pair<Vec3i, Vec3i>> all_edges = {{{0, 0, 0}, {1, 0, 0}},
                                                            {{0, 0, 0}, {0, 1, 0}},
                                                            {{0, 0, 0}, {0, 0, 1}},
                                                            {{1, 0, 0}, {1, 1, 0}},
                                                            {{1, 0, 0}, {1, 0, 1}},
                                                            {{0, 1, 0}, {1, 1, 0}},
                                                            {{0, 1, 0}, {0, 1, 1}},
                                                            {{0, 0, 1}, {1, 0, 1}},
                                                            {{0, 0, 1}, {0, 1, 1}},
                                                            {{1, 1, 0}, {1, 1, 1}},
                                                            {{1, 0, 1}, {1, 1, 1}},
                                                            {{0, 1, 1}, {1, 1, 1}}};


    parallel_for(size_t(0), grid_index.size(), [&](size_t i) {
        Vec3i index = grid_index[i];
        size_t counter = 0;
        quadric vq;
        for (auto e: all_edges) {
            Vec3i start = index + e.first;
            Vec3i end = index + e.second;
            auto it = grid.find(start);
            if (it == grid.end()) {
                continue;
            }
            auto diff = end - start;
            assert(diff[0] >= 0);
            assert(diff[1] >= 0);
            assert(diff[2] >= 0);
            size_t k = diff[0] == 1 ? 0 : diff[1] == 1 ? 1 : 2;
            size_t j = 3 * it->second.index + k;
            if (edge_has_zero_crossing[j]) {
                vq += edge_quadrics[j];
                ++counter;
            }
        }
        if (counter > 0) {
            voxel_point[i] = vq.minimizer();
            voxel_has_point[i] = true;
        }
    });

    end_t = std::chrono::high_resolution_clock::now();
    PROF("Computing positions")

    int edge_indices[3][3] = {
            {0, 2, 1},
            {1, 0, 2},
            {2, 1, 0}
    };

    start_t = std::chrono::high_resolution_clock::now();
    std::vector<Vec3> points;
    std::vector<int> vertex_map(voxel_point.size(), -1);
    {
        int i = 0;
        for (size_t j = 0; j < voxel_point.size(); ++j) {
            if (voxel_has_point[j]) {
                vertex_map[j] = i;
                points.push_back(voxel_point[j]);
                ++i;
            }
        }
    }
    end_t = std::chrono::high_resolution_clock::now();
    PROF("Generating vertex map")

    std::vector<std::array<int, 4>> quads;

    start_t = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < grid_index.size(); ++i) {
        Vec3i a0 = grid_index[i];
        if (a0.x == 0 || a0.y == 0 || a0.z == 0 || a0.x == n - 1 || a0.y == n - 1 || a0.z == n - 1) {
            continue;
        }
        double v0 = grid_values[i];
        // iterate over the three edges starting at grid_idx
        for (auto [j, idx1, idx2]: edge_indices) {
            // check if the edge has a zero crossing
            if (!edge_has_zero_crossing[i * 3 + j]) {
                continue;
            }

            assert(voxel_has_point[i]);

            double v1 = grid[a0 + Vec3i(j == 0, j == 1, j == 2)].value;

            // generate quad
            Vec3i a1 = a0;
            a1[idx1] -= 1;
            Vec3i a2 = a0;
            a2[idx2] -= 1;
            Vec3i a3 = a0;
            a3[idx1] -= 1;
            a3[idx2] -= 1;

            auto it1 = grid.find(a1);
            auto it2 = grid.find(a2);
            auto it3 = grid.find(a3);

            // TODO: this is a bit of a hack. The problem is that we might prune a grid point that is part of
            // a voxel for which we do generate a vertex. In that case we don't have an entry in the grid hashmap.
            if(it1 == grid.end() || it2 == grid.end() || it3 == grid.end()) {
                continue;
            }

            std::array<int, 4> quad{vertex_map[i], vertex_map[it1->second.index], vertex_map[it3->second.index],
                                    vertex_map[it2->second.index]};
            // there are 7 relevant cases:
            // v0 == 0 && v1 == 0 -> ambiguous, lets orient toward v0 for now
            // v0 == 0 && v1 < 0 -> orient toward v0
            // v1 == 0 && v0 > 0 -> orient toward v0
            // v0 > 0 && v1 < 0 -> orient toward v0
            // v0 == 0 && v1 > 0 -> orient toward v1
            // v1 == 0 && v0 < 0 -> orient toward v1
            // v1 > 0 && v0 < 0 -> orient toward v1
            // by default the face is oriented towards v0
            if (v0 == 0 && v1 > 0 || v1 == 0 && v0 < 0 || v1 > 0 && v0 < 0) {
                std::reverse(quad.begin(), quad.end());
            }
            assert(quad[0] != -1);
            assert(quad[1] != -1);
            assert(quad[2] != -1);
            assert(quad[3] != -1);
            quads.push_back(quad);
        }
    }
    end_t = std::chrono::high_resolution_clock::now();
    PROF("Generating quads")

    return {std::move(points), std::move(quads)};
}
