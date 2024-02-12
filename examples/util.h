#pragma once

#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/curve_network.h>

#include <utility>

#include "Delaunay_psm.h"

using index_t = GEO::index_t;

namespace ps = polyscope;

inline void draw_point(std::string name, GEO::vec3 a) {
    std::vector<GEO::vec3> vertices = {a};
    ps::registerPointCloud(std::move(name), vertices);
}

inline void draw_edge(std::string name, GEO::vec3 a, GEO::vec3 b) {
    std::vector<GEO::vec3> vertices = {a, b};
    ps::registerCurveNetworkLine(std::move(name), vertices);
}

inline void draw_oriented_edge(const std::string& name, GEO::vec3 a, GEO::vec3 b) {
    std::vector<GEO::vec3> vertices = {a, b};
    ps::registerCurveNetworkLine(name, vertices);
    ps::registerPointCloud(name + std::string("tip"), std::vector{b});
}

inline void draw_triangle_face(std::string name, GEO::vec3 a, GEO::vec3 b, GEO::vec3 c) {
    std::vector<GEO::vec3> vertices = {a, b, c};
    std::vector<std::array<index_t, 3>> indices = {{0, 1, 2}};
    ps::registerSurfaceMesh(std::move(name), vertices, indices);
}

constexpr float cells_shrink = 0.0001f;

inline void draw_cell(GEO::ConvexCell &C, std::string name) {
    auto s = double(cells_shrink);

    GEO::vec3 g;
    if (cells_shrink != 0.0f) {
        g = C.barycenter();
    }

    std::vector<GEO::vec3> vertices;
    std::vector<std::array<index_t, 3>> triangles;

    // For each vertex of the Voronoi cell
    // Start at 1 (vertex 0 is point at infinity)
    for (index_t v = 1; v < C.nb_v(); ++v) {
        index_t t = C.vertex_triangle(v);

        // Happens if a clipping plane did not clip anything.
        if (t == VBW::END_OF_LIST) {
            continue;
        }

        // Now iterate on the Voronoi vertices of the
        // Voronoi facet. In dual form this means iterating
        // on the triangles incident to vertex v

        GEO::vec3 P[3];
        index_t n = 0;
        do {
            // Triangulate the Voronoi facet
            if (n == 0) {
                P[0] = C.triangle_point(VBW::ushort(t));
            } else if (n == 1) {
                P[1] = C.triangle_point(VBW::ushort(t));
            } else {
                P[2] = C.triangle_point(VBW::ushort(t));
                vertices.push_back(s * g + (1.0 - s) * P[0]);
                vertices.push_back(s * g + (1.0 - s) * P[1]);
                vertices.push_back(s * g + (1.0 - s) * P[2]);
                triangles.push_back({{index_t(vertices.size() - 3), index_t(vertices.size() - 2),
                                      index_t(vertices.size() - 1)}});//for (index_t i = 0; i < 3; ++i) {
                P[1] = P[2];
            }

            // Move to the next triangle incident to
            // vertex v.
            index_t lv = C.triangle_find_vertex(t, v);
            t = C.triangle_adjacent(t, (lv + 1) % 3);

            ++n;
        } while (t != C.vertex_triangle(v));
    }

    ps::registerSurfaceMesh(std::move(name), vertices, triangles);
}

inline void draw_cell_vertices(GEO::ConvexCell &C, std::string name) {
    std::vector<GEO::vec3> vertices;

    for (index_t t = 0; t < C.nb_t(); ++t) {
        index_t v = C.triangle_vertex(t, 0);
        if (v == VBW::END_OF_LIST) {
            continue;
        }
        GEO::vec3 pt = C.triangle_point(VBW::ushort(t));
        vertices.push_back(pt);
    }

    ps::registerPointCloud(std::move(name), vertices);
}

void load_obj(const std::string &path,
              std::vector<GEO::vec3> &points,
              std::vector<std::array<uint32_t, 3>> &triangles) {
    points.clear();
    triangles.clear();

    std::ifstream file(path);

    std::string line;
    GEO::vec3 min_pt(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max());
    GEO::vec3 max_pt(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest());

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            GEO::vec3 point;
            iss >> point.x >> point.y >> point.z;
            points.push_back(point);

            // Update the bounding box
            for (int i = 0; i < 3; i++) {
                min_pt[i] = std::min(min_pt[i], point[i]);
                max_pt[i] = std::max(max_pt[i], point[i]);
            }

        } else if (prefix == "f") {
            std::array<uint32_t, 3> triangle{};
            iss >> triangle[0] >> triangle[1] >> triangle[2];

            // OBJ indices start from 1, so we need to subtract 1 to make them 0-based
            triangle[0]--;
            triangle[1]--;
            triangle[2]--;

            triangles.push_back(triangle);
        }
    }

    // Calculate the scale and center for the unit cube transformation
    GEO::vec3 sizes = max_pt - min_pt;
    double scale = 1.0 / std::max({sizes.x, sizes.y, sizes.z});
    GEO::vec3 center = (min_pt + max_pt) * 0.5;

    // Scale to unit cube
    for (auto &pt: points) {
        pt = (pt - center) * scale;
    }
}

inline void load_box(std::vector<GEO::vec3> &points, std::vector<std::array<index_t, 3>> &triangles) {
    // Define the 8 vertices of a cube
    points.emplace_back(0, 0, 0);
    points.emplace_back(1, 0, 0);
    points.emplace_back(1, 1, 0);
    points.emplace_back(0, 1, 0);
    points.emplace_back(0, 0, 1);
    points.emplace_back(1, 0, 1);
    points.emplace_back(1, 1, 1);
    points.emplace_back(0, 1, 1);

    // Define the 12 triangles (two for each face)
    triangles.push_back({1, 0, 2});
    triangles.push_back({2, 0, 3});
    triangles.push_back({5, 1, 6});
    triangles.push_back({6, 1, 2});
    triangles.push_back({4, 5, 7});
    triangles.push_back({7, 5, 6});
    triangles.push_back({0, 4, 3});
    triangles.push_back({3, 4, 7});
    triangles.push_back({2, 3, 6});
    triangles.push_back({6, 3, 7});
    triangles.push_back({0, 1, 4});
    triangles.push_back({4, 1, 5});
}

inline void load_concave_thingy(std::vector<GEO::vec3> &points, std::vector<std::array<index_t, 3>> &triangles) {
    // Define the 8 vertices of a cube
    points.emplace_back(0, 0.6, 0.6);
    points.emplace_back(1, 0.6, 0.6);
    points.emplace_back(1, 1, 0);
    points.emplace_back(0, 1, 0);
    points.emplace_back(0, 0, 1);
    points.emplace_back(1, 0, 1);
    points.emplace_back(1, 1, 1);
    points.emplace_back(0, 1, 1);

    // Define the 12 triangles (two for each face)
    triangles.push_back({1, 0, 2});
    triangles.push_back({2, 0, 3});
    triangles.push_back({5, 1, 6});
    triangles.push_back({6, 1, 2});
    triangles.push_back({4, 5, 7});
    triangles.push_back({7, 5, 6});
    triangles.push_back({0, 4, 7});
    triangles.push_back({0, 7, 3});
    triangles.push_back({2, 3, 6});
    triangles.push_back({6, 3, 7});
    triangles.push_back({0, 1, 4});
    triangles.push_back({4, 1, 5});
}