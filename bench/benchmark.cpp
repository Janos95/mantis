#include "Delaunay_psm.h"
#include "mantis.h"

#include <random>
#include <chrono>

#include "Model.h"

#define FCPW_USE_ENOKI
#define FCPW_SIMD_WIDTH 4
#include <fcpw/fcpw.h>

using index_t = GEO::index_t;

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
            std::array<uint32_t, 3> triangle;
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

void load_box(std::vector<GEO::vec3> &points, std::vector<std::array<index_t, 3>> &triangles) {
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


void load_concave_thingy(std::vector<GEO::vec3> &points, std::vector<std::array<index_t, 3>> &triangles) {
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


constexpr double limit_cube_len = 1e3;

auto build_p2m(const std::vector<GEO::vec3> &pts, const std::vector<std::array<index_t, 3>> &triangles) {
    std::vector<Eigen::Vector3d> points;
    std::vector<std::pair<int, int>> segments;
    std::vector<Eigen::Vector3i> faces;

    points.reserve(pts.size());
    for (const auto &p: pts) {
        points.emplace_back(p.x, p.y, p.z);
    }

    faces.reserve(triangles.size());
    for (const auto &t: triangles) {
        faces.emplace_back(t[0], t[1], t[2]);
    }

    return Model(points, segments, faces, limit_cube_len);
}

int main(int, char **) {

    std::vector<GEO::vec3> points;
    std::vector<std::array<index_t, 3>> triangles;

    std::string path = std::string(ASSETS_DIR) + "dragon.obj";
    load_obj(path, points, triangles);

    std::vector<float> points_data(points.size() * 3);
    for(size_t i = 0; i < points.size(); ++i) {
        points_data[3*i + 0] = (float)points[i].x;
        points_data[3*i + 1] = (float)points[i].y;
        points_data[3*i + 2] = (float)points[i].z;
    }

    auto start = std::chrono::high_resolution_clock::now();
    mantis::AccelerationStructure accelerator(points_data.data(), points.size(), (uint32_t*)triangles.data(), triangles.size(), limit_cube_len);
    auto end = std::chrono::high_resolution_clock::now();

    printf("mantis build time: %f\n", std::chrono::duration<double, std::milli>(end - start).count());

    // print average length of interception list
    start = std::chrono::high_resolution_clock::now();
    auto model = build_p2m(points, triangles);
    end = std::chrono::high_resolution_clock::now();
    printf("p2m build time: %f ms\n", std::chrono::duration<double, std::milli>(end - start).count());

    const index_t n = 1'000'000;

    //std::vector<GEO::vec3> test_queries(n);

    std::default_random_engine gen(0);
    std::uniform_real_distribution<double> dist(-5, 5);

    std::vector<GEO::vec3> test_queries(n);

    for (index_t i = 0; i < n; ++i) {
        test_queries[i] = {dist(gen), dist(gen), dist(gen)};
    }

    std::vector<double> result_p2m(n);

    // first test model
    start = std::chrono::high_resolution_clock::now();
    for (index_t i = 0; i < n; ++i) {
        P2M_Result result;
        auto q = test_queries[i];
        model.p2m({q.x, q.y, q.z}, result);
        result_p2m[i] = result.dis;
    }
    end = std::chrono::high_resolution_clock::now();
    printf("model time: %f ms\n", std::chrono::duration<double, std::milli>(end - start).count());

    std::vector<float> result_mantis(n);

    start = std::chrono::high_resolution_clock::now();
    for (index_t i = 0; i < n; ++i) {
        auto pt = test_queries[i];
        result_mantis[i] = std::sqrt(accelerator.calc_closest_point(pt.x, pt.y, pt.z).distance_squared);
    }
    end = std::chrono::high_resolution_clock::now();
    printf("mantis time: %f ms\n", std::chrono::duration<double, std::milli>(end - start).count());

    {
        fcpw::Scene<3> scene;

        scene.setObjectTypes({{fcpw::PrimitiveType::Triangle}});

        scene.setObjectVertexCount(points.size(), 0);
        scene.setObjectTriangleCount(triangles.size(), 0);

        for (int i = 0; i < points.size(); i++) {
            Eigen::Vector3f pt = {points[i].x, points[i].y, points[i].z};
            scene.setObjectVertex(pt, i, 0);
        }

        for (int i = 0; i < triangles.size(); i++) {
            int tri[3] = {(int)triangles[i][0], (int)triangles[i][1], (int)triangles[i][2]};
            scene.setObjectTriangle(tri, i, 0);
        }

        scene.computeObjectNormals(0);

        scene.computeSilhouettes();

        scene.build(fcpw::AggregateType::Bvh_SurfaceArea, true); // the second boolean argument enables vectorization

        fcpw::Interaction<3> cpqInteraction;

        std::vector<float> result_fcpw(n);

        start = std::chrono::high_resolution_clock::now();
        for (index_t i = 0; i < n; ++i) {
            auto pt = test_queries[i];
            Eigen::Vector3f queryPoint = {pt.x, pt.y, pt.z};
            scene.findClosestPoint(queryPoint, cpqInteraction);
            result_fcpw[i] = cpqInteraction.d;
        }
        end = std::chrono::high_resolution_clock::now();

        printf("fcpw time: %f ms\n", std::chrono::duration<double, std::milli>(end - start).count());

        // find maximum difference
        //double max_diff = 0.0;
        //for(index_t i = 0; i < n; ++i) {
        //    double diff = std::abs(result_mantis[i] - result_p2m[i]);
        //    max_diff = std::max(max_diff, diff);

        //}
        //printf("max_diff: %f\n", max_diff);
    }
}
