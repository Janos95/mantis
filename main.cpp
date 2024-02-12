#include "Delaunay_psm.h"
#include "mantis.h"
#include "util.h"

#include <random>

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
    GEO::initialize();
    //ps::init();

    std::vector<GEO::vec3> points;
    std::vector<std::array<index_t, 3>> triangles;
    //load_box(points, triangles);
    //load_concave_thingy(points, triangles);
    load_obj("/Users/jmeny/CLionProjects/mantis/bunny.obj", points, triangles);

    //ps::registerSurfaceMesh("mesh", points, triangles);

    auto start = std::chrono::high_resolution_clock::now();
    DistanceToMesh distanceToMesh(points, triangles, limit_cube_len);

    printf("starting computing interception list\n");
    distanceToMesh.compute_interception_list();
    printf("finished computing interception list\n");

    auto end = std::chrono::high_resolution_clock::now();

    printf("mantis build time: %f\n", std::chrono::duration<double, std::milli>(end - start).count());

    //draw_point("v5", points[5]);
    //ps::show();

    // print max length of interception list
    //index_t max_interception_list_length = 0;
    //for (index_t v = 0; v < points.size(); ++v) {
    //    index_t length = distanceToMesh.intercepted_faces[v].size() + distanceToMesh.intercepted_edges[v].size();
    //    max_interception_list_length = std::max(max_interception_list_length, length);
    //}
    //printf("max interception list length: %d\n", (int) max_interception_list_length);

    // print average length of interception list
    double avg_edge_interception_list_length = 0.0;
    double avg_face_interception_list_length = 0.0;
    for (index_t v = 0; v < points.size(); ++v) {
        avg_edge_interception_list_length += distanceToMesh.intercepted_edges_packed[v].size() * 4;
        avg_face_interception_list_length += distanceToMesh.intercepted_faces_packed[v].size() * 4;
    }
    avg_edge_interception_list_length /= points.size();
    avg_face_interception_list_length /= points.size();
    printf("avg face interception list length: %f\n", avg_face_interception_list_length);
    printf("avg edge interception list length: %f\n", avg_edge_interception_list_length);

    start = std::chrono::high_resolution_clock::now();
    auto model = build_p2m(points, triangles);
    end = std::chrono::high_resolution_clock::now();
    printf("p2m build time: %f\n", std::chrono::duration<double, std::milli>(end - start).count());

    avg_edge_interception_list_length = 0;
    avg_face_interception_list_length = 0;
    for(index_t v = 0 ; v < points.size(); ++v) {
        avg_edge_interception_list_length += model.regions[v].edges.size();
        avg_face_interception_list_length += model.regions[v].faces.size();
    }
    avg_edge_interception_list_length /= points.size();
    avg_face_interception_list_length /= points.size();

    printf("p2m: avg face interception list length: %f\n", avg_face_interception_list_length);
    printf("p2m: avg edge interception list length: %f\n", avg_edge_interception_list_length);

    //const index_t n = 10'000'000;
    const index_t n = 1'000'000;
    //const index_t n = 1'000;
    //std::vector<GEO::vec3> test_queries(n);

    std::default_random_engine gen(0);
    std::uniform_real_distribution<double> dist(-1, 1);

    std::vector<GEO::vec3> test_queries(n);

    for (index_t i = 0; i < n; ++i) {
        test_queries[i] = {dist(gen), dist(gen), dist(gen)};
    }

    std::vector<double> result_model(n);

    // first test model
    start = std::chrono::high_resolution_clock::now();
    for (index_t i = 0; i < n; ++i) {
        P2M_Result result;
        auto q = test_queries[i];
        model.p2m({q.x, q.y, q.z}, result);
        result_model[i] = result.dis;
    }
    end = std::chrono::high_resolution_clock::now();
    printf("model time: %f\n", std::chrono::duration<double, std::milli>(end - start).count());

    std::vector<double> result_ours(n);

    start = std::chrono::high_resolution_clock::now();
    for (index_t i = 0; i < n; ++i) {
        result_ours[i] = std::sqrt(distanceToMesh.calc_closest_point(test_queries[i]).distance_squared);
    }
    end = std::chrono::high_resolution_clock::now();
    printf("ours time: %f\n", std::chrono::duration<double, std::milli>(end - start).count());

    // compare results
    double max_diff = 0.0;
    for (index_t i = 0; i < n; ++i) {
        //if(std::abs(result_model[i] - result_ours[i]) > 1e-5) {
        //    printf("%d\n", int(i));
        //}
        max_diff = std::max(max_diff, std::abs(result_model[i] - result_ours[i]));
    }
    printf("max diff: %.10f\n", max_diff);
}
