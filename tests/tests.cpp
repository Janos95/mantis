#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "mantis.h"
#include <Model.h> // original p2m implementation

#include <random>

void load_obj(const std::string &path,
              std::vector<std::array<float, 3>> &points,
              std::vector<std::array<uint32_t, 3>> &triangles) {
    points.clear();
    triangles.clear();

    std::ifstream file(path);

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            std::array<float, 3> point{};
            iss >> point[0] >> point[1] >> point[2];
            points.push_back(point);
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

    // scale to unit cube
    std::array<float,3> min = points[0];
    std::array<float,3> max = points[0];
    for (auto &pt: points) {
        for (int i = 0; i < 3; i++) {
            min[i] = std::min(min[i], pt[i]);
            max[i] = std::max(max[i], pt[i]);
        }
    }
    std::array<float,3> center = {min[0] + (max[0] - min[0]) / 2.0f,
                                  min[1] + (max[1] - min[1]) / 2.0f,
                                  min[2] + (max[2] - min[2]) / 2.0f};
    std::array<float,3> d = {max[0] - min[0], max[1] - min[1], max[2] - min[2]};
    float scale = 1.0f / std::max({d[0], d[1], d[2]});
    for (auto &pt: points) {
        pt[0] = (pt[0] - center[0]) * scale;
        pt[1] = (pt[1] - center[1]) * scale;
        pt[2] = (pt[2] - center[2]) * scale;
    }
}

constexpr double limit_cube_len = 1e3;

auto build_p2m(const std::vector<std::array<float, 3>> &pts, const std::vector<std::array<uint32_t, 3>> &triangles) {
    std::vector<Eigen::Vector3d> points;
    std::vector<std::pair<int, int>> segments;
    std::vector<Eigen::Vector3i> faces;

    points.reserve(pts.size());
    for (const auto &p: pts) {
        points.emplace_back(p[0], p[1], p[2]);
    }

    faces.reserve(triangles.size());
    for (const auto &t: triangles) {
        faces.emplace_back(t[0], t[1], t[2]);
    }

    return Model(points, segments, faces, limit_cube_len);
}

std::array<float, 3> cross_product(const std::array<float, 3>& u, const std::array<float, 3>& v) {
    return {
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]
    };
}

float dot_product(const std::array<float, 3>& u, const std::array<float, 3>& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

float magnitude(const std::array<float, 3>& v) {
    return std::sqrt(dot_product(v, v));
}

float distp2p(const std::array<float, 3> &p1, const std::array<float, 3> &p2) {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    float dz = p1[2] - p2[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::array<float, 3> sub(const std::array<float, 3>& u, const std::array<float, 3>& v) {
    return {u[0] - v[0], u[1] - v[1], u[2] - v[2]};
}

float distance(const std::array<float, 3>& u, const std::array<float, 3>& v) {
    std::array<float, 3> diff = sub(u, v);
    return magnitude(diff);
}

float dist_edge(const std::array<float, 3> &p, const std::array<float, 3> &e1, const std::array<float, 3> &e2) {
    std::array<float, 3> edge_vector = sub(e2, e1);
    std::array<float, 3> point_vector = sub(p, e1);
    float c1 = dot_product(point_vector, edge_vector);
    float c2 = dot_product(edge_vector, edge_vector);

    if (c1 <= 0) {
        return distance(p, e1);
    }
    if (c2 <= c1) {
        return distance(p, e2);
    }

    float b = c1 / c2;
    std::array<float, 3> projection = {e1[0] + b * edge_vector[0], e1[1] + b * edge_vector[1], e1[2] + b * edge_vector[2]};
    return distance(p, projection);
}

float dist_plane(const std::array<float, 3> &p, const std::array<float, 3> &a, const std::array<float, 3> &b, const std::array<float, 3> &c) {
    std::array<float, 3> ab = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    std::array<float, 3> ac = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    std::array<float, 3> n = cross_product(ab, ac);
    std::array<float, 3> ap = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
    float distance = std::abs(dot_product(n, ap)) / magnitude(n);
    return distance;
}

bool check_random_samples(const mantis::AccelerationStructure &accelerator, const Model &model, size_t n, double eps) {
    std::default_random_engine gen(0);
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<std::array<float, 3>> test_queries(n);

    for (size_t i = 0; i < n; ++i) {
        test_queries[i] = {dist(gen), dist(gen), dist(gen)};
    }

    auto positions = accelerator.get_positions();
    auto edges = accelerator.get_edge_vertices();
    auto faces = accelerator.get_faces();

    double max_diff = 0.0;

    for (size_t i = 0; i < n; ++i) {
        P2M_Result p2m_result;
        auto q = test_queries[i];
        model.p2m({q[0], q[1], q[2]}, p2m_result);

        float x = test_queries[i][0];
        float y = test_queries[i][1];
        float z = test_queries[i][2];

        auto result = accelerator.calc_closest_point(x, y, z);
        float distance = std::sqrt(result.distance_squared);

        if(result.type == mantis::PrimitiveType::Vertex) {
            CHECK_LT(result.primitive_index , positions.size());
            std::array<float, 3> p = positions[result.primitive_index];
            float distToVertex = distp2p(p, {x, y, z});
            CHECK_EQ(distance, doctest::Approx(distToVertex).epsilon(eps));
        } else if(result.type == mantis::PrimitiveType::Edge) {
            CHECK_LT(result.primitive_index , edges.size());
            auto edge = edges[result.primitive_index];
            std::array<float, 3> e1 = positions[edge.first];
            std::array<float, 3> e2 = positions[edge.second];
            float distToEdge = dist_edge({x, y, z}, e1, e2);
            CHECK_EQ(distance, doctest::Approx(distToEdge).epsilon(eps));
        } else if(result.type == mantis::PrimitiveType::Face) {
            CHECK_LT(result.primitive_index , faces.size());
            auto face = faces[result.primitive_index];
            std::array<float, 3> a = positions[face[0]];
            std::array<float, 3> b = positions[face[1]];
            std::array<float, 3> c = positions[face[2]];
            float distToFace = dist_plane({x, y, z}, a, b, c);
            CHECK_EQ(distance, doctest::Approx(distToFace).epsilon(eps));
        }

        auto expectedDist = doctest::Approx(p2m_result.dis).epsilon(eps);
        float distToCp = distp2p({result.closest_point[0], result.closest_point[1], result.closest_point[2]}, {x, y, z});

        CHECK_EQ(distToCp, expectedDist);
        CHECK_EQ(distance, expectedDist);
    }

    CHECK(max_diff == doctest::Approx(0.0).epsilon(eps));

    if(max_diff > eps) {
        return false;
    }
    return true;
}

void run_test_case(const std::string& name, size_t num_samples, double eps) {
    std::string path = std::string(ASSETS_DIR) + name;
    std::vector<std::array<float, 3>> points;
    std::vector<std::array<uint32_t, 3>> triangles;
    load_obj(path, points, triangles);
    mantis::AccelerationStructure accelerator(points, triangles, limit_cube_len);
    auto model = build_p2m(points, triangles);
    check_random_samples(accelerator, model, num_samples, eps);
}


// TODO: not working because has boundary edges
//TEST_CASE("eba") {
//    run_test_case("eba.obj", 1e4, 1e-6);
//}

// TODO: not working because has non-manifold edges
//TEST_CASE("dragon") {
//    run_test_case("dragon.obj", 1e4, 1e-6);
//}

TEST_CASE("bunny") {
    run_test_case("bunny.obj", 1e4, 1e-6);
}

TEST_CASE("funny_cube") {
    run_test_case("funny_cube.obj", 1e4, 1e-6);
}

TEST_CASE("fandisk") {
    run_test_case("fandisk.obj", 1e4, 1e-6);
}

TEST_CASE("crank_pin") {
    run_test_case("crank_pin.obj", 1e4, 1e-6);
}