#include <iostream>
#include <chrono>
#include <memory>
#include <fstream>

#include <polyscope/point_cloud.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>

#include "implicit_meshing.h"
#include "mantis.h"

#include "util.h"

namespace ps = polyscope;

float fOpUnionRound(float a, float b, float r) {
    glm::vec2 u = glm::max(glm::vec2(r - a, r - b), glm::vec2(0));
    return glm::max(r, glm::min(a, b)) - glm::length(u);
}

mantis::AccelerationStructure *accelerator = nullptr;

std::vector<glm::vec3> vertex_normals;
std::vector<glm::vec3> edge_normals;
std::vector<glm::vec3> face_normals;

// This is the function that will be called every frame
void callback() {
    double t = ImGui::GetTime();
    glm::vec3 center = {0.5 * std::sin(t), 0.5* std::cos(t), 0};
    //auto f = [center](Vec3 q) -> double {
    //    double cube = sdBox(q, Vec3(0.5));
    //    double sphere = length(q - center) - 0.3f;
    //    return fOpUnionRound(cube, sphere, 0.1f);
    //};

    auto get_normal = [](mantis::Result result) {
        switch (result.type) {
            case mantis::PrimitiveType::Vertex:
                return vertex_normals[result.primitive_index];
            case mantis::PrimitiveType::Edge: {
                return edge_normals[result.primitive_index];
            }
            case mantis::PrimitiveType::Face: {
                return face_normals[result.primitive_index];
            }
            default: return glm::vec3{};
        }
    };

    auto f = [center, get_normal](Vec3 q1) -> double {
        glm::vec3 q(q1.x, q1.y, q1.z);
        auto result = accelerator->calc_closest_point(q.x, q.y, q.z);
        glm::vec3 n = get_normal(result);
        glm::vec3 cp(result.closest_point[0], result.closest_point[1], result.closest_point[2]);
        float sign = glm::dot(n, q - cp) > 0 ? 1.f : -1.f;
        float sdf_mesh = std::sqrt(result.distance_squared) * sign;
        float sphere = glm::length(q - center) - 0.3f;
        return fOpUnionRound(sdf_mesh, sphere, 0.25);
    };

    auto start = std::chrono::high_resolution_clock::now();
    auto mesh = generate_mesh(f, 800);
    auto end = std::chrono::high_resolution_clock::now();
    ps::registerSurfaceMesh("mesh", mesh.vertices, mesh.quads);
}

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

int main() {
    ps::options::groundPlaneMode = ps::GroundPlaneMode::ShadowOnly;
    ps::init();

    //ps::SurfaceMesh* mesh = nullptr;

    // load mesh and build acceleration structure
    {
        std::vector<std::array<float, 3>> points;
        std::vector<std::array<uint32_t, 3>> triangles;
        load_obj("/Users/jmeny/CLionProjects/hackweek-testbed/bunny.obj", points, triangles);

        if (points.empty()) {
            std::cerr << "No points loaded" << std::endl;
            return 0;
        }

        float limit_cube_len = 20.;
        accelerator = new mantis::AccelerationStructure(points, triangles, limit_cube_len);

        //mesh = ps::registerSurfaceMesh("my mesh", points, triangles);
    }

    // compute normals
    {
        std::vector<std::array<float, 3>> positions = accelerator->get_positions();
        std::vector<std::array<uint32_t, 3>> faces = accelerator->get_faces();
        std::vector<std::array<uint32_t, 3>> face_edges = accelerator->get_face_edges();

        auto num_edges = accelerator->num_edges();

        face_normals.resize(faces.size());
        edge_normals.resize(num_edges);

        for (size_t i = 0; i < faces.size(); ++i) {
            const auto &face = faces[i];
            const auto &edges = face_edges[i];

            const auto &p0 = positions[face[0]];
            const auto &p1 = positions[face[1]];
            const auto &p2 = positions[face[2]];

            glm::vec3 e1 = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            glm::vec3 e2 = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};

            glm::vec3 normal = glm::normalize(glm::cross(e1, e2));
            face_normals[i] = normal;

            for (size_t j = 0; j < 3; ++j) {
                const auto &edge = edges[j];
                edge_normals[edge] += normal;
            }
        }

        for (auto &edge_normal: edge_normals) {
            edge_normal = glm::normalize(edge_normal);
        }

        vertex_normals.resize(positions.size());
        for (size_t j = 0; j < faces.size(); ++j) {

            glm::vec3 normal = face_normals[j];
            auto face = faces[j];

            for (int i = 0; i < 3; ++i) {
                auto currVertex = face[i];
                auto nextVertex = face[(i + 1) % 3];
                auto prevVertex = face[(i + 2) % 3];

                glm::vec3 edge1 = {positions[nextVertex][0] - positions[currVertex][0],
                                   positions[nextVertex][1] - positions[currVertex][1],
                                   positions[nextVertex][2] - positions[currVertex][2]};

                glm::vec3 edge2 = {positions[prevVertex][0] - positions[currVertex][0],
                                   positions[prevVertex][1] - positions[currVertex][1],
                                   positions[prevVertex][2] - positions[currVertex][2]};

                float angle = glm::acos(glm::dot(edge1, edge2) / (glm::length(edge1) * glm::length(edge2)));
                vertex_normals[currVertex][0] += normal[0] * angle;
                vertex_normals[currVertex][1] += normal[1] * angle;
                vertex_normals[currVertex][2] += normal[2] * angle;
            }
        }
    }

    ps::state::userCallback = callback;

    //mesh->addVertexVectorQuantity("vertex normals", vertex_normals);
    //mesh->addFaceVectorQuantity("face normals", face_normals);

    ps::show();

    return 0;
}
