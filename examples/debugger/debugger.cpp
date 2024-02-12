#include "Delaunay_psm.h"
#include "mantis.h"
#include "util.h"

#include <random>

#include <Model.h>

double max_cube_len = 2;

Model *build_p2m(const std::vector<GEO::vec3> &pts, const std::vector<std::array<index_t, 3>> &triangles) {
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

    return new Model(points, segments, faces, max_cube_len);
}

std::vector<GEO::vec3> points;
std::vector<std::array<index_t, 3>> triangles;

mantis::AccelerationStructure *distanceToMesh = nullptr;
Model *model = nullptr;

int clamp(int x, int lower, int upper) {
    return std::min(std::max(x, lower), upper);
}

enum ElementType {
    Edge,
    Face
};

// for a vertex-element combination render the interception region
void callback() {
    static int v = 0;
    static int element = 0;
    static ElementType type = ElementType::Edge;

    bool update = false;
    static bool first = true;

    static bool both = false;

    if (ImGui::RadioButton("show both", both)) {
        both = !both;
        update = true;
    }
    update |= ImGui::InputInt("vertex", &v);
    update |= ImGui::InputInt("element", &element);

    v = clamp(v, 0, points.size() - 1);

    if (type == ElementType::Edge) {
        element = clamp(element, 0, distanceToMesh->edges.size() - 1);
    }
    if (type == ElementType::Face) {
        element = clamp(element, 0, distanceToMesh->faces.size() - 1);
    }

    // dropdown for edge or face
    ImGui::Combo("type", (int *) &type, "Edge\0Face\0");

    if(ImGui::Button("interception list length")) {
        auto length = distanceToMesh->intercepted_edges[v].size() + distanceToMesh->intercepted_faces[v].size();
        printf("interception list length: %d\n", int(length));
        auto p2m_length = model->regions[v].edges.size() + model->regions[v].faces.size();
        printf("P2M: interception list length: %d\n", int(p2m_length));
    }

    if (ImGui::Button("is in interception list")) {
        bool p2m_found = false;
        bool found = false;
        auto &intercepted =
                type == ElementType::Edge ? distanceToMesh->intercepted_edges[v] : distanceToMesh->intercepted_faces[v];
        for (auto el: intercepted) {
            if (el == element) {
                found = true;
                break;
            }
        }

        if(type == ElementType::Edge) {
            auto s = distanceToMesh->edges[element].start;
            auto t = distanceToMesh->edges[element].end;
            for(auto e : model->regions[v].edges) {
                if(s == model->edges[e].s && t == model->edges[e].t) {
                    p2m_found = true;
                    break;
                }
            }
        } else {
            for(auto f : model->regions[v].faces) {
                if(element == f) {
                    p2m_found = true;
                    break;
                }
            }
        }
        printf("Is in interception list %d\n", int(found));
        printf("P2M: Is in interception list %d\n", int(p2m_found));
    }

    if (update || first) {
        first = false;

        auto pt = points[v];
        draw_point("point", pt);

        if (type == ElementType::Edge) {
            GEO::vec3 a = points[distanceToMesh->edges[element].start];
            GEO::vec3 b = points[distanceToMesh->edges[element].end];
            ps::removeSurfaceMesh("face");
            draw_edge("edge", a, b);
        }
        if (type == ElementType::Face) {
            GEO::vec3 a = points[triangles[element][0]];
            GEO::vec3 b = points[triangles[element][1]];
            GEO::vec3 c = points[triangles[element][2]];
            ps::removeCurveNetwork("edge");
            draw_triangle_face("face", a, b, c);
        }

        ps::removeSurfaceMesh("vor_cell");
        ps::removeSurfaceMesh("trim_cell");
        ps::removeSurfaceMesh("cell");

        if (type == ElementType::Edge) {
            if (both) {
                auto vor_cell = distanceToMesh->vor_cells[v];
                auto edge_cell = distanceToMesh->edge_cells[element];
                draw_cell(vor_cell, "vor_cell");
                draw_cell(edge_cell, "trim_cell");
            } else {
                auto cell = distanceToMesh->vertex_edge_cells[{v, element}];
                draw_cell(cell, "cell");
            }
        } else {
            if (both) {
                auto vor_cell = distanceToMesh->vor_cells[v];
                auto face_cell = distanceToMesh->face_cells[element];
                draw_cell(vor_cell, "vor_cell");
                draw_cell(face_cell, "trim_cell");
            } else {
                auto cell = distanceToMesh->vertex_face_cells[{v, element}];
                draw_cell(cell, "cell");
                draw_cell_vertices(cell, "cell_vertices");
            }
        }
    }
}

int main(int, char **) {
    GEO::initialize();
    ps::init();

    //load_box(points, triangles);
    //load_concave_thingy(points, triangles);
    load_obj("/Users/jmeny/CLionProjects/mantis/funny_cube.obj", points, triangles);

    auto mesh = ps::registerSurfaceMesh("mesh", points, triangles);
    mesh->setTransparency(0.5);

    distanceToMesh = new DisctanceToMesh(points, triangles, max_cube_len);
    distanceToMesh->compute_interception_list();

    //for (index_t v) {
    //    printf("nb intercepted elements %d\n", int(list.size()));
    //}

    printf("finished computing interception list\n");

    model = build_p2m(points, triangles);

    ps::state::userCallback = callback;
    ps::show();
}
