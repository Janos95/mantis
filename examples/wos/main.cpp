#include "mantis.h"
#include "util.h"

#include <thread>

std::vector<std::array<index_t, 3>> triangles;
std::vector<GEO::vec3> points;

const int n = 30;

std::vector<GEO::vec3> square_points;
std::vector<std::array<index_t, 3>> square_triangles;

std::vector<float> values;

std::unique_ptr<mantis::AccelerationStructure> accelerator;

template<class F>
void parallel_for(size_t begin, size_t end, F f) {
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

template<class F>
float wos(GEO::vec3 p0, F g) {
    const float eps = 0.01;
    const int nWalks = 128;
    const int maxSteps = 16;

    static std::random_device rd{};
    static std::default_random_engine gen{rd()};
    std::normal_distribution dist;

    float sum = 0.;
    for (int i = 0; i < nWalks; i++) {
        GEO::vec3 p = p0;
        float R;
        mantis::Result result;
        int steps = 0;
        do {
            result = accelerator->calc_closest_point(p.x, p.y, p.z);
            R = std::sqrt(result.distance_squared);
            p = p + R * GEO::normalize(GEO::vec3(dist(gen), dist(gen), dist(gen)));
            steps++;
        } while (R > eps && steps < maxSteps);

        sum += g(result);
    }
    return sum / nWalks;
}

GEO::vec3 calc_barys(GEO::vec3 p, GEO::vec3 a, GEO::vec3 b, GEO::vec3 c) {
    GEO::vec3 v0 = b - a, v1 = c - a, v2 = p - a;
    double d00 = GEO::dot(v0, v0);
    double d01 = GEO::dot(v0, v1);
    double d11 = GEO::dot(v1, v1);
    double d20 = GEO::dot(v2, v0);
    double d21 = GEO::dot(v2, v1);
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;
    return {u, v, w};
}

double calc_bary_along_line(GEO::vec3 p, GEO::vec3 a, GEO::vec3 b) {
    GEO::vec3 ab = b - a;
    GEO::vec3 ap = p - a;
    double ab_length_squared = GEO::dot(ab, ab);
    double ab_ap_dot = GEO::dot(ab, ap);
    return ab_ap_dot / ab_length_squared;
}

GEO::vec3 to_vec3(float *p) {
    return {p[0], p[1], p[2]};
}

// for a vertex-element combination render the interception region
void callback() {

    auto g = [](mantis::Result result) -> float {
        switch (result.type) {
            case mantis::PrimitiveType::Vertex:
                return values[result.primitive_index];
            case mantis::PrimitiveType::Edge: {
                auto edge = accelerator->get_edge(result.primitive_index);
                uint32_t a = edge.first;
                uint32_t b = edge.second;
                GEO::vec3 pa = points[a];;
                GEO::vec3 pb = points[b];
                GEO::vec3 cp = to_vec3(result.closest_point);
                auto t = (float) calc_bary_along_line(cp, pa, pb);
                return (1.f - t) * values[a] + t * values[b];
            }
            case mantis::PrimitiveType::Face: {
                index_t a = triangles[result.primitive_index][0];
                index_t b = triangles[result.primitive_index][1];
                index_t c = triangles[result.primitive_index][2];
                float av = values[a];
                float bv = values[b];
                float cv = values[c];
                GEO::vec3 pa = points[a];
                GEO::vec3 pb = points[b];
                GEO::vec3 pc = points[c];
                GEO::vec3 cp = to_vec3(result.closest_point);
                GEO::vec3 bary = calc_barys(cp, pa, pb, pc);
                return av * (float) bary.x + bv * (float) bary.y + cv * (float) bary.z;
            }
            default:
                assert(false);
                return 0.f;
        }
    };

    double t = ImGui::GetTime();

    auto queries = square_points;
    std::vector<float> interpolated_normals_x(queries.size());

    auto now = std::chrono::high_resolution_clock::now();
    parallel_for(0, queries.size(), [&](size_t i) {
        queries[i].z = std::sin(t) * 0.4;
        interpolated_normals_x[i] = wos(queries[i], g);
    });
    auto end = std::chrono::high_resolution_clock::now();
    // print in ms
    printf("elapsed time: %f ms\n", std::chrono::duration<double, std::milli>(end - now).count());
    ps::registerSurfaceMesh("square", queries, square_triangles)
    ->addVertexScalarQuantity("values",interpolated_normals_x)
    ->setEnabled(true);
}

void make_square() {
    float step = 2.0f / n; // Calculate the step size for subdivisions

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            float x = (-1.f + float(j) * step) * 0.7f;
            float y = (-1.f + float(i) * step) * 0.7f;
            square_points.emplace_back(x, y, 0); // z-coordinate is 0 as it's in the xy-plane
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            index_t topLeft = i * (n + 1) + j;
            index_t topRight = topLeft + 1;
            index_t bottomLeft = topLeft + n + 1;
            index_t bottomRight = bottomLeft + 1;
            square_triangles.push_back({topLeft, bottomLeft, topRight});
            square_triangles.push_back({topRight, bottomLeft, bottomRight});
        }
    }
}

int main(int, char **) {
    ps::options::groundPlaneMode = ps::GroundPlaneMode::ShadowOnly;
    ps::init();

    std::string path = std::string(ASSETS_DIR) + "bunny.obj";
    load_obj(path, points, triangles);

    std::vector<float> pts_data(points.size() * 3);
    for (int i = 0; i < points.size(); ++i) {
        pts_data[i * 3 + 0] = (float) points[i].x;
        pts_data[i * 3 + 1] = (float) points[i].y;
        pts_data[i * 3 + 2] = (float) points[i].z;
    }

    std::vector<GEO::vec3> vertex_normals(points.size());
    for (auto &triangle: triangles) {
        GEO::vec3 a = points[triangle[0]];
        GEO::vec3 b = points[triangle[1]];
        GEO::vec3 c = points[triangle[2]];
        GEO::vec3 normal = GEO::normalize(GEO::cross(b - a, c - a));
        vertex_normals[triangle[0]] += normal;
        vertex_normals[triangle[1]] += normal;
        vertex_normals[triangle[2]] += normal;
    }

    values.resize(points.size(), 0.f);
    for (size_t i = 0; i < points.size(); ++i) {
        values[i] = (float)vertex_normals[i].x;
    }

    auto m = ps::registerSurfaceMesh("mesh", points, triangles);
    auto vs = m->addVertexScalarQuantity("values", values);
    vs->setEnabled(true);

    make_square();

    printf("finished computing interception list\n");

    //model = build_p2m(points, triangles);
    accelerator = std::make_unique<mantis::AccelerationStructure>(pts_data.data(), points.size(),
                                                                  (index_t *) triangles.data(), triangles.size());

    ps::state::userCallback = callback;
    ps::show();
}