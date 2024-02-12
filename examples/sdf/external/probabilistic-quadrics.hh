#pragma once

// ===============================================================================
// Reference implementation of the Eurographics 2020 paper 
// "Fast and Robust QEF Minimization using Probabilistic Quadrics".
// 
// Project page: https://graphics.rwth-aachen.de/probabilistic-quadrics
// 
// ```
// @article{Trettner2020,
//     journal = {Computer Graphics Forum},
//     title = {{Fast and Robust QEF Minimization using Probabilistic Quadrics}},
//     author = {Philip Trettner and Leif Kobbelt},
//     year = {2020},
// }
// ```
// NOTE: the entry is not complete as the pages and doi have yet to be assigned.
// ===============================================================================


// ===============================================================================
// MIT License
// 
// Copyright (c) 2020 Philip Trettner
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ===============================================================================

#include <type_traits>

namespace pq
{
// ============== Forward Declarations ==============

/// A quadric is defined using a math type traits class
///
/// Usage examples:
///
///   // our probabilistic quadrics
///   #include "probabilistic-quadrics.hh"
///   
///   // some math library (see below for different options)
///   #include "minimal-math.hh"
///   
///   // optional: typedef your quadric type
///   using quadric3 = pq::quadric<pq::minimal_math<float>>;
///   using dquadric3 = pq::quadric<pq::minimal_math<double>>;
///   
///   // quadrics are value types with proper operator overloads
///   quadric3 q;
///   q = q + q;
///   q = q - q;
///   q = q * 3;
///   q = q / 2.5f;
///   
///   // quadrics can be evaluated at positions
///   q(1, 2, 3);
///   q({1, 2, 3});
///   q(some_pos);
///   
///   // quadrics can be created from coefficients
///   q = quadric3::from_coefficients(some_mat3, some_vec3, some_scalar);
///   
///   // quadric minimizers can be computed (using matrix inversion internally)
///   pq::pos3 min_p = q.minimizer();
///   
///   // some classical quadrics are predefined:
///   q = quadric3::point_quadric(some_pos);
///   q = quadric3::plane_quadric(some_pos, some_normal_vec);
///   q = quadric3::triangle_quadric(p0, p1, p2);
///   
///   // our probabilistic plane quadrics in isotropic or general form:
///   float stddev_pos = ...;
///   float stddev_normal = ...;
///   pq:mat3 sigma_pos = ...;
///   pq:mat3 sigma_normal = ...;
///   q = quadric3::probabilistic_plane_quadric(mean_pos, mean_normal, stddev_pos, stddev_normal);
///   q = quadric3::probabilistic_plane_quadric(mean_pos, mean_normal, sigma_pos, sigma_normal);
///   
///   // our probabilistic triangle quadrics in isotropic or general form:
///   float stddev_pos = ...;
///   pq:mat3 sigma_p0 = ...;
///   pq:mat3 sigma_p1 = ...;
///   pq:mat3 sigma_p2 = ...;
///   q = quadric3::probabilistic_triangle_quadric(p0, p1, p2, stddev_pos);
///   q = quadric3::probabilistic_triangle_quadric(p0, p1, p2, sigma_p0, sigma_p1, sigma_p2);
///
///
/// The following math classes are tested:
///
///   the built-in minimal-math.hh:
///     pq::minimal_math<float>
///     pq::minimal_math<double>
///
///   Typed Geometry:
///     pq::math<float, tg::pos3, tg::vec3, tg::mat3>
///     pq::math<double, tg::dpos3, tg::dvec3, tg::dmat3>
///
///   GLM:
///     pq::math<float, glm::vec3, glm::vec3, glm::mat3>
///     pq::math<double, glm::dvec3, glm::dvec3, glm::dmat3>
///
///   Eigen:
///     pq::math<float, Eigen::Vector3f, Eigen::Vector3f, Eigen::Matrix3f>
///     pq::math<double, Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix3d>
///
template <class MathT>
struct quadric;

// ============== Math Abstraction ==============

/**
 *
 * Generic adaptor to support different vector math libraries.
 * Tested with:
 *   - Typed Geometry
 *   - GLM
 *   - Eigen
 *   - builtin minimal-math.hh
 *
 * To make your custom math library work, it needs to provide the following operations:
 *   - `pos - pos -> vec`
 *   - `pos + vec -> pos`
 *   - `pos - vec -> pos`
 *   - `vec + vec -> vec`
 *   - `vec - vec -> vec`
 *   - `vec * scalar -> vec`
 *   - `vec / scalar -> vec`
 *   - `pos * scalar -> pos`
 *   - `pos / scalar -> pos`
 *   - `mat * scalar -> mat`
 *   - `mat * vec -> vec`
 *   - `mat + mat -> mat`
 *   - `mat - mat -> mat`
 *   - `pos[int-literal] -> scalar&`
 *   - `vec[int-literal] -> scalar&`
 *   - `mat[col][row] -> scalar&` OR `mat(row, col) -> scalar&`
 *   - `mat`, `pos` and `vec` default constructor
 *   - `mat`, `pos`, `vec`, and `scalar` behave like value types
 *   - `scalar` must be constructable from integer literals
 *
 * where
 *   - pos is a 3D position type
 *   - vec is a 3D vector type
 *   - pos and vec can be the same, e.g. glm::vec3
 *   - mat is a 3x3 matrix type
 */
template <class ScalarT, class Pos3, class Vec3, class Mat3>
struct math
{
    // make types available as math::type
    using scalar_t = ScalarT;
    using pos3 = Pos3;
    using vec3 = Vec3;
    using mat3 = Mat3;

    template <class MatT = Mat3>
    static auto get_impl(MatT& m, int x, int y, int) -> decltype(m[x][y])
    {
        return m[x][y];
    }
    template <class MatT = Mat3> // workaround for Eigen matrices
    static auto get_impl(MatT& m, int x, int y, char) -> decltype(m(y, x))
    {
        return m(y, x);
    }
    static scalar_t& get(Mat3& m, int x, int y) { return get_impl(m, x, y, 0); }
    static scalar_t const& get(Mat3 const& m, int x, int y) { return get_impl(m, x, y, 0); }

    static pos3 make_pos(scalar_t x, scalar_t y, scalar_t z)
    {
        pos3 p;
        p[0] = x;
        p[1] = y;
        p[2] = z;
        return p;
    }

    static vec3 make_vec(scalar_t x, scalar_t y, scalar_t z)
    {
        vec3 v;
        v[0] = x;
        v[1] = y;
        v[2] = z;
        return v;
    }

    static vec3 to_vec(pos3 const& p)
    {
        vec3 v;
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        return v;
    }

    static mat3 make_identity()
    {
        mat3 m;

        math::get(m, 0, 0) = scalar_t(1);
        math::get(m, 0, 1) = scalar_t(0);
        math::get(m, 0, 2) = scalar_t(0);

        math::get(m, 1, 0) = scalar_t(0);
        math::get(m, 1, 1) = scalar_t(1);
        math::get(m, 1, 2) = scalar_t(0);

        math::get(m, 2, 0) = scalar_t(0);
        math::get(m, 2, 1) = scalar_t(0);
        math::get(m, 2, 2) = scalar_t(1);

        return m;
    }

    static vec3 cross(vec3 const& a, vec3 const& b)
    {
        return math::make_vec(a[1] * b[2] - a[2] * b[1], //
                              a[2] * b[0] - a[0] * b[2], //
                              a[0] * b[1] - a[1] * b[0]);
    }

    template <class A, class B> // can be pos3 or vec3
    static scalar_t dot(A const& a, B const& b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    static scalar_t trace_of_product(mat3 const& A, mat3 const& B)
    {
        // trace(A * B)
        auto r = ScalarT(0);
        for (auto i = 0; i < 3; ++i)
            for (auto j = 0; j < 3; ++j)
                r += math::get(A, i, j) * math::get(B, j, i);
        return r;
    }

    static mat3 self_outer_product(vec3 const& v)
    {
        auto const a = v[0];
        auto const b = v[1];
        auto const c = v[2];

        mat3 M;

        math::get(M, 0, 0) = a * a;
        math::get(M, 1, 1) = b * b;
        math::get(M, 2, 2) = c * c;

        math::get(M, 1, 0) = a * b;
        math::get(M, 2, 0) = a * c;
        math::get(M, 2, 1) = b * c;

        math::get(M, 0, 1) = a * b;
        math::get(M, 0, 2) = a * c;
        math::get(M, 1, 2) = b * c;

        return M;
    }

    static mat3 cross_product_squared_transpose(vec3 const& v)
    {
        auto const a = v[0];
        auto const b = v[1];
        auto const c = v[2];
        auto const a2 = a * a;
        auto const b2 = b * b;
        auto const c2 = c * c;

        mat3 M;

        math::get(M, 0, 0) = b2 + c2;
        math::get(M, 1, 1) = a2 + c2;
        math::get(M, 2, 2) = a2 + b2;

        math::get(M, 1, 0) = -a * b;
        math::get(M, 2, 0) = -a * c;
        math::get(M, 2, 1) = -b * c;

        math::get(M, 0, 1) = -a * b;
        math::get(M, 0, 2) = -a * c;
        math::get(M, 1, 2) = -b * c;

        return M;
    }

    static mat3 cross_interference_matrix(mat3 const& A, mat3 const& B)
    {
        mat3 m;

        auto constexpr x = 0;
        auto constexpr y = 1;
        auto constexpr z = 2;

        auto const cxx = math::get(A, y, z) * math::get(B, y, z);
        auto const cyy = math::get(A, x, z) * math::get(B, x, z);
        auto const czz = math::get(A, x, y) * math::get(B, x, y);

        math::get(m, x, x) = math::get(A, y, y) * math::get(B, z, z) - cxx - cxx + math::get(A, z, z) * math::get(B, y, y);
        math::get(m, y, y) = math::get(A, x, x) * math::get(B, z, z) - cyy - cyy + math::get(A, z, z) * math::get(B, x, x);
        math::get(m, z, z) = math::get(A, x, x) * math::get(B, y, y) - czz - czz + math::get(A, y, y) * math::get(B, x, x);

        math::get(m, x, y) = -math::get(A, x, y) * math::get(B, z, z) + math::get(A, x, z) * math::get(B, y, z)
                             + math::get(A, y, z) * math::get(B, x, z) - math::get(A, z, z) * math::get(B, x, y);
        math::get(m, x, z) = math::get(A, x, y) * math::get(B, y, z) - math::get(A, x, z) * math::get(B, y, y)
                             - math::get(A, y, y) * math::get(B, x, z) + math::get(A, y, z) * math::get(B, x, y);
        math::get(m, y, z) = -math::get(A, x, x) * math::get(B, y, z) + math::get(A, x, y) * math::get(B, x, z)
                             + math::get(A, x, z) * math::get(B, x, y) - math::get(A, y, z) * math::get(B, x, x);

        math::get(m, y, x) = math::get(m, x, y);
        math::get(m, z, x) = math::get(m, x, z);
        math::get(m, z, y) = math::get(m, y, z);

        return m;
    }

    static mat3 first_order_tri_quad(vec3 const& a, mat3 const& sigma)
    {
        mat3 M;

        auto const xx = a[0] * a[0];
        auto const xy = a[0] * a[1];
        auto const xz = a[0] * a[2];
        auto const yy = a[1] * a[1];
        auto const yz = a[1] * a[2];
        auto const zz = a[2] * a[2];

        auto const two = scalar_t(2);

        math::get(M, 0, 0) = -math::get(sigma, 1, 1) * zz + two * math::get(sigma, 1, 2) * yz - math::get(sigma, 2, 2) * yy;
        math::get(M, 0, 1) = math::get(sigma, 0, 1) * zz - math::get(sigma, 0, 2) * yz - math::get(sigma, 1, 2) * xz + math::get(sigma, 2, 2) * xy;
        math::get(M, 0, 2) = -math::get(sigma, 0, 1) * yz + math::get(sigma, 0, 2) * yy + math::get(sigma, 1, 1) * xz - math::get(sigma, 1, 2) * xy;
        math::get(M, 1, 1) = -math::get(sigma, 0, 0) * zz + two * math::get(sigma, 0, 2) * xz - math::get(sigma, 2, 2) * xx;
        math::get(M, 1, 2) = math::get(sigma, 0, 0) * yz - math::get(sigma, 0, 1) * xz - math::get(sigma, 0, 2) * xy + math::get(sigma, 1, 2) * xx;
        math::get(M, 2, 2) = -math::get(sigma, 0, 0) * yy + two * math::get(sigma, 0, 1) * xy - math::get(sigma, 1, 1) * xx;

        math::get(M, 1, 0) = math::get(M, 0, 1);
        math::get(M, 2, 0) = math::get(M, 0, 2);
        math::get(M, 2, 1) = math::get(M, 1, 2);

        return M;
    }
};

// ============== Quadric Class ==============

template <class MathT>
struct quadric
{
    using math = MathT;
    using scalar_t = typename MathT::scalar_t;
    using vec3 = typename MathT::vec3;
    using pos3 = typename MathT::pos3;
    using mat3 = typename MathT::mat3;

    // x^T A x - 2 b^T x + c
public:
    scalar_t A00 = scalar_t(0);
    scalar_t A01 = scalar_t(0);
    scalar_t A02 = scalar_t(0);
    scalar_t A11 = scalar_t(0);
    scalar_t A12 = scalar_t(0);
    scalar_t A22 = scalar_t(0);

    scalar_t b0 = scalar_t(0);
    scalar_t b1 = scalar_t(0);
    scalar_t b2 = scalar_t(0);

    scalar_t c = scalar_t(0);

    // constructors and factories
public:
    constexpr quadric() = default;

    [[nodiscard]] static quadric from_coefficients(
        scalar_t A00, scalar_t A01, scalar_t A02, scalar_t A11, scalar_t A12, scalar_t A22, scalar_t b0, scalar_t b1, scalar_t b2, scalar_t c)
    {
        quadric q;
        q.A00 = A00;
        q.A01 = A01;
        q.A02 = A02;
        q.A11 = A11;
        q.A12 = A12;
        q.A22 = A22;
        q.b0 = b0;
        q.b1 = b1;
        q.b2 = b2;
        q.c = c;
        return q;
    }
    [[nodiscard]] static quadric from_coefficients(mat3 const& A, vec3 const& b, scalar_t c)
    {
        quadric q;
        q.A00 = math::get(A, 0, 0);
        q.A01 = math::get(A, 0, 1);
        q.A02 = math::get(A, 0, 2);
        q.A11 = math::get(A, 1, 1);
        q.A12 = math::get(A, 1, 2);
        q.A22 = math::get(A, 2, 2);
        q.b0 = b[0];
        q.b1 = b[1];
        q.b2 = b[2];
        q.c = c;
        return q;
    }

    [[nodiscard]] static quadric point_quadric(pos3 const& p)
    {
        auto const v = math::to_vec(p);
        return quadric::from_coefficients(math::make_identity(), v, math::dot(v, v));
    }

    [[nodiscard]] static quadric plane_quadric(pos3 const& p, vec3 const& n)
    {
        auto const d = math::dot(p, n);
        return quadric::from_coefficients(math::self_outer_product(n), n * d, d * d);
    }

    [[nodiscard]] static quadric probabilistic_plane_quadric(pos3 const& mean_p, vec3 const& mean_n, scalar_t stddev_p, scalar_t stddev_n)
    {
        auto const p = math::to_vec(mean_p);
        auto const sn2 = stddev_n * stddev_n;
        auto const sp2 = stddev_p * stddev_p;
        auto const d = math::dot(mean_p, mean_n);

        auto A = math::self_outer_product(mean_n);
        math::get(A, 0, 0) += sn2;
        math::get(A, 1, 1) += sn2;
        math::get(A, 2, 2) += sn2;

        auto const b = mean_n * d + p * sn2;
        auto const c = d * d + sn2 * math::dot(p, p) + sp2 * math::dot(mean_n, mean_n) + 3 * sp2 * sn2;

        return quadric::from_coefficients(A, b, c);
    }

    [[nodiscard]] static quadric probabilistic_plane_quadric(pos3 const& mean_p, vec3 const& mean_n, mat3 const& sigma_p, mat3 const& sigma_n)
    {
        auto const p = math::to_vec(mean_p);
        auto const d = math::dot(mean_p, mean_n);

        auto const A = math::self_outer_product(mean_n) + sigma_n;
        auto const b = mean_n * d + sigma_n * p;
        auto const c = d * d + math::dot(p, sigma_n * p) + math::dot(mean_n, sigma_p * mean_n) + math::trace_of_product(sigma_n, sigma_p);

        return quadric::from_coefficients(A, b, c);
    }

    [[nodiscard]] static quadric triangle_quadric(pos3 const& p, pos3 const& q, pos3 const& r)
    {
        auto const vp = math::to_vec(p);
        auto const vq = math::to_vec(q);
        auto const vr = math::to_vec(r);

        auto const pxq = math::cross(vp, vq);
        auto const qxr = math::cross(vq, vr);
        auto const rxp = math::cross(vr, vp);

        auto const xsum = pxq + qxr + rxp;
        auto const det = math::dot(pxq, vr);

        return quadric::from_coefficients(math::self_outer_product(xsum), xsum * det, det * det);
    }

    [[nodiscard]] static quadric probabilistic_triangle_quadric(pos3 const& mean_p, //
                                                                pos3 const& mean_q,
                                                                pos3 const& mean_r,
                                                                scalar_t stddev)
    {
        auto const sigma = stddev * stddev;
        auto const p = math::to_vec(mean_p);
        auto const q = math::to_vec(mean_q);
        auto const r = math::to_vec(mean_r);

        auto const pxq = math::cross(p, q);
        auto const qxr = math::cross(q, r);
        auto const rxp = math::cross(r, p);

        auto const det_pqr = math::dot(pxq, r);

        auto const cross_pqr = pxq + qxr + rxp;

        auto const pmq = p - q;
        auto const qmr = q - r;
        auto const rmp = r - p;

        mat3 A = math::self_outer_product(cross_pqr) +         //
                 (math::cross_product_squared_transpose(pmq) + //
                  math::cross_product_squared_transpose(qmr) + //
                  math::cross_product_squared_transpose(rmp))
                     * sigma;

        auto ss = sigma * sigma;
        auto ss6 = 6 * ss;
        auto ss2 = 2 * ss;
        math::get(A, 0, 0) += ss6;
        math::get(A, 1, 1) += ss6;
        math::get(A, 2, 2) += ss6;

        vec3 b = cross_pqr * det_pqr;

        b = b - (math::cross(pmq, pxq) + math::cross(qmr, qxr) + math::cross(rmp, rxp)) * sigma;

        b = b + (p + q + r) * ss2;

        scalar_t c = det_pqr * det_pqr;

        c += sigma * (math::dot(pxq, pxq) + math::dot(qxr, qxr) + math::dot(rxp, rxp)); // 3x (a x b)^T M_c (a x b)

        c += ss2 * (math::dot(p, p) + math::dot(q, q) + math::dot(r, r)); // 3x a^T Ci[S_b, S_c] a

        c += ss6 * sigma; // Tr[S_r Ci[S_p, S_q]]

        return quadric::from_coefficients(A, b, c);
    }

    [[nodiscard]] static quadric probabilistic_triangle_quadric(pos3 const& mean_p, //
                                                                pos3 const& mean_q,
                                                                pos3 const& mean_r,
                                                                mat3 const& sigma_p,
                                                                mat3 const& sigma_q,
                                                                mat3 const& sigma_r)
    {
        auto const p = math::to_vec(mean_p);
        auto const q = math::to_vec(mean_q);
        auto const r = math::to_vec(mean_r);

        auto const pxq = math::cross(p, q);
        auto const qxr = math::cross(q, r);
        auto const rxp = math::cross(r, p);

        auto const det_pqr = math::dot(pxq, r);

        auto const cross_pqr = pxq + qxr + rxp;

        auto const pmq = p - q;
        auto const qmr = q - r;
        auto const rmp = r - p;

        auto const ci_pq = math::cross_interference_matrix(sigma_p, sigma_q);
        auto const ci_qr = math::cross_interference_matrix(sigma_q, sigma_r);
        auto const ci_rp = math::cross_interference_matrix(sigma_r, sigma_p);

        mat3 A = math::self_outer_product(cross_pqr);

        A = A - math::first_order_tri_quad(pmq, sigma_r);
        A = A - math::first_order_tri_quad(qmr, sigma_p);
        A = A - math::first_order_tri_quad(rmp, sigma_q);

        A = A + ci_pq + ci_qr + ci_rp;

        math::get(A, 1, 0) = math::get(A, 0, 1);
        math::get(A, 2, 0) = math::get(A, 0, 2);
        math::get(A, 2, 1) = math::get(A, 1, 2);

        vec3 b = cross_pqr * det_pqr;

        b = b - math::cross(pmq, sigma_r * pxq);
        b = b - math::cross(qmr, sigma_p * qxr);
        b = b - math::cross(rmp, sigma_q * rxp);

        b = b + ci_pq * r;
        b = b + ci_qr * p;
        b = b + ci_rp * q;

        scalar_t c = det_pqr * det_pqr;

        c += math::dot(pxq, sigma_r * pxq);
        c += math::dot(qxr, sigma_p * qxr);
        c += math::dot(rxp, sigma_q * rxp);

        c += math::dot(p, ci_qr * p);
        c += math::dot(q, ci_rp * q);
        c += math::dot(r, ci_pq * r);

        c += math::trace_of_product(sigma_r, ci_pq);

        return quadric::from_coefficients(A, b, c);
    }

    // aggregate getters
public:
    constexpr mat3 const A() const
    {
        mat3 m;
        math::get(m, 0, 0) = A00;
        math::get(m, 0, 1) = A01;
        math::get(m, 0, 2) = A02;
        math::get(m, 1, 0) = A01;
        math::get(m, 1, 1) = A11;
        math::get(m, 1, 2) = A12;
        math::get(m, 2, 0) = A02;
        math::get(m, 2, 1) = A12;
        math::get(m, 2, 2) = A22;
        return m;
    }
    constexpr vec3 const b() const { return math::make_vec(b0, b1, b2); }

    // functions
public:
    [[nodiscard]] pos3 minimizer() const
    {
        // Returns a point minimizing this quadric
        // Solving Ax = r with some common subexpressions precomputed

        auto a = A00;
        auto b = A01;
        auto c = A02;
        auto d = A11;
        auto e = A12;
        auto f = A22;
        auto r0 = b0;
        auto r1 = b1;
        auto r2 = b2;

        auto ad = a * d;
        auto ae = a * e;
        auto af = a * f;
        auto bc = b * c;
        auto be = b * e;
        auto bf = b * f;
        auto df = d * f;
        auto ce = c * e;
        auto cd = c * d;

        auto be_cd = be - cd;
        auto bc_ae = bc - ae;
        auto ce_bf = ce - bf;

        auto denom = scalar_t(1) / (a * df + scalar_t(2) * b * ce - ae * e - bf * b - cd * c);
        auto nom0 = r0 * (df - e * e) + r1 * ce_bf + r2 * be_cd;
        auto nom1 = r0 * ce_bf + r1 * (af - c * c) + r2 * bc_ae;
        auto nom2 = r0 * be_cd + r1 * bc_ae + r2 * (ad - b * b);

        return math::make_pos(nom0 * denom, nom1 * denom, nom2 * denom);
    }

    // operators
public:
    /// Residual L2 error as given by x^T A x - 2 r^T x + c
    scalar_t operator()(pos3 const& p) const
    {
        vec3 Ax = math::make_vec(A00 * p[0] + A01 * p[1] + A02 * p[2], //
                                 A01 * p[0] + A11 * p[1] + A12 * p[2], //
                                 A02 * p[0] + A12 * p[1] + A22 * p[2]);

        return math::dot(p, Ax)                                    // x^T A x
               - scalar_t(2) * (p[0] * b0 + p[1] * b1 + p[2] * b2) // - 2 r^T x
               + c;                                                // + c
    }

    scalar_t operator()(scalar_t x, scalar_t y, scalar_t z) const { return operator()(math::make_pos(x, y, z)); }

    quadric& operator+=(quadric const& rhs)
    {
        A00 += rhs.A00;
        A01 += rhs.A01;
        A02 += rhs.A02;
        A11 += rhs.A11;
        A12 += rhs.A12;
        A22 += rhs.A22;

        b0 += rhs.b0;
        b1 += rhs.b1;
        b2 += rhs.b2;

        c += rhs.c;

        return *this;
    }

    quadric& operator-=(quadric const& rhs)
    {
        A00 -= rhs.A00;
        A01 -= rhs.A01;
        A02 -= rhs.A02;
        A11 -= rhs.A11;
        A12 -= rhs.A12;
        A22 -= rhs.A22;

        b0 -= rhs.b0;
        b1 -= rhs.b1;
        b2 -= rhs.b2;

        c -= rhs.c;

        return *this;
    }

    quadric& operator*=(scalar_t const& s)
    {
        A00 *= s;
        A01 *= s;
        A02 *= s;
        A11 *= s;
        A12 *= s;
        A22 *= s;

        b0 *= s;
        b1 *= s;
        b2 *= s;

        c *= s;

        return *this;
    }

    quadric operator+() const { return *this; }

    quadric operator-() const
    {
        quadric q;
        q.A00 = -A00;
        q.A01 = -A01;
        q.A02 = -A02;
        q.A11 = -A11;
        q.A12 = -A12;
        q.A22 = -A22;

        q.b0 = -b0;
        q.b1 = -b1;
        q.b2 = -b2;

        q.c = -c;
        return q;
    }

    quadric& operator/=(scalar_t const& s) { return operator*=(scalar_t(1) / s); }

    quadric operator+(quadric const& b) const
    {
        auto r = *this; // copy
        r += b;
        return r;
    }

    quadric operator-(quadric const& b) const
    {
        auto r = *this; // copy
        r -= b;
        return r;
    }

    quadric operator*(scalar_t const& b) const
    {
        auto r = *this; // copy
        r *= b;
        return r;
    }

    quadric operator/(scalar_t const& b) const
    {
        auto r = *this; // copy
        r /= b;
        return r;
    }

    friend quadric operator*(scalar_t const& s, quadric const& q) { return q * s; }
};

} // namespace pq
