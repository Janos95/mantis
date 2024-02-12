#pragma once

#include <vector>
#include <array>
#include <functional>
#include <cassert>
#include <cmath>
#include <cfloat>

template<typename T>
struct TVec3 {
    T x = 0, y = 0, z = 0;

    TVec3() = default;

    TVec3(T v) : x(v), y(v), z(v) {}

    TVec3(T x, T y, T z) : x(x), y(y), z(z) {}

    TVec3 operator+(const TVec3 &other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    TVec3 operator-(const TVec3 &other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    TVec3 operator*(T scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    TVec3 operator*(const TVec3 &other) const {
        return {x * other.x, y * other.y, z * other.z};
    }

    TVec3 operator/(T scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    T &operator[](int index) {
        assert(index >= 0 && index < 3);
        return (&x)[index];
    }

    const T &operator[](int index) const {
        assert(index >= 0 && index < 3);
        return (&x)[index];
    }

    inline TVec3<T>& operator+=(const TVec3<T> &vec) {
        return *this = *this + vec;
    }

    TVec3 normalized() const {
        return *this / norm();
    }

    TVec3& normalize() {
        *this = normalized();
        return *this;
    }

    T dot(const TVec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    T norm() const {
        return std::sqrt(dot(*this));
    }

    T squaredNorm() const {
        return dot(*this);
    }

    TVec3 cross(const TVec3& other) const {
        return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }

    template<class U>
    TVec3<U> cast() const {
        return {static_cast<U>(x), static_cast<U>(y), static_cast<U>(z)};
    }
};

using Vec3 = TVec3<double>;
using Vec3f = TVec3<float>;

struct Vec4 {
    double x = 0, y = 0, z = 0, w = 0;

    Vec4() = default;

    Vec4(double v) : x(v), y(v), z(v), w(v) {}

    Vec4(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {}

    Vec4 operator+(const Vec4 &other) const {
        return {x + other.x, y + other.y, z + other.z, w + other.w};
    }

    Vec4 operator-(const Vec4 &other) const {
        return {x - other.x, y - other.y, z - other.z, w - other.w};
    }

    Vec4 operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar, w * scalar};
    }

    Vec4 operator*(const Vec4 &other) const {
        return {x * other.x, y * other.y, z * other.z, w * other.w};
    }

    Vec4 operator/(double scalar) const {
        return {x / scalar, y / scalar, z / scalar, w / scalar};
    }

    double &operator[](int index) {
        assert(index >= 0 && index < 4);
        return (&x)[index];
    }

    const double &operator[](int index) const {
        assert(index >= 0 && index < 4);
        return (&x)[index];
    }

    double dot(const Vec4& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }
};

struct Vec4f {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;

    Vec4f() = default;

    explicit Vec4f(const Vec4& other) : x(other.x), y(other.y), z(other.z), w(other.w) {}

    explicit Vec4f(float v) : x(v), y(v), z(v), w(v) {}

    Vec4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    Vec4f operator+(const Vec4f &other) const {
        return {x + other.x, y + other.y, z + other.z, w + other.w};
    }

    Vec4f operator-(const Vec4f &other) const {
        return {x - other.x, y - other.y, z - other.z, w - other.w};
    }

    Vec4f operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar, w * scalar};
    }

    Vec4f operator*(const Vec4f &other) const {
        return {x * other.x, y * other.y, z * other.z, w * other.w};
    }

    Vec4f operator/(float scalar) const {
        return {x / scalar, y / scalar, z / scalar, w / scalar};
    }

    float &operator[](int index) {
        assert(index >= 0 && index < 4);
        return (&x)[index];
    }

    const float &operator[](int index) const {
        assert(index >= 0 && index < 4);
        return (&x)[index];
    }

    float dot(const Vec4f& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }
};

struct Vec3i {
    int x = 0, y = 0, z = 0;

    Vec3i() = default;

    Vec3i(int x, int y, int z) : x(x), y(y), z(z) {}

    int &operator[](int i) {
        assert(i >= 0 && i < 3);
        return (&x)[i];
    }

    const int &operator[](int i) const {
        assert(i >= 0 && i < 3);
        return (&x)[i];
    }

    Vec3i operator+(const Vec3i &other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vec3i operator-(const Vec3i &other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vec3i operator*(const Vec3i &other) const {
        return {x * other.x, y * other.y, z * other.z};
    }

    Vec3i operator/(int s) const {
        return {x / s, y / s, z / s};
    }

    bool operator==(const Vec3i &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct Mat3 {
    Vec3 col0, col1, col2;

    Mat3 operator+(const Mat3 &other) const {
        return {col0 + other.col0, col1 + other.col1, col2 + other.col2};
    }

    Mat3 operator-(const Mat3 &other) const {
        return {col0 - other.col0, col1 - other.col1, col2 - other.col2};
    }

    Mat3 operator*(double scalar) const {
        return {col0 * scalar, col1 * scalar, col2 * scalar};
    }

    Vec3 operator*(const Vec3 &vec) const {
        return {
                col0.x * vec.x + col1.x * vec.y + col2.x * vec.z,
                col0.y * vec.x + col1.y * vec.y + col2.y * vec.z,
                col0.z * vec.x + col1.z * vec.y + col2.z * vec.z
        };
    }

    Vec3 &operator[](int idx) {
        return (&col0)[idx];
    }

    const Vec3 &operator[](int idx) const {
        return (&col0)[idx];
    }
};

inline Mat3 operator*(double scalar, const Mat3 &mat) {
    return mat * scalar;
}

inline Vec4 operator-(const Vec4& v) {
    return {-v.x, -v.y, -v.z, -v.w};
}

inline Vec4f operator-(const Vec4f& v) {
    return {-v.x, -v.y, -v.z, -v.w};
}

template<class T>
inline TVec3<T> operator-(const TVec3<T> &vec) {
    return {-vec.x, -vec.y, -vec.z};
}

template<typename T>
inline TVec3<T> operator*(T scalar, const TVec3<T> &vec) {
    return vec * scalar;
}

template<typename T>
inline TVec3<T> operator/(T scalar, const TVec3<T> &vec) {
    return vec / scalar;
}

template<typename T>
inline TVec3<T> abs(const TVec3<T> &v) {
    return {std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)};
}

template<typename T>
inline TVec3<T> max(const TVec3<T> &v1, const TVec3<T> &v2) {
    return {std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z)};
}

template<typename T>
inline T length(const TVec3<T> &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

template<typename T>
inline T length2(const TVec3<T> &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

template<typename T>
inline TVec3<T> normalize(const TVec3<T> &v) {
    return v / length(v);
}

struct QuadMesh {
    std::vector<Vec3> vertices;
    std::vector<std::array<int, 4>> quads;
};

// generate a mesh from an implicit function f with n^3 grid points
QuadMesh generate_mesh(std::function<double(Vec3)> f, int n = 50);

