#include "render/mesh.hpp"

#include <limits>

#include <Eigen/Geometry>  // cross_impl specialization used by .cross() below

void ComputeFaceNormals(Mesh &mesh) {
  mesh.face_normals.resize(mesh.faces.size());
  for (size_t i = 0; i < mesh.faces.size(); ++i) {
    const Eigen::Vector3i &f = mesh.faces[i];
    const Eigen::Vector3f &v0 = mesh.vertices[f[0]];
    const Eigen::Vector3f &v1 = mesh.vertices[f[1]];
    const Eigen::Vector3f &v2 = mesh.vertices[f[2]];
    Eigen::Vector3f e1 = v1 - v0;
    Eigen::Vector3f e2 = v2 - v0;
    Eigen::Vector3f n = e1.cross(e2);
    float len = n.norm();
    mesh.face_normals[i] = len > 1e-12f ? Eigen::Vector3f(n / len) : Eigen::Vector3f::Zero();
  }
}

Aabb ComputeAabb(const Mesh &mesh) {
  Aabb box;
  box.min.setConstant(std::numeric_limits<float>::max());
  box.max.setConstant(std::numeric_limits<float>::lowest());
  for (const auto &v : mesh.vertices) {
    box.min = box.min.cwiseMin(v);
    box.max = box.max.cwiseMax(v);
  }
  return box;
}
