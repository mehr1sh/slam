#pragma once

#include <Eigen/Core>
#include <vector>

struct Mesh {
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> faces;         // indices into vertices
  std::vector<Eigen::Vector3f> face_normals;  // parallel to faces; computed, not read from file
};

// normal = normalize((v1-v0) x (v2-v0)) per face.
void ComputeFaceNormals(Mesh &mesh);

struct Aabb {
  Eigen::Vector3f min, max;
  Eigen::Vector3f Center() const { return 0.5f * (min + max); }
  float BoundingSphereRadius() const { return 0.5f * (max - min).norm(); }
};

Aabb ComputeAabb(const Mesh &mesh);
