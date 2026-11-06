#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "render/mesh.hpp"

struct Rgb8 {
  unsigned char r, g, b;
};

// Reads a PLY mesh (vertex x,y,z + triangle/polygon faces). Supports both
// "format ascii 1.0" and "format binary_little_endian 1.0" bodies (the
// Stanford 3D Scanning Repository ships the bunny as ASCII; binary support
// is included for other real-world PLY distributions). Any vertex/face
// properties beyond x,y,z and the face index list (confidence, intensity,
// normals, color, ...) are skipped. Polygons with more than 3 vertices are
// fan-triangulated. Computes face normals on success.
bool ReadPLYMesh(const std::string &path, Mesh &mesh, std::string *error = nullptr);

// Minimal ASCII PLY writer: vertices (+ optional per-vertex color) and
// triangle faces. Companion to tests/report_utils.hpp's point-only WritePLY.
void WritePLYMesh(const std::string &path,
                   const std::vector<Eigen::Vector3f> &vertices,
                   const std::vector<Eigen::Vector3i> &faces,
                   const std::vector<Rgb8> &vertex_colors = {});
