#include "render/ply_mesh_io.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

struct PLYProperty {
  bool is_list = false;
  std::string type;        // scalar type, if not a list
  std::string count_type;  // list only
  std::string item_type;   // list only
  std::string name;
};

struct PLYElement {
  std::string name;
  size_t count = 0;
  std::vector<PLYProperty> properties;
};

size_t TypeByteWidth(const std::string &type) {
  if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
  if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
  if (type == "int" || type == "uint" || type == "int32" || type == "uint32" ||
      type == "float" || type == "float32")
    return 4;
  if (type == "double" || type == "float64") return 8;
  return 0;
}

// Reads one binary scalar of the given PLY type and returns it as a double
// (adequate for coordinates/indices; this reader never needs the extra
// precision of a raw double for anything but double-typed fields anyway).
double ReadBinaryScalar(std::ifstream &f, const std::string &type) {
  unsigned char buf[8] = {0};
  size_t w = TypeByteWidth(type);
  f.read(reinterpret_cast<char *>(buf), (std::streamsize)w);
  if (type == "float" || type == "float32") {
    float v;
    std::memcpy(&v, buf, 4);
    return v;
  }
  if (type == "double" || type == "float64") {
    double v;
    std::memcpy(&v, buf, 8);
    return v;
  }
  if (type == "char" || type == "int8") {
    int8_t v;
    std::memcpy(&v, buf, 1);
    return v;
  }
  if (type == "uchar" || type == "uint8") {
    uint8_t v;
    std::memcpy(&v, buf, 1);
    return v;
  }
  if (type == "short" || type == "int16") {
    int16_t v;
    std::memcpy(&v, buf, 2);
    return v;
  }
  if (type == "ushort" || type == "uint16") {
    uint16_t v;
    std::memcpy(&v, buf, 2);
    return v;
  }
  if (type == "int" || type == "int32") {
    int32_t v;
    std::memcpy(&v, buf, 4);
    return v;
  }
  if (type == "uint" || type == "uint32") {
    uint32_t v;
    std::memcpy(&v, buf, 4);
    return v;
  }
  return 0;
}

void SkipElementInstance(std::ifstream &f, bool binary, const PLYElement &elem) {
  if (binary) {
    for (const auto &p : elem.properties) {
      if (p.is_list) {
        int n = (int)ReadBinaryScalar(f, p.count_type);
        for (int k = 0; k < n; ++k) ReadBinaryScalar(f, p.item_type);
      } else {
        ReadBinaryScalar(f, p.type);
      }
    }
  } else {
    std::string line;
    std::getline(f, line);
  }
}

}  // namespace

bool ReadPLYMesh(const std::string &path, Mesh &mesh, std::string *error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return false;
  }

  std::string line;
  if (!std::getline(f, line) || line.substr(0, 3) != "ply") {
    if (error) *error = path + " is not a PLY file";
    return false;
  }

  std::string format;
  std::vector<PLYElement> elements;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    std::string tok;
    iss >> tok;
    if (tok == "format") {
      iss >> format;
    } else if (tok == "comment" || tok.empty()) {
      continue;
    } else if (tok == "element") {
      PLYElement e;
      iss >> e.name >> e.count;
      elements.push_back(e);
    } else if (tok == "property") {
      std::string t;
      iss >> t;
      PLYProperty p;
      if (t == "list") {
        p.is_list = true;
        iss >> p.count_type >> p.item_type >> p.name;
      } else {
        p.type = t;
        iss >> p.name;
      }
      if (elements.empty()) {
        if (error) *error = path + ": property before any element";
        return false;
      }
      elements.back().properties.push_back(p);
    } else if (tok == "end_header") {
      break;
    }
  }

  bool binary = format == "binary_little_endian";
  if (!binary && format != "ascii") {
    if (error) *error = path + ": unsupported PLY format '" + format + "'";
    return false;
  }

  mesh.vertices.clear();
  mesh.faces.clear();

  for (const auto &elem : elements) {
    if (elem.name == "vertex") {
      int xi = -1, yi = -1, zi = -1;
      for (size_t i = 0; i < elem.properties.size(); ++i) {
        const std::string &name = elem.properties[i].name;
        if (name == "x") xi = (int)i;
        else if (name == "y") yi = (int)i;
        else if (name == "z") zi = (int)i;
      }
      if (xi < 0 || yi < 0 || zi < 0) {
        if (error) *error = path + ": vertex element missing x/y/z";
        return false;
      }
      mesh.vertices.reserve(elem.count);
      for (size_t v = 0; v < elem.count; ++v) {
        std::vector<double> values(elem.properties.size(), 0.0);
        if (binary) {
          for (size_t p = 0; p < elem.properties.size(); ++p)
            values[p] = ReadBinaryScalar(f, elem.properties[p].type);
        } else {
          std::getline(f, line);
          std::istringstream iss(line);
          for (size_t p = 0; p < elem.properties.size(); ++p) iss >> values[p];
        }
        mesh.vertices.emplace_back((float)values[xi], (float)values[yi], (float)values[zi]);
      }
    } else if (elem.name == "face") {
      if (elem.properties.empty() || !elem.properties[0].is_list) {
        if (error) *error = path + ": face element missing an index list property";
        return false;
      }
      const PLYProperty &list_prop = elem.properties[0];
      mesh.faces.reserve(elem.count);
      for (size_t fidx = 0; fidx < elem.count; ++fidx) {
        int n = 0;
        std::vector<int> idx;
        if (binary) {
          n = (int)ReadBinaryScalar(f, list_prop.count_type);
          idx.resize(n);
          for (int k = 0; k < n; ++k) idx[k] = (int)ReadBinaryScalar(f, list_prop.item_type);
        } else {
          std::getline(f, line);
          std::istringstream iss(line);
          iss >> n;
          idx.resize(n);
          for (int k = 0; k < n; ++k) iss >> idx[k];
        }
        for (int k = 1; k + 1 < n; ++k) mesh.faces.emplace_back(idx[0], idx[k], idx[k + 1]);
      }
    } else {
      for (size_t i = 0; i < elem.count; ++i) SkipElementInstance(f, binary, elem);
    }
  }

  ComputeFaceNormals(mesh);
  return true;
}

void WritePLYMesh(const std::string &path, const std::vector<Eigen::Vector3f> &vertices,
                   const std::vector<Eigen::Vector3i> &faces,
                   const std::vector<Rgb8> &vertex_colors) {
  bool has_color = vertex_colors.size() == vertices.size();
  std::ofstream f(path);
  f << "ply\nformat ascii 1.0\n";
  f << "element vertex " << vertices.size() << "\n";
  f << "property float x\nproperty float y\nproperty float z\n";
  if (has_color) f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  f << "element face " << faces.size() << "\n";
  f << "property list uchar int vertex_indices\n";
  f << "end_header\n";
  for (size_t i = 0; i < vertices.size(); ++i) {
    f << vertices[i].x() << " " << vertices[i].y() << " " << vertices[i].z();
    if (has_color)
      f << " " << (int)vertex_colors[i].r << " " << (int)vertex_colors[i].g << " "
        << (int)vertex_colors[i].b;
    f << "\n";
  }
  for (const auto &face : faces) f << "3 " << face[0] << " " << face[1] << " " << face[2] << "\n";
}
