#pragma once

#include <string>

namespace klft {

struct PartitioningParams {
  bool enabled = false;
  std::string table_file;
};

} // namespace klft
