#pragma once

#include <string>

namespace klft {

struct KokkosTuningParams {
  bool enabled;
  std::string cache_file;

  KokkosTuningParams() : enabled(false), cache_file("ktune_cache.dat") {}
};

} // namespace klft
