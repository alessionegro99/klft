#pragma once

#include "core/common.hpp"

#include <vector>

namespace klft {

struct MultilevelLevelParams {
  index_t slab_links;
  index_t updates;

  MultilevelLevelParams() : slab_links(2), updates(10) {}
  MultilevelLevelParams(const index_t slab_links, const index_t updates)
      : slab_links(slab_links), updates(updates) {}
};

struct MultilevelParams {
  std::vector<MultilevelLevelParams> levels;
};

} // namespace klft
