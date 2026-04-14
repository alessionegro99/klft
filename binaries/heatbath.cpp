#include "io/driver_utils.hpp"
#include "klft.hpp"

#include <Kokkos_Core.hpp>

using namespace klft;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  std::string input_file;
  rc = parse_driver_args(argc, argv, input_file);
  if (rc == 0) {
    rc = Heatbath(input_file);
  } else if (rc == 1) {
    rc = write_sample_heatbath_input_file(input_file);
  } else if (rc == -2) {
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
