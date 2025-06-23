#pragma once
#include <cstdint>

namespace dklib::dnn {

enum class InferenceDevice : uint32_t {
  kCPU = 0,
  kCUDA,
  kNum,
};

}