// Copyright (c) 2025 deankh. All rights reserved.
#pragma once
#include <memory>

#define DKLIB_CLASS_PTRS(classname) \
  using Ptr = std::shared_ptr<classname>; \
  using ConstPtr = std::shared_ptr<const classname>; \
  using WeakPtr = std::weak_ptr<classname>; \
  using ConstWeakPtr = std::weak_ptr<const classname>;
