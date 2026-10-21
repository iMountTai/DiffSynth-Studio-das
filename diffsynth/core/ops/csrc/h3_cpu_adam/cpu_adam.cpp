// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team
// Derived from DeepSpeed v0.18.4.
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update_fused", &ds_adam_fused, "FP32 Adam with gradient scaling and BF16 output");
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    m.def("adam_rollback", &ds_adam_rollback, "DeepSpeed CPU Adam rollback (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");
}
