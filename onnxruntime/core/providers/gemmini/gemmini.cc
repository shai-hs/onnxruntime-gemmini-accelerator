// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/gemmini/gemmini.h"

namespace onnxruntime {
namespace contrib {
namespace gemmini {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    kGEMMINIExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

}  // namespace gemmini
}  // namespace contrib
}  // namespace onnxruntime
