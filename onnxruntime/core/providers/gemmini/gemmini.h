// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"
#include "core/framework/tensor.h"
#include "core/providers/gemmini/gemmini_execution_provider.h"

#include "gemmini_runtime.h"

namespace onnxruntime {
using ConvPadVector = ConvAttributes::ConvPadVector;

namespace contrib {
namespace gemmini {


template <typename T>
class Conv : public onnxruntime::Conv<T> {
 public:
  explicit Conv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info), conv_attrs_(info) {
    gemmini_rt = new GemminiRT();
  }

  Status Compute(OpKernelContext* context) const override {
      std::cout << "gemmi::conv::compute" << std::endl;
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);  // optional. nullptr if not provided
  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

    TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  TensorShapeVector Y_dims({N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = context->Output(0, Y_dims);
  TensorShape output_shape = Y->Shape().Slice(2);


    // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  const size_t kernel_rank = kernel_shape.size();

  BufferUniquePtr col_buffer;

  // Pointwise convolutions can use the original input tensor in place,
  // otherwise a temporary buffer is required for the im2col transform.
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

    auto* col_data = alloc->Alloc(sizeof(T) * SafeInt<size_t>(col_buffer_size));
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(std::move(alloc)));
  }

  unsigned long sw_time, hw_time, mem_time;
  gemmini_rt->conv2d(input_image_size, kernel_size, sw_time, hw_time, mem_time);

      //Status s = onnxruntime::Conv<T>::Compute(context);
      //return s;
      return Status::OK();
  }

 private:
     ConvAttributes conv_attrs_;
     GemminiRT *gemmini_rt;
};
}  // namespace gemmini
}  // namespace contrib
}  // namespace onnxruntime
