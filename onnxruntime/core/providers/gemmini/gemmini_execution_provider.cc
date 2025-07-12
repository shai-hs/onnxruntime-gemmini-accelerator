// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/gemmini/gemmini_execution_provider.h"
#include "core/providers/gemmini/gemmini_provider_factory.h"  // defines flags

#include <algorithm>

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
//#include "core/providers/gemmini/builders/helper.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

//#include "core/providers/gemmini/builders/model_builder.h"
//#include "core/providers/gemmini/model/host_utils.h"
//#include "core/providers/gemmini/model/model.h"
//#include "core/providers/gemmini/shape_utils.h"

#include "core/framework/op_kernel.h"
#include "core/graph/constants.h"

#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/kernel_registry.h"

#include "gemmini.h"

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};
}  // namespace

namespace onnxruntime {

//constexpr const char* GEMMINI = "GEMMINI";

namespace contrib {
namespace gemmini {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kGEMMINIExecutionProvider, kOnnxDomain, 1, Conv);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterGemminiContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kGEMMINIExecutionProvider, kOnnxDomain, 1, Conv)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

}  // namespace gemmini
}  // namespace contrib

KernelRegistryAndStatus GetGemminiKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = ::onnxruntime::contrib::gemmini::RegisterGemminiContribKernels(*ret.kernel_registry);
  return ret;
}

std::shared_ptr<KernelRegistry> GEMMINIExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus ret = GetGemminiKernelRegistry();
  // throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(ret.st);
  return ret.kernel_registry;
}

GEMMINIExecutionProvider::GEMMINIExecutionProvider(const GEMMINIOptions& options)
    : IExecutionProvider{onnxruntime::kGEMMINIExecutionProvider},
      coreml_options_(options),
      coreml_version_(/*coreml::util::CoreMLVersion()*/0) {
  LOGS_DEFAULT(VERBOSE) << "GEMMINI version: " << coreml_version_;
  //if (coreml_version_ < MINIMUM_COREML_VERSION) {
  //  ORT_THROW("GEMMINI EP is not supported on this platform.");
  //}
}

GEMMINIExecutionProvider::~GEMMINIExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
GEMMINIExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup,
                                       const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                       IResourceAccountant* /* resource_accountant */) const {

  std::vector<NodeIndex> candidates;
  for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph_viewer.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      continue;
    }

    const KernelCreateInfo* gemmini_kernel_def = kernel_lookup.LookUpKernel(node);
    if (gemmini_kernel_def == nullptr) {
      LOGS_DEFAULT(WARNING) << "GEMMINI kernel not found in registries for Op type: " << node.OpType()
                            << " node name: " << node.Name();
      continue;
    }

    candidates.push_back(node.Index());
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  return result;
}

}  // namespace onnxruntime
