// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/gemmini/gemmini_options.h"
#include "core/framework/execution_provider.h"
//#include "core/framework/model_metadef_id_generator.h"

namespace onnxruntime {

class GEMMINIExecutionProvider : public IExecutionProvider {
 public:
  GEMMINIExecutionProvider(const GEMMINIOptions& options);
  virtual ~GEMMINIExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                IResourceAccountant* resource_accountant) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  GEMMINIOptions GetOptions() const { return coreml_options_; }
 private:
  // The bit flags which define bool options for COREML EP, bits are defined as
  // COREMLFlags in include/onnxruntime/core/providers/coreml/coreml_provider_factory.h
  GEMMINIOptions coreml_options_;
  const int32_t coreml_version_;
  //ModelMetadefIdGenerator metadef_id_generator_;

  // map of fused_node_name to compiled_coreml_model
  //InlinedHashMap<std::string, std::unique_ptr<onnxruntime::coreml::Model>> coreml_models_;
};
}  // namespace onnxruntime
