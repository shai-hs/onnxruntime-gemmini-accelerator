// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/gemmini/gemmini_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "gemmini_execution_provider.h"
#include "gemmini_provider_factory_creator.h"

using namespace onnxruntime;

namespace onnxruntime {

struct GEMMINIProviderFactory : IExecutionProviderFactory {
  GEMMINIProviderFactory(const GEMMINIOptions& options)
      : options_(options) {}
  ~GEMMINIProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  GEMMINIOptions options_;
};

std::unique_ptr<IExecutionProvider> GEMMINIProviderFactory::CreateProvider() {
  return std::make_unique<GEMMINIExecutionProvider>(options_);
}

std::shared_ptr<IExecutionProviderFactory> GEMMINIProviderFactoryCreator::Create(uint32_t coreml_flags) {
  GEMMINIOptions coreml_options(coreml_flags);
  return std::make_shared<onnxruntime::GEMMINIProviderFactory>(coreml_options);
}

std::shared_ptr<IExecutionProviderFactory> GEMMINIProviderFactoryCreator::Create(const ProviderOptions& options) {
  GEMMINIOptions coreml_options(options);
  return std::make_shared<onnxruntime::GEMMINIProviderFactory>(coreml_options);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_GEMMINI,
                    _In_ OrtSessionOptions* options, uint32_t coreml_flags) {
  options->provider_factories.push_back(onnxruntime::GEMMINIProviderFactoryCreator::Create(coreml_flags));
  return nullptr;
}
