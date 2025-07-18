# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import os
import platform
import sys
import unittest

import numpy as np
from autoep_helper import AutoEpTestCase
from helper import get_name

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())

available_providers = list(onnxrt.get_available_providers())


class TestAutoEP(AutoEpTestCase):
    def test_cuda_ep_register_and_inference(self):
        """
        Test registration of CUDA EP, adding its OrtDevice to the SessionOptions, and running inference.
        """
        ep_lib_path = "onnxruntime_providers_cuda.dll"
        ep_registration_name = "CUDAExecutionProvider"

        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        if ep_registration_name not in available_providers:
            self.skipTest("Skipping test because it needs to run on CUDA EP")

        self.register_execution_provider_library(ep_registration_name, ep_lib_path)

        ep_devices = onnxrt.get_ep_devices()
        has_cpu_ep = False
        cuda_ep_device = None
        for ep_device in ep_devices:
            ep_name = ep_device.ep_name
            if ep_name == "CPUExecutionProvider":
                has_cpu_ep = True
            if ep_name == ep_registration_name:
                cuda_ep_device = ep_device

        self.assertTrue(has_cpu_ep)
        self.assertIsNotNone(cuda_ep_device)
        self.assertEqual(cuda_ep_device.ep_vendor, "Microsoft")

        hw_device = cuda_ep_device.device
        self.assertEqual(hw_device.type, onnxrt.OrtHardwareDeviceType.GPU)

        # Add CUDA's OrtEpDevice to session options
        sess_options = onnxrt.SessionOptions()
        sess_options.add_provider_for_devices([cuda_ep_device], {"prefer_nhwc": "1"})
        self.assertTrue(sess_options.has_providers())

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        # TODO(adrianlizarraga): Unregistering CUDA EP library causes issues. Investigate.
        # self.unregister_execution_provider_library(ep_registration_name)

    def test_cuda_prefer_gpu_and_inference(self):
        """
        Test selecting CUDA EP via the PREFER_GPU policy and running inference.
        """
        ep_lib_path = "onnxruntime_providers_cuda.dll"
        ep_registration_name = "CUDAExecutionProvider"

        if sys.platform != "win32":
            self.skipTest("Skipping test because device discovery is only supported on Windows")

        if ep_registration_name not in available_providers:
            self.skipTest("Skipping test because it needs to run on CUDA EP")

        self.register_execution_provider_library(ep_registration_name, ep_lib_path)

        # Set a policy to prefer GPU. Cuda should be selected.
        sess_options = onnxrt.SessionOptions()
        sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_GPU)
        self.assertTrue(sess_options.has_providers())

        # Run sample model and check output
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=sess_options)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        # TODO(adrianlizarraga): Unregistering CUDA EP library causes issues. Investigate.
        # self.unregister_execution_provider_library(ep_registration_name)

    def test_example_plugin_ep_devices(self):
        """
        Test registration of an example EP plugin and retrieval of its OrtEpDevice.
        """
        if sys.platform != "win32":
            self.skipTest("Skipping test because it device discovery is only supported on Windows")

        ep_lib_path = "example_plugin_ep.dll"
        try:
            ep_lib_path = get_name("example_plugin_ep.dll")
        except FileNotFoundError:
            self.skipTest(f"Skipping test because EP library '{ep_lib_path}' cannot be found")

        ep_registration_name = "example_ep"
        self.register_execution_provider_library(ep_registration_name, os.path.realpath(ep_lib_path))

        ep_devices = onnxrt.get_ep_devices()
        has_cpu_ep = False
        test_ep_device = None
        for ep_device in ep_devices:
            ep_name = ep_device.ep_name

            if ep_name == "CPUExecutionProvider":
                has_cpu_ep = True
            if ep_name == ep_registration_name:
                test_ep_device = ep_device

        self.assertTrue(has_cpu_ep)
        self.assertIsNotNone(test_ep_device)

        # Test the OrtEpDevice getters. Expected values are from /onnxruntime/test/autoep/library/example_plugin_ep.cc
        self.assertEqual(test_ep_device.ep_vendor, "Contoso")

        ep_metadata = test_ep_device.ep_metadata
        self.assertEqual(ep_metadata["version"], "0.1")

        ep_options = test_ep_device.ep_options
        self.assertEqual(ep_options["run_really_fast"], "true")

        # The CPU hw device info will vary by machine so check for the common values.
        hw_device = test_ep_device.device
        self.assertEqual(hw_device.type, onnxrt.OrtHardwareDeviceType.CPU)
        self.assertGreaterEqual(hw_device.vendor_id, 0)
        self.assertGreaterEqual(hw_device.device_id, 0)
        self.assertGreater(len(hw_device.vendor), 0)

        hw_metadata = hw_device.metadata
        self.assertGreater(len(hw_metadata), 0)  # Should have at least SPDRP_HARDWAREID on Windows

        # Test adding this EP plugin's OrtEpDevice to the SessionOptions.
        sess_options = onnxrt.SessionOptions()
        with self.assertRaises(InvalidArgument) as context:
            # Will raise InvalidArgument because ORT currently only supports provider bridge APIs.
            # Actual plugin EPs will be supported in the future.
            sess_options.add_provider_for_devices([test_ep_device], {"opt1": "val1"})
        self.assertIn("EP is not currently supported", str(context.exception))

        self.unregister_execution_provider_library(ep_registration_name)


if __name__ == "__main__":
    unittest.main(verbosity=1)
