Build the gemmini onnxruntime:

Dependencies:
  Clone and build gemmini-accelerator-driver-and-runtime
  https://github.com/shai-hs/Gemmini-accelerator-driver-and-runtime

  Provided that the gemmini-accelerator-driver-and-runtime is installed to
  the GEMMINI_DRV_RT directory.

  $ cd onnxruntime-gemmini
  $ CXX=g++ CC=gcc ./build.sh --use_gemmini --gemmini_root=$GEMMINI_DRV_RT --config Debug --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
