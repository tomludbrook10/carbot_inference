find_path(TensorRT_INCLUDE_DIR
NAMES NvInfer.h
  HINTS
    ${TensorRT_DIR}/include
    /usr/include/aarch64-linux-gnu/
    /usr/include/
)

find_library(TensorRT_NVINFER_LIBRARY
NAMES nvinfer
  HINTS
    ${TensorRT_DIR}/lib
    /usr/lib/
    /usr/lib/aarch64-linux-gnu/
)

include(FindPackageHandleStandardArgs)
## sets TensorRT_FOUND to TRUE if all REQUIRED_VARS are found.
find_package_handle_standard_args(TensorRT 
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY)


if(TensorRT_FOUND AND NOT TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
endif()
