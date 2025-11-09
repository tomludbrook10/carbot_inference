cmake_minimum_required(VERSION 3.22)
project(carbot_inference)
set(CMAKE_CXX_STANDARD 17)

## add so that cmake can find the locally installed packages.
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")
## find modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Find GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
find_package(CUDAToolkit REQUIRED COMPONENTS Runtime)
find_package(TensorRT REQUIRED COMPONENTS nvinfer)
find_package(DeepStream REQUIRED COMPONENTS nvdsgst_meta nvds_meta nvdspreprocess)

add_library(carbot_inference SHARED 
    src/CudaManager.cpp
    src/InferPipelineManager.cpp
)

target_include_directories(carbot_inference
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GSTREAMER_INCLUDE_DIRS}
)

target_link_libraries(carbot_inference
    PUBLIC
    CUDA::cudart
    PRIVATE
    TensorRT::nvinfer
    DeepStream::nvdsgst_meta
    DeepStream::nvds_meta
    DeepStream::nvdspreprocess
    ${GSTREAMER_LIBRARIES}
)

add_executable(run_infer src/main.cpp)
target_link_libraries(run_infer PRIVATE carbot_inference)


#### INSTALLATION ####
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/.local)

## includes standard install dirs like bin/lib/include
include(GNUInstallDirs)

install(TARGETS carbot_inference
    EXPORT CarbotInferenceTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ## should eva to /lib for .so files
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} ## for .a files
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} ## should eva to /bin for executables
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} ## should eva to /include for header files
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Configure the config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/CarbotInferenceConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CarbotInferenceConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CarbotInference
)

# Export the targets to a script
install(EXPORT CarbotInferenceTargets
    FILE CarbotInferenceTargets.cmake
    NAMESPACE CarbotInference::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CarbotInference
)

# Install the config file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CarbotInferenceConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CarbotInference
)


