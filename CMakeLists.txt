cmake_minimum_required(VERSION 3.22)
project(carbot_inference)
set(CMAKE_CXX_STANDARD 17)

## add so that cmake can find the locally installed packages.
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

# Find GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)