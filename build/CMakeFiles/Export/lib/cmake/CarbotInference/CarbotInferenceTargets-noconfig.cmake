#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CarbotInference::carbot_inference" for configuration ""
set_property(TARGET CarbotInference::carbot_inference APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(CarbotInference::carbot_inference PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcarbot_inference.so"
  IMPORTED_SONAME_NOCONFIG "libcarbot_inference.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS CarbotInference::carbot_inference )
list(APPEND _IMPORT_CHECK_FILES_FOR_CarbotInference::carbot_inference "${_IMPORT_PREFIX}/lib/libcarbot_inference.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
