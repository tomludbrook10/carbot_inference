find_path(DeepStream_INCLUDE_DIR_CORE
NAMES nvdsmeta.h gstnvdsmeta.h
  HINTS
    ${DeepStream_DIR}/include
    /opt/nvidia/deepstream/deepstream-7.1/sources/includes/
)

find_path(DeepStream_INCLUDE_DIR_GSTPREPROCESS 
NAMES nvdspreprocess_meta.h
  HINTS
    ${DeepStream_DIR}/include
    /opt/nvidia/deepstream/deepstream-7.1/sources/gst-plugins/gst-nvdspreprocess/include/
)

## only need the header for nvdspreprocess_meta
if(DeepStream_INCLUDE_DIR_GSTPREPROCESS AND NOT TARGET DeepStream::nvdspreprocess)
    add_library(DeepStream::nvdspreprocess INTERFACE IMPORTED)
    set_target_properties(DeepStream::nvdspreprocess PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${DeepStream_INCLUDE_DIR_GSTPREPROCESS}"
    )
    set(DeepStream_nvdspreprocess_FOUND TRUE)
endif()

## lib_name is .so file, name is the target name. 
function(find_deepstream_library name lib_name)
    find_library(${name}_LIB
    NAMES ${lib_name}
      HINTS
        ${DeepStream_DIR}/lib
        /opt/nvidia/deepstream/deepstream-7.1/lib/
        /opt/nvidia/deepstream/deepstream-*/lib/ 
    )

    if (${name}_LIB AND DeepStream_INCLUDE_DIR_CORE AND NOT TARGET DeepStream::${name})
        add_library(DeepStream::${name} UNKNOWN IMPORTED)
        set_target_properties(DeepStream::${name} PROPERTIES
            IMPORTED_LOCATION "${${name}_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${DeepStream_INCLUDE_DIR_CORE}"
        )
    ## need to set the found variable for find_package_handle_standard_args
    set(DeepStream_${name}_FOUND TRUE PARENT_SCOPE)
    endif()
endfunction()


## the nvdspreprocess_meta is header only, so no lib needed.
find_deepstream_library(nvdsgst_meta nvdsgst_meta)
find_deepstream_library(nvds_meta nvds_meta)

## here we setting DeepStream_FOUND if all required libs are found.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DeepStream
    REQUIRED_VARS DeepStream_INCLUDE_DIR_CORE
    DeepStream_INCLUDE_DIR_GSTPREPROCESS
    DeepStream_nvdsgst_meta_FOUND
    DeepStream_nvds_meta_FOUND
    DeepStream_nvdspreprocess_FOUND
)