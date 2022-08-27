# Note that MDL does not have a built-in cmake integration support so we wrap our own...
function(FIND_MDL MDL_INCLUDE_DIR MDL_DLL_DIR)

    set(MDL_DIR "NOT-SPECIFIED" CACHE PATH "Path to the MDL library.")

    # Check the MDL dependency.
    if(NOT EXISTS ${MDL_DIR})
        message(FATAL_ERROR "MDL_DIR could not be found. Please download a prebuilt binary release from https://www.nvidia.com/en-us/design-visualization/technologies/material-definition-language/.")
        return()
    endif()

    # Collect include directories and linking libraries.
    set(${MDL_INCLUDE_DIR} ${MDL_DIR}/include PARENT_SCOPE)
    message("MDL_INCLUDE_DIR: ${MDL_INCLUDE_DIR}")

    set(${MDL_DLL_DIR} ${MDL_DIR}/nt-x86-64/lib PARENT_SCOPE)
    message("MDL_DLL_DIR: ${MDL_DLL_DIR}")
endfunction()