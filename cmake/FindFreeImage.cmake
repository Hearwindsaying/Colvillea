# Note that MDL does not have a built-in cmake integration support so we wrap our own...
function(FIND_FREEIMAGE FREEIMAGE_INCLUDE_DIR FREEIMAGE_PLUS_INCLUDE_DIR FREEIMAGE_LIB_DIR FREEIMAGE_DLL_DIR)

    set(FREE_IMAGE_DIR "NOT-SPECIFIED" CACHE PATH "Path to the FreeImage library.")

    # Check the FreeImage dependency.
    if(NOT EXISTS ${FREE_IMAGE_DIR})
        message(FATAL_ERROR "FREE_IMAGE_DIR could not be found. Please download a prebuilt binary release from https://freeimage.sourceforge.io/download.html.")
        return()
    endif()

    # Collect include directories and linking libraries.
    set(${FREEIMAGE_INCLUDE_DIR} ${FREE_IMAGE_DIR}/dist/x64 PARENT_SCOPE)
    #message("FREEIMAGE_INCLUDE_DIR: ${FREEIMAGE_INCLUDE_DIR}")

    # We use FreeImagePlus wrapper.
    set(${FREEIMAGE_PLUS_INCLUDE_DIR} ${FREE_IMAGE_DIR}/Wrapper/FreeImagePlus/dist/x64 PARENT_SCOPE)
    #message("FREEIMAGE_PLUS_INCLUDE_DIR: ${FREEIMAGE_PLUS_INCLUDE_DIR}")

    set(${FREEIMAGE_LIB_DIR} ${FREE_IMAGE_DIR}/Dist/x64/FreeImage.lib PARENT_SCOPE)
    #message("FREEIMAGE_LIB_DIR: ${FREEIMAGE_LIB_DIR}")

    set(${FREEIMAGE_DLL_DIR} ${FREE_IMAGE_DIR}/Dist/x64/FreeImage.dll PARENT_SCOPE)
endfunction()