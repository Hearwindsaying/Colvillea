### Authored by Zihong Zhou

set(SOURCE_GLAD
    src/glad.c
)

set(INCLUDE_GLAD
    include/glad/glad.h
    include/khr/khrplatform.h
)

add_library(glad
STATIC
    ${INCLUDE_GLAD}
    ${SOURCE_GLAD}
)

target_include_directories(glad
PUBLIC
    ${CMAKE_SOURCE_DIR}/3rdParty/glad/include
)

set_target_properties(glad PROPERTIES
    FOLDER external
)