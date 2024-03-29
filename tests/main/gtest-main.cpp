#include <cstdio>
#include "gtest/gtest.h"

GTEST_API_ int main(int argc, char** argv)
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}