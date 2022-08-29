# Repository Structures

```
+-- \3rdParty       Source dependencies (Already Included)
+-- \build          Your optional building directories
|   +-- \_deps      Source dependencies (Resolved by CMake)
|   +-- \bin        Built binary distributions
|   +-- \lib        Built static libraries and pdbs
+-- \cmake          CMake scripts for binary dependencies (Resolved Manually)
+-- \docs           Documents
+-- \ext            CMake scripts for source dependencies (Resolved by CMake)
+-- \include        Renderer SDK include files
|   +-- \delegate   Delegate library
|   +-- \libkernel  Renderer CUDA kernel library
|   +-- \librender  Renderer core library
|   +-- \nodes      Nodes within Renderer core library
+-- \src            Renderer source code
+-- \tests          Unit tests
+-- .clang-format   Clang format file
+-- .gitignore      Git ignore file
+-- CMakeLists.txt  Build generator script for CMake
+-- README.md       Main page
```

# SDK
Currently we are lacking CMake install scripts for SDK building. Add one in the future; for now please building **Colvillea** from source and use as a static library (no ABI compatibility guarantee for API interfaces currently, so do not build as a shared library at the moment).

