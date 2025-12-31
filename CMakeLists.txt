cmake_minimum_required(VERSION 3.15)
project(graphzero)

# 1. Setup Dependencies
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP)

# 2. Define the Module (Must match the Python import name!)
nanobind_add_module(graphzero 
    src/bindings.cpp
)

# 3. Compiler Settings
target_include_directories(graphzero PRIVATE src)
target_compile_features(graphzero PUBLIC cxx_std_20)
target_compile_definitions(graphzero PRIVATE NOMINMAX)

# Link OpenMP if found (Essential for your Day 20 speedup)
if(OpenMP_CXX_FOUND)
    target_link_libraries(graphzero PRIVATE OpenMP::OpenMP_CXX)
endif()

# 4. Installation (Critical for pip)
install(TARGETS graphzero DESTINATION graphzero)