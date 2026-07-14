cmake_minimum_required(VERSION 3.15)
project(vulkan_hut)

set(CMAKE_CXX_STANDARD 17)

# 查找依赖包
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# 添加可执行文件
add_executable(vulkan_hut main.cpp)

# 链接库
target_link_libraries(vulkan_hut Vulkan::Vulkan glfw)

