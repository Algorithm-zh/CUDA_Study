cmake_minimum_required(VERSION 3.18)
project(sum_array_offset LANGUAGES CUDA CXX)
add_executable(sum_array_offset sum_array_offset.cu)
# 开启 L1 缓存 (dlcm=ca)
target_compile_options(sum_array_offset PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -dlcm=ca>
)
