# A code that fetches planetmodel's C++ library as the documentation tells
# it to, here from a local checkout, or with PLANETMODEL_INSTALLED on finds an
# installed copy instead: `scales` uses what needs no netCDF, `netcdf`
# includes the netCDF header and calls netCDF-C.
cmake_minimum_required(VERSION 3.24)
project(planetmodel_consumer LANGUAGES CXX)

if(PLANETMODEL_INSTALLED)
  find_package(planetmodel REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(planetmodel
    SOURCE_DIR    ${PLANETMODEL_ROOT}
    SOURCE_SUBDIR cpp)
  FetchContent_MakeAvailable(planetmodel)
endif()

add_executable(scales scales.cpp)
target_link_libraries(scales PRIVATE planetmodel::planetmodel)

add_executable(netcdf netcdf.cpp)
target_link_libraries(netcdf PRIVATE planetmodel::planetmodel)
