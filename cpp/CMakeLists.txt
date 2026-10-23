# planetmodel's C++ library: header-only readers and writers of what the
# Python writes, and the interpolation of its rules.
#
# A code fetches it at a release tag and links one target:
#
#   FetchContent_Declare(planetmodel
#     GIT_REPOSITORY https://github.com/da380/planetmodel.git
#     GIT_TAG        v<version>
#     SOURCE_SUBDIR  cpp)
#   FetchContent_MakeAvailable(planetmodel)
#   target_link_libraries(my_code PRIVATE planetmodel::planetmodel)
#
# or finds an installed copy, the same target by find_package(planetmodel).
#
# The model file needs netCDF-C, behind PLANETMODEL_WITH_NETCDF: on by
# default and then required, so that a build lacking it fails at
# configuration and says so; off, nothing of netCDF is looked for and the
# headers that need it refuse to compile.
cmake_minimum_required(VERSION 3.24)

# The version is the Python package's, read from the one place it is set.
set(_pm_pyproject "${CMAKE_CURRENT_LIST_DIR}/../pyproject.toml")
if(NOT EXISTS "${_pm_pyproject}")
  message(FATAL_ERROR
    "planetmodel: ${_pm_pyproject} is missing; the C++ library is fetched "
    "with the whole repository and SOURCE_SUBDIR cpp")
endif()
file(STRINGS "${_pm_pyproject}" _pm_version_line
     REGEX "^version = \"[^\"]+\"$" LIMIT_COUNT 1)
string(REGEX REPLACE "^version = \"([^\"]+)\"$" "\\1" PLANETMODEL_VERSION_STRING
       "${_pm_version_line}")
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" _pm_version_numbers
       "${PLANETMODEL_VERSION_STRING}")
if(NOT _pm_version_numbers)
  message(FATAL_ERROR
    "planetmodel: no version found in ${_pm_pyproject}")
endif()

project(planetmodel VERSION ${_pm_version_numbers} LANGUAGES CXX)

# The compile commands of planetmodel's own build, for editors (cpp/.clangd
# looks for them in build/cpp at the repository's root).
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(PLANETMODEL_WITH_NETCDF
       "Read and write model files, over netCDF-C (required when on)" ON)
option(PLANETMODEL_BUILD_TESTS
       "Build planetmodel's own tests" ${PROJECT_IS_TOP_LEVEL})
option(PLANETMODEL_BUILD_EXAMPLES
       "Build the example programs (with the model file)" ${PROJECT_IS_TOP_LEVEL})
option(PLANETMODEL_BUILD_DOCS
       "Build the Doxygen reference, target planetmodel_docs" OFF)
option(PLANETMODEL_WARNINGS_AS_ERRORS
       "Treat warnings in planetmodel's own tests and examples as errors" OFF)
# On even when this is a subproject, unlike the options above. A code that
# links planetmodel into a library it installs and exports -- statically, or
# publicly -- cannot be installed unless planetmodel is in an export set too:
# CMake refuses with "requires target planetmodel that is not in any export
# set". Interpolation and NumericConcepts install themselves as subprojects for
# the same reason, so the whole chain lands in the code's prefix.
option(PLANETMODEL_INSTALL "Generate the install and export rules" ON)

# The Interpolation revision fetched when none is installed. The default is
# main: the libraries are one author's and move together, so on a development
# branch tracking main is the point. A release must name a state that can be
# rebuilt, so the commit that tags a release sets this default to the commit
# it was tested against, and the commit after it sets it back to main. From
# outside: cmake -DPLANETMODEL_INTERPOLATION_TAG=<commit>
set(PLANETMODEL_INTERPOLATION_TAG main CACHE STRING
    "The da380/Interpolation revision to fetch if none is installed")
mark_as_advanced(PLANETMODEL_INTERPOLATION_TAG)

include(FetchContent)
include(GNUInstallDirs)

# Interpolation brings NumericConcepts with it, at the revision it asks for; a
# code that declared either first keeps its own declaration.
FetchContent_Declare(Interpolation
  GIT_REPOSITORY https://github.com/da380/Interpolation.git
  GIT_TAG        ${PLANETMODEL_INTERPOLATION_TAG}
  FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(Interpolation)

add_library(planetmodel INTERFACE)
add_library(planetmodel::planetmodel ALIAS planetmodel)
target_include_directories(planetmodel INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(planetmodel INTERFACE cxx_std_23)
target_link_libraries(planetmodel INTERFACE Interpolation::Interpolation)
target_compile_definitions(planetmodel INTERFACE
  "PLANETMODEL_VERSION=\"${PLANETMODEL_VERSION_STRING}\"")

if(PLANETMODEL_WITH_NETCDF)
  # netCDF-C's own CMake package first, then pkg-config; the imported
  # target is made global so that the codes linking planetmodel, in other
  # directories, see it too.
  set(_pm_netcdf "")
  find_package(netCDF CONFIG QUIET)
  if(TARGET netCDF::netcdf)
    set(_pm_netcdf netCDF::netcdf)
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(PLANETMODEL_NETCDF QUIET IMPORTED_TARGET GLOBAL netcdf)
      if(TARGET PkgConfig::PLANETMODEL_NETCDF)
        set(_pm_netcdf PkgConfig::PLANETMODEL_NETCDF)
      endif()
    endif()
  endif()
  if(NOT _pm_netcdf)
    message(FATAL_ERROR
      "planetmodel: netCDF-C was not found, neither by find_package(netCDF "
      "CONFIG) (netCDF_DIR or CMAKE_PREFIX_PATH) nor by pkg-config (netcdf.pc "
      "on PKG_CONFIG_PATH). Install it and point CMake at it, or set "
      "PLANETMODEL_WITH_NETCDF=OFF to build without the model file.")
  endif()
  get_target_property(_pm_netcdf_imported ${_pm_netcdf} IMPORTED)
  if(_pm_netcdf_imported)
    set_target_properties(${_pm_netcdf} PROPERTIES IMPORTED_GLOBAL TRUE)
  endif()
  target_link_libraries(planetmodel INTERFACE ${_pm_netcdf})
  target_compile_definitions(planetmodel INTERFACE PLANETMODEL_WITH_NETCDF=1)
  message(STATUS "planetmodel: the model file over ${_pm_netcdf}")
else()
  message(STATUS "planetmodel: built without netCDF-C, so without the model file")
endif()

# The installed package: the headers, the target, and a configuration file
# that finds Interpolation and netCDF-C again -- netCDF-C the way it was found
# here, since the exported target names that imported target (netCDF::netcdf
# from netCDF-C's own package, or PkgConfig::PLANETMODEL_NETCDF).
if(PLANETMODEL_INSTALL)
  include(CMakePackageConfigHelpers)
  set(_pm_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/planetmodel)

  install(TARGETS planetmodel EXPORT planetmodelTargets)
  install(DIRECTORY include/planetmodel
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT planetmodelTargets
          FILE planetmodelTargets.cmake
          NAMESPACE planetmodel::
          DESTINATION ${_pm_cmake_dir})

  # Within a major version the format and the API only grow. Header-only,
  # so nothing in the package is architecture-specific.
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/planetmodelConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)

  if(PLANETMODEL_WITH_NETCDF)
    set(PLANETMODEL_NETCDF_TARGET ${_pm_netcdf})
  else()
    set(PLANETMODEL_NETCDF_TARGET "")
  endif()
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/planetmodelConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/planetmodelConfig.cmake
    INSTALL_DESTINATION ${_pm_cmake_dir})

  install(FILES
          ${CMAKE_CURRENT_BINARY_DIR}/planetmodelConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/planetmodelConfigVersion.cmake
          DESTINATION ${_pm_cmake_dir})
endif()

# The model files the writer's tests and the kernel example write, for the
# Python to read back (tests/test_cpp_written.py, with
# PLANETMODEL_CPP_WRITTEN naming it).
set(PLANETMODEL_WRITTEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/tests/written")

if(PLANETMODEL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PLANETMODEL_BUILD_EXAMPLES AND PLANETMODEL_WITH_NETCDF)
  add_subdirectory(examples)
endif()

# The reference, by the Doxyfile the documentation's own build runs too;
# every warning fails it.
if(PLANETMODEL_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  add_custom_target(planetmodel_docs
    COMMAND ${CMAKE_COMMAND} -E env
            PLANETMODEL_DOCS_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/docs
            PLANETMODEL_VERSION=${PLANETMODEL_VERSION_STRING}
            ${DOXYGEN_EXECUTABLE} docs/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "planetmodel: the reference, into ${CMAKE_CURRENT_BINARY_DIR}/docs/cpp-reference"
    VERBATIM)
endif()
