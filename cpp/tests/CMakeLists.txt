# planetmodel's own tests: GoogleTest units, held to fixtures that the
# Python writes into PLANETMODEL_FIXTURES_DIR (make_fixtures.py), and
# consumer builds that fetch the library as a code would.
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  FIND_PACKAGE_ARGS NAMES GTest)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(PLANETMODEL_FIXTURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fixtures" CACHE PATH
    "Where make_fixtures.py wrote the fixtures")

add_executable(planetmodel_tests
  test_deck.cpp
  test_error.cpp
  test_json.cpp
  test_manifest.cpp
  $<$<BOOL:${PLANETMODEL_WITH_NETCDF}>:test_modelfile.cpp>
  $<$<BOOL:${PLANETMODEL_WITH_NETCDF}>:test_rules.cpp>
  $<$<BOOL:${PLANETMODEL_WITH_NETCDF}>:test_surfacefile.cpp>
  $<$<BOOL:${PLANETMODEL_WITH_NETCDF}>:test_writer.cpp>
  test_units.cpp)
target_link_libraries(planetmodel_tests PRIVATE planetmodel::planetmodel
                      GTest::gtest_main)
target_compile_definitions(planetmodel_tests PRIVATE
  "PLANETMODEL_FIXTURES_DIR=\"${PLANETMODEL_FIXTURES_DIR}\""
  "PLANETMODEL_WRITTEN_DIR=\"${PLANETMODEL_WRITTEN_DIR}\"")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(planetmodel_tests PRIVATE -Wall -Wextra -Wpedantic
    $<$<BOOL:${PLANETMODEL_WARNINGS_AS_ERRORS}>:-Werror>)
endif()
gtest_discover_tests(planetmodel_tests)

# A code fetching the library from this checkout, as FetchContent would
# from a tag: built with and without netCDF-C, the netCDF header refused
# when it is off, and the configuration refused by name when netCDF-C is
# asked for and cannot be found.  The dependencies already fetched here
# are handed on, so that nothing is fetched twice.
set(_pm_consumer_args
  -DCONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumer
  -DPLANETMODEL_ROOT=${PROJECT_SOURCE_DIR}/..
  -DWORK=${CMAKE_CURRENT_BINARY_DIR}/consumer
  -DCXX=${CMAKE_CXX_COMPILER}
  -DINTERPOLATION_SOURCE=${interpolation_SOURCE_DIR}
  -DNUMERICCONCEPTS_SOURCE=${numericconcepts_SOURCE_DIR})
foreach(_case IN ITEMS with_netcdf without_netcdf header_refused netcdf_absent)
  add_test(NAME consumer_${_case}
           COMMAND ${CMAKE_COMMAND} ${_pm_consumer_args} -DCASE=${_case}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/consumer/run.cmake)
  set_tests_properties(consumer_${_case} PROPERTIES LABELS consumer)
endforeach()
