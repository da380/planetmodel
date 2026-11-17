# Programs of the library in use: one writes a model file of fields of its
# own, one writes the file a code computing sensitivity kernels writes,
# one reads any model file, and two do the same for a flat model, a box
# layered in depth.  With the tests, the readers read what the writers
# wrote, and the kernel and the box are left beside the writer's tests'
# files for the Python to read.
foreach(_example IN ITEMS write_model write_kernel read_model write_flat_model
                          read_flat_model)
  add_executable(planetmodel_${_example} ${_example}.cpp)
  target_link_libraries(planetmodel_${_example} PRIVATE planetmodel::planetmodel)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(planetmodel_${_example} PRIVATE -Wall -Wextra -Wpedantic
      $<$<BOOL:${PLANETMODEL_WARNINGS_AS_ERRORS}>:-Werror>)
  endif()
endforeach()

if(PLANETMODEL_BUILD_TESTS)
  set(_pm_example_file ${CMAKE_CURRENT_BINARY_DIR}/example.nc)
  set(_pm_kernel_file ${PLANETMODEL_WRITTEN_DIR}/kernel.nc)
  set(_pm_flat_file ${PLANETMODEL_WRITTEN_DIR}/flat_example.nc)
  file(MAKE_DIRECTORY ${PLANETMODEL_WRITTEN_DIR})
  add_test(NAME example_write_model
           COMMAND planetmodel_write_model ${_pm_example_file})
  add_test(NAME example_read_model
           COMMAND planetmodel_read_model ${_pm_example_file})
  add_test(NAME example_write_kernel
           COMMAND planetmodel_write_kernel ${_pm_kernel_file})
  add_test(NAME example_read_kernel
           COMMAND planetmodel_read_model ${_pm_kernel_file})
  add_test(NAME example_write_flat_model
           COMMAND planetmodel_write_flat_model ${_pm_flat_file})
  add_test(NAME example_read_flat_model
           COMMAND planetmodel_read_flat_model ${_pm_flat_file})
  set_tests_properties(example_write_model PROPERTIES FIXTURES_SETUP example_file)
  set_tests_properties(example_read_model PROPERTIES FIXTURES_REQUIRED example_file)
  set_tests_properties(example_write_kernel PROPERTIES FIXTURES_SETUP kernel_file)
  set_tests_properties(example_read_kernel PROPERTIES FIXTURES_REQUIRED kernel_file)
  set_tests_properties(example_write_flat_model PROPERTIES FIXTURES_SETUP flat_file)
  set_tests_properties(example_read_flat_model PROPERTIES FIXTURES_REQUIRED flat_file)
endif()
