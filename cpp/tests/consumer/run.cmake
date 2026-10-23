# One consumer case, run by CTest as `cmake -P`: configure and build the
# consumer project in WORK/CASE and check the outcome the case expects.
#
#   with_netcdf     the default: configures, builds both programs, runs them
#   without_netcdf  PLANETMODEL_WITH_NETCDF=OFF: configures without looking
#                   for netCDF-C, builds `scales`
#   header_refused  PLANETMODEL_WITH_NETCDF=OFF: `netcdf` fails to compile,
#                   with the header's message
#   netcdf_absent   the default with netCDF-C hidden: the configuration
#                   fails, naming netCDF-C and the option
#   installed       the library installed into a prefix (with Interpolation
#                   and NumericConcepts beside it), then found there by
#                   find_package: builds and runs both programs
set(build "${WORK}/${CASE}")
file(REMOVE_RECURSE "${build}")
set(fetched
    -DFETCHCONTENT_SOURCE_DIR_INTERPOLATION=${INTERPOLATION_SOURCE}
    -DFETCHCONTENT_SOURCE_DIR_NUMERICCONCEPTS=${NUMERICCONCEPTS_SOURCE})
set(configure ${CMAKE_COMMAND} -S "${CONSUMER_SOURCE}" -B "${build}"
    -DCMAKE_CXX_COMPILER=${CXX}
    -DPLANETMODEL_ROOT=${PLANETMODEL_ROOT}
    ${fetched})
if(CASE STREQUAL "installed")
  set(prefix "${build}/prefix")
  set(library "${build}/planetmodel")
  function(library_step what)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE status
                    OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT status EQUAL 0)
      message(FATAL_ERROR "the library's ${what} failed:\n${out}\n${err}")
    endif()
  endfunction()
  library_step(configuration
    ${CMAKE_COMMAND} -S "${PLANETMODEL_ROOT}/cpp" -B "${library}"
    -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_INSTALL_PREFIX=${prefix}
    -DPLANETMODEL_BUILD_TESTS=OFF -DPLANETMODEL_BUILD_EXAMPLES=OFF
    ${fetched})
  library_step(build ${CMAKE_COMMAND} --build "${library}")
  library_step(installation ${CMAKE_COMMAND} --install "${library}")
  # Only the prefix: nothing of the source tree or the fetched dependencies.
  set(configure ${CMAKE_COMMAND} -S "${CONSUMER_SOURCE}" -B "${build}/consumer"
      -DCMAKE_CXX_COMPILER=${CXX} -DPLANETMODEL_INSTALLED=ON
      -DCMAKE_PREFIX_PATH=${prefix})
  set(build "${build}/consumer")
elseif(CASE STREQUAL "without_netcdf" OR CASE STREQUAL "header_refused")
  list(APPEND configure -DPLANETMODEL_WITH_NETCDF=OFF)
elseif(CASE STREQUAL "netcdf_absent")
  # Neither of the two ways in can find it: the package lookup is disabled
  # and pkg-config is given nowhere to look.
  list(APPEND configure -DCMAKE_DISABLE_FIND_PACKAGE_netCDF=ON)
  set(ENV{PKG_CONFIG_LIBDIR} "${build}/nowhere")
  set(ENV{PKG_CONFIG_PATH} "")
endif()

execute_process(COMMAND ${configure} RESULT_VARIABLE status
                OUTPUT_VARIABLE out ERROR_VARIABLE err)
set(log "${out}\n${err}")

if(CASE STREQUAL "netcdf_absent")
  if(status EQUAL 0)
    message(FATAL_ERROR "configuration succeeded with netCDF-C hidden:\n${log}")
  endif()
  if(NOT log MATCHES "netCDF-C was not found" OR
     NOT log MATCHES "PLANETMODEL_WITH_NETCDF=OFF")
    message(FATAL_ERROR "configuration failed, but not by name:\n${log}")
  endif()
  return()
endif()
if(NOT status EQUAL 0)
  message(FATAL_ERROR "configuration failed:\n${log}")
endif()
if(CASE STREQUAL "without_netcdf" AND log MATCHES "over netCDF")
  message(FATAL_ERROR "netCDF-C was looked for with the option off:\n${log}")
endif()

if(CASE STREQUAL "header_refused")
  execute_process(COMMAND ${CMAKE_COMMAND} --build "${build}" --target netcdf
                  RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(status EQUAL 0)
    message(FATAL_ERROR "the netCDF header compiled with the option off")
  endif()
  if(NOT "${out}${err}" MATCHES "configured without netCDF-C")
    message(FATAL_ERROR "the build failed, but not with the header's message:\n${out}\n${err}")
  endif()
  return()
endif()

set(targets scales)
if(CASE STREQUAL "with_netcdf" OR CASE STREQUAL "installed")
  list(APPEND targets netcdf)
endif()
foreach(target IN LISTS targets)
  execute_process(COMMAND ${CMAKE_COMMAND} --build "${build}" --target ${target}
                  RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "building ${target} failed:\n${out}\n${err}")
  endif()
  execute_process(COMMAND "${build}/${target}" RESULT_VARIABLE status
                  OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "running ${target} failed:\n${out}\n${err}")
  endif()
  message(STATUS "${target}: ${out}")
endforeach()
