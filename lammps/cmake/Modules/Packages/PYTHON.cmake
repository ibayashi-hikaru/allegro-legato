if(CMAKE_VERSION VERSION_LESS 3.12)
  find_package(PythonLibs REQUIRED) # Deprecated since version 3.12
  target_include_directories(lammps PRIVATE ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(lammps PRIVATE ${PYTHON_LIBRARIES})
else()
  find_package(Python REQUIRED COMPONENTS Development)
  target_link_libraries(lammps PRIVATE Python::Python)
endif()
target_compile_definitions(lammps PRIVATE -DLMP_PYTHON)
