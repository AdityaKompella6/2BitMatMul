cmake_minimum_required(VERSION 3.14)
project(hw1 LANGUAGES C CXX)

# Packaging system
if (NOT ${GROUP_NO} STREQUAL 00)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "CS5220Group${GROUP_NO}_hw1")
    install(FILES dgemm-blocked.c DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()

foreach (BENCHMARK IN ITEMS naive blocked blas)
    # Create benchmarking executable
    add_executable(benchmark-${BENCHMARK} dgemm-${BENCHMARK}.c)
    target_link_libraries(benchmark-${BENCHMARK} PRIVATE benchmark ${BLAS_LIBRARIES})
    target_compile_features(benchmark-${BENCHMARK} PRIVATE c_std_11 c_restrict)
    target_compile_options(benchmark-${BENCHMARK} PRIVATE -Wall -pedantic -march=znver3)

    # Generate job script
    configure_file(job.in job-${BENCHMARK})
endforeach ()
