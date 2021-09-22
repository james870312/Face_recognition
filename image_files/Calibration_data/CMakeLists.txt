cmake_minimum_required(VERSION 2.8)

# Install image files
file(GLOB PROJECT_IMGS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.png
    ${CMAKE_CURRENT_SOURCE_DIR}/*.jpg
    ${CMAKE_CURRENT_SOURCE_DIR}/*.tiff
    )

install(FILES ${PROJECT_IMGS} DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/Calibration_data")
