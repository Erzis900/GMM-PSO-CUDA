# cmake requirements
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Build options have to be before PROJECT(...)
SET(CMAKE_CONFIGURATION_TYPES "Release" CACHE PATH "Configuration types")
SET(CMAKE_BUILD_TYPE "Release" CACHE PATH "Current build configuration")

# GAUSSIANMIXTURE Project configuration
PROJECT(GAUSSIANMIXTURE LANGUAGES CXX CUDA)
SET(CMAKE_CXX_FLAGS "-std=c++11 -pthread")
#set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
#set(CUDA_NVCC_FLAGS "-g -G")

# Folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# Settings
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include")
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include/Defs")

# core libraries
option(BUILD_GAUSSIANMIXTURE "Build GaussianMixture libraries" ON)
option(BUILD_GAUSSIANMIXTURE_DEFS "Build Gaussian Mixture Defs library" ON)
mark_as_advanced(BUILD_GAUSSIANMIXTURE_DEFS)

# demos
option(BUILD_GAUSSIANMIXTURE_DEMO "Build Gaussian Mixture demonstration programs and libraries" ON)
mark_as_advanced(BUILD_GAUSSIANMIXTURE_DEMO_BASIC)
option(BUILD_GAUSSIANMIXTURE_DEMO_BASIC "Build basic Gaussian Mixture test" ON)

#additional dependencies

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

#Eigen
add_definitions(-Wno-deprecated -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

# Executable output directory
SET(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Executable output directory")
mark_as_advanced(RUNTIME_OUTPUT_DIRECTORY)

# Dynamic library output directory
SET(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Dynamic library output directory")
mark_as_advanced(LIBRARY_OUTPUT_DIRECTORY)

# Static library output directory
SET(ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Static library output directory")
mark_as_advanced(ARCHIVE_OUTPUT_DIRECTORY)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_DIRECTORY})
foreach(CONFIGURATION_TYPE ${CMAKE_CONFIGURATION_TYPES})
	string(TOUPPER ${CONFIGURATION_TYPE} CONFIGURATION_TYPE)
	SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${RUNTIME_OUTPUT_DIRECTORY})
	SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${LIBRARY_OUTPUT_DIRECTORY})
	SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${ARCHIVE_OUTPUT_DIRECTORY})
endforeach(CONFIGURATION_TYPE CMAKE_CONFIGURATION_TYPES)

if (MSVC)
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

###############################################################################
#
# GAUSSIANMIXTURE tinyXML2 library
#
###############################################################################

add_library(tinyxml2 SHARED ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/tinyXML/tinyxml2.h ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/tinyXML/tinyxml2.cpp)

###############################################################################
#
# GAUSSIANMIXTURE defs library
#
###############################################################################

# if(BUILD_GAUSSIANMIXTURE AND BUILD_GAUSSIANMIXTURE_DEFS)
#         file(GLOB DEFS_HEADERS
#             "${CMAKE_SOURCE_DIR}/include/defs.h"
#         )
#         ADD_LIBRARY(GAUSSIANMIXTURE_DEFS STATIC ${DEFS_HEADERS})
#         set_target_properties(GAUSSIANMIXTURE_DEFS PROPERTIES LINKER_LANGUAGE CXX)
# # 	INSTALL(TARGETS GAUSSIANMIXTUREDefs RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
#         INSTALL(FILES ${DEFS_HEADERS} DESTINATION include/Defs/)
# 
# endif(BUILD_GAUSSIANMIXTURE AND BUILD_GAUSSIANMIXTURE_DEFS)


###############################################################################
#
# Gaussian Mixture library
#
###############################################################################

if(BUILD_GAUSSIANMIXTURE)
        file(GLOB LIBGAUSSIANMIXTURE_SOURCES
            "${CMAKE_SOURCE_DIR}/src/*.cpp"
        )
        file(GLOB LIBGAUSSIANMIXTURE_HEADERS
            "${CMAKE_SOURCE_DIR}/include/*.h"
        )
        file(GLOB LIBGAUSSIANMIXTURE_CUDA_SOURCES
            "${CMAKE_SOURCE_DIR}/src/*.cu"
        )
        ADD_LIBRARY(libGaussianMixture STATIC ${LIBGAUSSIANMIXTURE_SOURCES} ${LIBGAUSSIANMIXTURE_CUDA_SOURCES} ${LIBGAUSSIANMIXTURE_HEADERS})
        TARGET_LINK_LIBRARIES(libGaussianMixture tinyxml2 cublas cusolver ${CUDA_LIBRARIES})
#         INSTALL(TARGETS Utilities RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
#         INSTALL(FILES ${LIBGAUSSIANMIXTURE_HEADERS} DESTINATION include/GaussianMixture/)
endif(BUILD_GAUSSIANMIXTURE)

###############################################################################
#
# GAUSSIANMIXTURE DEMO executableslanguage
#
###############################################################################

if(BUILD_GAUSSIANMIXTURE_DEMO AND BUILD_GAUSSIANMIXTURE_DEMO_BASIC)

        SET(DEMO_FILES
                ${PROJECT_ROOT}/resources/configGlobal.xml
        )

        SET(DEMO_SOURCES ./main.cpp)
        ADD_EXECUTABLE(GaussianMixture ${DEMO_SOURCES})
        TARGET_LINK_LIBRARIES(GaussianMixture libGaussianMixture tinyxml2)
#         INSTALL(TARGETS GaussianMixture RUNTIME DESTINATION bin)
#         INSTALL(FILES ${DEMO_FILES} DESTINATION bin)

endif(BUILD_GAUSSIANMIXTURE_DEMO AND BUILD_GAUSSIANMIXTURE_DEMO_BASIC)
