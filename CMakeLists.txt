project(SemanticSegmentation-Libtorch)

cmake_minimum_required (VERSION 3.9)

add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(./include)

if(WIN32)
    include_directories(./Poco-1.9.4)
    link_directories(./Poco-1.9.4/lib/x64)
	set(CMAKE_PREFIX_PATH "D:/local/libtorch/")
	set(OpenCV_DIR "D:/local/opencv/build")
	set(CUDNN_INCLUDE_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include")
endif(WIN32)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(UNIX)
    find_package(Poco REQUIRED Foundation JSON)
endif(UNIX)

set(SRC_FILE src/cocoapi/cocoNote.cpp
			 src/cocoapi/mask.cpp
             src/cocoapi/mask_api.cpp
             src/DataSet/COCODataSet.cpp
             src/DataSet/COCODetection.cpp
             src/models/segmentation/DeeplabV3.cpp
             src/models/segmentation/SegmentationModel.cpp
             src/models/IntermediateLayerGetter
             src/models/segmentation/SegmentationModel.cpp
             src/models/IntermediateLayerGetter.cpp
             src/models/resnet.cpp
             src/utills/metric_logger.cpp
             src/utills/transforms.cpp
             src/main.cpp
            )

add_executable(${PROJECT_NAME} ${SRC_FILE})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

if(WIN32)
    target_link_libraries(${PROJECT_NAME} "PocoFoundation.lib" )
    target_link_libraries(${PROJECT_NAME} "PocoJSON.lib" )
endif(WIN32)

if(UNIX)
    target_link_libraries(${PROJECT_NAME} Poco::JSON)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif(UNIX)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)