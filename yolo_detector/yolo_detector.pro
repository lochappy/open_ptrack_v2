QT += core
QT -= gui

CONFIG += c++11

TARGET = yolo_detector
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

DEFINES += OPENCV
LIBS += -lopencv_world

INCLUDEPATH += $$PWD/darknet_opt/include
INCLUDEPATH += $$PWD/darknet_opt/src
INCLUDEPATH += $$PWD/include

QMAKE_CXXFLAGS_DEBUG += -O4 -g
QMAKE_CFLAGS_DEBUG += -O4 -g

SOURCES += \
    darknet_opt/src/activation_layer.c \
    darknet_opt/src/activations.c \
    darknet_opt/src/avgpool_layer.c \
    darknet_opt/src/batchnorm_layer.c \
    darknet_opt/src/blas.c \
    darknet_opt/src/box.c \
    darknet_opt/src/col2im.c \
    #darknet_opt/src/compare.c \
    darknet_opt/src/connected_layer.c \
    darknet_opt/src/convolutional_layer.c \
    darknet_opt/src/cost_layer.c \
    darknet_opt/src/crnn_layer.c \
    darknet_opt/src/crop_layer.c \
    darknet_opt/src/cuda.c \
    darknet_opt/src/data.c \
    darknet_opt/src/deconvolutional_layer.c \
    darknet_opt/src/demo.c \
    darknet_opt/src/detection_layer.c \
    darknet_opt/src/dropout_layer.c \
    darknet_opt/src/gemm.c \
    darknet_opt/src/gru_layer.c \
    darknet_opt/src/im2col.c \
    darknet_opt/src/image.c \
    darknet_opt/src/l2norm_layer.c \
    darknet_opt/src/layer.c \
    darknet_opt/src/list.c \
    darknet_opt/src/local_layer.c \
    darknet_opt/src/logistic_layer.c \
    darknet_opt/src/lstm_layer.c \
    darknet_opt/src/matrix.c \
    darknet_opt/src/maxpool_layer.c \
    darknet_opt/src/network.c \
    darknet_opt/src/normalization_layer.c \
    darknet_opt/src/option_list.c \
    darknet_opt/src/parser.c \
    darknet_opt/src/region_layer.c \
    darknet_opt/src/reorg_layer.c \
    darknet_opt/src/rnn_layer.c \
    darknet_opt/src/route_layer.c \
    darknet_opt/src/shortcut_layer.c \
    darknet_opt/src/softmax_layer.c \
    darknet_opt/src/tree.c \
    darknet_opt/src/upsample_layer.c \
    darknet_opt/src/utils.c \
    darknet_opt/src/yolo_layer.c \
    main.cpp \
    include/run_yolo.c \
    src/yolo_based_people_detector_node.cpp
SOURCES -= src/yolo_based_people_detector_node.cpp
# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

HEADERS += \
    darknet_opt/include/darknet.h \
    darknet_opt/src/activation_layer.h \
    darknet_opt/src/activations.h \
    darknet_opt/src/avgpool_layer.h \
    darknet_opt/src/batchnorm_layer.h \
    darknet_opt/src/blas.h \
    darknet_opt/src/box.h \
    darknet_opt/src/classifier.h \
    darknet_opt/src/col2im.h \
    darknet_opt/src/connected_layer.h \
    darknet_opt/src/convolutional_layer.h \
    darknet_opt/src/cost_layer.h \
    darknet_opt/src/crnn_layer.h \
    darknet_opt/src/crop_layer.h \
    darknet_opt/src/cuda.h \
    darknet_opt/src/data.h \
    darknet_opt/src/deconvolutional_layer.h \
    darknet_opt/src/demo.h \
    darknet_opt/src/detection_layer.h \
    darknet_opt/src/dropout_layer.h \
    darknet_opt/src/gemm.h \
    darknet_opt/src/gru_layer.h \
    darknet_opt/src/im2col.h \
    darknet_opt/src/image.h \
    darknet_opt/src/l2norm_layer.h \
    darknet_opt/src/layer.h \
    darknet_opt/src/list.h \
    darknet_opt/src/local_layer.h \
    darknet_opt/src/logistic_layer.h \
    darknet_opt/src/lstm_layer.h \
    darknet_opt/src/matrix.h \
    darknet_opt/src/maxpool_layer.h \
    darknet_opt/src/network.h \
    darknet_opt/src/normalization_layer.h \
    darknet_opt/src/option_list.h \
    darknet_opt/src/parser.h \
    darknet_opt/src/region_layer.h \
    darknet_opt/src/reorg_layer.h \
    darknet_opt/src/rnn_layer.h \
    darknet_opt/src/route_layer.h \
    darknet_opt/src/shortcut_layer.h \
    darknet_opt/src/softmax_layer.h \
    darknet_opt/src/stb_image.h \
    darknet_opt/src/stb_image_write.h \
    darknet_opt/src/tree.h \
    darknet_opt/src/upsample_layer.h \
    darknet_opt/src/utils.h \
    darknet_opt/src/yolo_layer.h \
    include/run_yolo.h

DISTFILES += \
    darknet_opt/src/activation_kernels.cu \
    darknet_opt/src/activation_kernels.cu \
    darknet_opt/src/avgpool_layer_kernels.cu \
    darknet_opt/src/blas_kernels.cu \
    darknet_opt/src/col2im_kernels.cu \
    darknet_opt/src/convolutional_kernels.cu \
    darknet_opt/src/crop_layer_kernels.cu \
    darknet_opt/src/deconvolutional_kernels.cu \
    darknet_opt/src/dropout_layer_kernels.cu \
    darknet_opt/src/im2col_kernels.cu \
    darknet_opt/src/maxpool_layer_kernels.cu

#-------------------------------------------------
#
# Cuda config
#
#-------------------------------------------------
#DEFINES += GPU
#CUDA_SOURCES += \
#    darknet_opt/src/activation_kernels.cu \
#    darknet_opt/src/activation_kernels.cu \
#    darknet_opt/src/avgpool_layer_kernels.cu \
#    darknet_opt/src/blas_kernels.cu \
#    darknet_opt/src/col2im_kernels.cu \
#    darknet_opt/src/convolutional_kernels.cu \
#    darknet_opt/src/crop_layer_kernels.cu \
#    darknet_opt/src/deconvolutional_kernels.cu \
#    darknet_opt/src/dropout_layer_kernels.cu \
#    darknet_opt/src/im2col_kernels.cu \
#    darknet_opt/src/maxpool_layer_kernels.cu

## C++ flags
#QMAKE_CXXFLAGS_RELEASE = -O3

## Path to cuda toolkit install
#CUDA_DIR = /usr/local/cuda

## Path to header and libs files
#INCLUDEPATH += $$CUDA_DIR/include
#QMAKE_LIBDIR += $$CUDA_DIR/lib64 # Note I'm using a 64 bits Operating system

## libs used in your code
#LIBS += -lcuda -lcudart

## GPU architecture
#CUDA_ARCH = -gencode arch=compute_50,code=compute_50 \
#            -gencode arch=compute_52,code=sm_52 \
#            -gencode arch=compute_60,code=sm_60 \
#            -gencode arch=compute_61,code=sm_61
# # I've a old device. Adjust with your compute capability

## Here are some NVCC flags I've always used by default.
#NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

## Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
#CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

#CONFIG(debug, debug|release) {
#cuda.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG -m64 -O3 $$CUDA_ARCH -c $$NVCCFLAGS \
#$$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} 2>&1 \
#| sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
#}
#else{
#cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 $$CUDA_ARCH -c $$NVCCFLAGS \
#$$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} 2>&1 \
#| sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
#}

#cuda.dependency_type = TYPE_C
#cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
#cuda.input = CUDA_SOURCES
#cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

## Tell Qt that we want add more stuff to the Makefile
#QMAKE_EXTRA_COMPILERS += cuda
