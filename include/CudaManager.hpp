#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "Coms.hpp"

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        } else if (severity == Severity::kINFO) {
            std::cout << "[INFO]: " << msg << std::endl;
        }
    }
};

class CudaManager {
public:
    bool setup(const std::string &model_path, const int OUTPUT_SIZE, const int SIZE_OF_DATA_IN_BYTES);
    void infer(ProcessData* data, 
               ResultData* result_data, 
               const int INPUT_SIZE, 
               const int OUTPUT_SIZE, 
               const int SIZE_OF_DATA_IN_BYTES,
               bool *running_);

private:
    cudaStream_t stream_;
    cudaError_t err_;
    void* output_buf_;

    std::unique_ptr<Logger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<char> readModelFromFile(const std::string& model_path);
};