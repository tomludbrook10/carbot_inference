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
    CudaManager(ModelContext model_context) : model_context_(model_context) {}
    bool setup(const std::string &model_path);
    void infer(ProcessData* input_data,
               ResultData* result_data, 
               const uint64_t MAX_INFER_TIME_NS,
               uint64_t *base_time,
               const std::string& save_path,
               bool *running_);

private:
    cudaStream_t stream_;
    cudaError_t err_;
    void *input_buf_;
    void* output_buf_;
    ModelContext model_context_;

    std::unique_ptr<Logger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<char> readModelFromFile(const std::string& model_path);
    CudaBuffer getCudaBuffer(GstBuffer* gst_buffer);

};