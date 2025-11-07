#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        } else if (severity == Severity::kINFO) {
            std::cout << "[INFO]: " << msg << std::endl;
        }
    }
} logger;


// we could speed this up by memory mapping the file if needed.
// as cuda engine requires pointer and the size. 
std::vector<char> readModelFromFile(const std::string& model_path) {
    // add the std::ios::ate flag to open at the end of the file to get the size
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return {};
    }

    // reduce allocations. 
    const std::streamsize size = file.tellg();
    std::vector<char> buffer(size);

    // reset the pointer to the beginning of the file
    file.seekg(0, std::ios::beg); // std::ios::beg is the beginning of the file and 0 offset. 
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read model file: " << model_path << std::endl;
        return {};
    }
    return buffer;
}

int main() {
    
    auto start = std::chrono::steady_clock::now();
    std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
    if (!runtime) {
        std::cerr << "Failed to create TensorRT Runtime." << std::endl;
        return -1;
    }

    std::vector<char> model_data = readModelFromFile("/home/tom/models/carbot_v1/modelv1.engine");
    if (model_data.empty()) {
        return -1;
    }

    // for now in-memory deserialization is sufficient for smaller models.
    std::unique_ptr<ICudaEngine> engine{runtime->deserializeCudaEngine(model_data.data(), model_data.size())};
    if (!engine) {
        std::cerr << "Failed to create TensorRT Engine." << std::endl;
        return -1;
    }

    std::unique_ptr<IExecutionContext> context{engine->createExecutionContext()};
    if (!context) {
        std::cerr << "Failed to create Execution Context." << std::endl;
        return -1;
    }

    size_t input_size = 1 * 3 * 256 * 256;
    size_t output_size = 1 * 8 * 2;
    void* input_buf;
    void* output_buf;
    cudaMalloc(&input_buf, input_size * sizeof(float));
    cudaMalloc(&output_buf, output_size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->setTensorAddress("input", input_buf);
    context->setTensorAddress("output", output_buf);

    context->enqueueV3(stream);

    auto duration = std::chrono::steady_clock::now() - start;
    std::cout << "Model loaded and engine created in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
              << " ms." << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(input_buf);
    cudaFree(output_buf);

    return 0;

}