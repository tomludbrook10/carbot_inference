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

std::vector<float> loadTestInputData(const std::string& input_path, size_t expected_size) {
    std::ifstream file(input_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open input data file: " << input_path << std::endl;
        return {};
    }
    std::vector<float> input_data(expected_size);
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(input_data.data()), expected_size * sizeof(float))) {
        std::cerr << "Failed to read input data file: " << input_path << std::endl;
        return {};
    }
    return input_data;
}

std::vector<float> saveTestOutputData(const std::string& output_path, std::vector<float>& output_data) {
    std::ofstream file(output_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open output data file for writing: " << output_path << std::endl;
        return {};
    }
    if (!file.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float))) {
        std::cerr << "Failed to write output data to file: " << output_path << std::endl;
        return {};
    }
    return output_data;
}

int main() {
    
    auto start = std::chrono::steady_clock::now();
    std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
    if (!runtime) {
        std::cerr << "Failed to create TensorRT Runtime." << std::endl;
        return -1;
    }

    std::vector<char> model_data = readModelFromFile("/home/tom/models/carbot_v1/outside_hallway_v3.engine");
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

    std::string input_path = "/home/tom/carbot_inference/rollouts/input_tensor_0.bin";
    std::string output_path = "/home/tom/carbot_inference/output/output_tensor_0.bin";

    size_t input_size = 3 * 280 * 280;
    size_t output_size = 2 * 6;
    void* input_buf;
    void* output_buf;

    std::vector<float> input_data = loadTestInputData(input_path, input_size);
    if (input_data.empty()) {
        return -1;
    }

    cudaMalloc(&input_buf, input_size * sizeof(float));
    cudaMalloc(&output_buf, output_size * sizeof(float));

    cudaMemcpy(input_buf, input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);


    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->setTensorAddress("input", input_buf);
    context->setTensorAddress("output", output_buf);

    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    std::vector<float> output_data(output_size);
    cudaMemcpy(output_data.data(), output_buf, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    saveTestOutputData(output_path, output_data);

    auto duration = std::chrono::steady_clock::now() - start;
    std::cout << "Model loaded and engine created in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
              << " ms." << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(input_buf);
    cudaFree(output_buf);

    return 0;

}