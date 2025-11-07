#pragma once
#include <gst/gst.h>
#include <string>
#include <memory>
#include <thread>

#include "CudaManager.hpp"
#include "Coms.hpp"



class InferPipelineManager {
public:
    explicit InferPipelineManager(
        const std::string& config_file_path = "/home/tom/carbot_inference/config_preprocess.txt",
        const std::string& model_path = "/home/tom/models/carbot_v1/modelv1.engine"
    );
    ~InferPipelineManager();
    bool setup();
    bool start_pipeline_async();
    void stop_pipeline();

    // blocking call, until new result is available.
    void getLatestResult(std::vector<float>& output_data);
private:
    const int WIDTH = 1920;
    const int HEIGHT = 1080;
    const int OUTPUT_SIZE = 16; // [1, 8, 2], current model is static sized. [1, 3, 256, 256] -> [1, 8, 2]
    const int INPUT_SIZE = 1 * 3 * 256 * 256;
    const int SIZE_OF_DATA_IN_BYTES = 4; // float.

    GstElement* pipeline_;
    const std::string config_file_path_;
    const std::string model_path_;

    std::unique_ptr<CudaManager> cuda_manager_;
    std::unique_ptr<ProcessData> process_data_;
    std::unique_ptr<ResultData> result_data_;

    bool cuda_manager_running_{true};
    std::thread gst_pipeline_thread_;
    std::thread cuda_infer_thread_;

    bool start_pipeline();
};