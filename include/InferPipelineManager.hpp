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
        ModelContext model_context,
        const std::string& save_path = "",
        const std::string& config_file_path = "/home/tom/carbot_inference/config/config_preprocess.txt",
        const std::string& model_path = "/home/tom/models/carbot_v1/modelv2.engine");
    bool setup();
    bool startPipelineAsync();
    void stopPipeline();

    // blocking call, until new result is available.
    bool getLatestResult(std::vector<float>& output_data);
private:
    const int WIDTH = 1920;
    const int HEIGHT = 1080;
    const uint64_t MAX_INFER_TIME_NS = 20000000; // 20ms
    const int BUFFER_POOL_SIZE = 7; // need to set higher than CONTEXT_BUFFER_SIZE
    ModelContext model_context_;

    GstElement* pipeline_, *source_;
    const std::string config_file_path_;
    const std::string model_path_;
    const std::string save_path_;

    uint64_t base_time_;

    std::unique_ptr<CudaManager> cuda_manager_;
    std::unique_ptr<ProcessData> process_data_;
    std::unique_ptr<ResultData> result_data_;

    bool cuda_manager_running_{true};
    bool receive_running_{true};
    std::thread gst_pipeline_thread_;
    std::thread cuda_infer_thread_;

    void startPipeline();
};