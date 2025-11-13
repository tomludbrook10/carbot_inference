#include "InferPipelineManager.hpp"

#include <csignal>
#include <chrono>
#include <thread>

static bool run = true;

int main() {

    ModelContext model_context;
    model_context.NUM_WAYPOINTS = 4;
    model_context.CONTEXT_BUFFER_SIZE = 1;

    std::signal(SIGINT, [](int){ run = false; });

    InferPipelineManager infer_manager(model_context,
                                       "/home/tom/carbot_inference/rollouts",
                                       "/home/tom/carbot_inference/config/config_preprocess.txt",
                                       "/home/tom/models/carbot_v1/outside_office.engine");
    if (!infer_manager.setup()) {
        std::cerr << "Failed to set up InferPipelineManager." << std::endl;
        return -1;
    }

    if (!infer_manager.startPipelineAsync()) {
        std::cerr << "Failed to start inference pipeline." << std::endl;
        return -1;
    }

    std::thread receive_thread([&infer_manager, &model_context]() {
        std::vector<float> model_output;
        while (infer_manager.getLatestResult(model_output)) {
            if (model_output.size() == model_context.OUTPUT_SIZE) {
                for (int i = 0; i < model_context.OUTPUT_SIZE; i+=2) {
                    std::cout << "Point " << i/2 << " x: " << model_output[i] << ", y: " << model_output[i+1] << std::endl;
                }
            } else {
                std::cerr << "Unexpected model output size: " << model_output.size() << std::endl;
            }
            std::cout << "-----------------" << std::endl;
        }
    });

    while (run) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    infer_manager.stopPipeline();
    
    if (receive_thread.joinable()) {
        receive_thread.join();
    }
    return 0;
}
