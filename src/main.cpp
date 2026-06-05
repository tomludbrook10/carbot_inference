#include "InferPipelineManager.hpp"
#include "Coms.hpp"
#include "UndiscretisePredict.hpp"

#include <csignal>
#include <chrono>
#include <thread>

static bool run = true;

int main() {

    int num_waypoints = 4;
    int context_buffer_size = 1;
    int num_bins = 19;

    UndiscretisePredict undiscretiser(num_waypoints, num_bins);

    ModelContext model_context(num_waypoints, context_buffer_size, num_bins);
    std::signal(SIGINT, [](int){ run = false; });

    InferPipelineManager infer_manager(model_context,
                                       "/home/tom/carbot_inference/rollouts",
                                       "/home/tom/carbot_inference/config/config_preprocess.txt",
                                       "/home/tom/models/carbot_v1/house_dist_fine_tune.engine");
    if (!infer_manager.setup()) {
        std::cerr << "Failed to set up InferPipelineManager." << std::endl;
        return -1;
    }

    if (!infer_manager.startPipelineAsync()) {
        std::cerr << "Failed to start inference pipeline." << std::endl;
        return -1;
    }

    std::thread receive_thread([&infer_manager, &model_context, &undiscretiser]() {
        std::vector<float> model_output;
        while (infer_manager.getLatestResult(model_output)) {
            std::cout << "Received model output of size: " << model_output.size() << std::endl;
            std::vector<float> waypoints = undiscretiser.undiscretiseNSample(model_output);
            std::cout << "Got waypoints:" << std::endl;

            if (waypoints.size() == model_context.NUM_WAYPOINTS * 2) {
                for (int i = 0; i < model_context.NUM_WAYPOINTS * 2; i+=2) {
                    std::cout << "Point " << i/2 << " x: " << waypoints[i] << ", y: " << waypoints[i+1] << std::endl;
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
