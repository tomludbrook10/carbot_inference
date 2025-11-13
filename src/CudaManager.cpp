#include "CudaManager.hpp"
#include "Coms.hpp"

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <nvdsmeta.h>
#include <nvdspreprocess_meta.h>
#include <gstnvdsmeta.h>
#include <gst/gst.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <memory>
#include <queue>

using namespace nvinfer1;

bool CudaManager::setup(const  std::string &model_path) {
    logger_ = std::make_unique<Logger>();

    runtime_.reset(createInferRuntime(*logger_));
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT Runtime." << std::endl;
        return false;
    }

    std::vector<char> model_data = readModelFromFile(model_path);
    if (model_data.empty()) {
        return false;
    }

    // for now in-memory deserialization is sufficient for smaller models.
    engine_.reset(runtime_->deserializeCudaEngine(model_data.data(), model_data.size()));
    if (!engine_) {
        std::cerr << "Failed to create TensorRT Engine." << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create Execution Context." << std::endl;
        return false;
    }


    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return false;
    }

    if (cudaMalloc(&output_buf_, model_context_.OUTPUT_SIZE * model_context_.SIZE_OF_DATA_IN_BYTES) != cudaSuccess) {
        std::cerr << "Failed to allocate output buffer on GPU for the output tensor." << std::endl;
        return false;
    }

    if (cudaMalloc(&input_buf_, 
        model_context_.INPUT_SIZE * model_context_.SIZE_OF_DATA_IN_BYTES * model_context_.CONTEXT_BUFFER_SIZE) 
        != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer on GPU for the input tensor." << std::endl;
        return false;
    }

    std::cout << "Successfully loaded model and created TensorRT engine." << std::endl;
    return true;
}


void CudaManager::infer(ProcessData* input_data, 
                        ResultData* result_data, 
                        const uint64_t MAX_INFER_TIME_NS,
                        uint64_t *base_time,
                        const std::string& save_path,
                        bool *running_) {
    size_t frame_num = 0;
    std::cout << "Waiting 25 seconds for camera to warm up..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(25));

    {
        std::lock_guard<std::mutex> lock_init(input_data->mtx);
        for (; !input_data->buffer_queue.empty(); input_data->buffer_queue.pop()) {
            GstBuffer *buf = input_data->buffer_queue.front();
            gst_buffer_unref(buf);
        }
    }

    std::cout << "Starting inference loop..." << std::endl;
    std::vector<CudaBuffer> cuda_buffers;

    while (*running_) {
        GstBuffer* current_buffer = nullptr;
        {
            std::unique_lock<std::mutex> lock(input_data->mtx);
            input_data->cv.wait(lock, [input_data, running_]() { return !input_data->buffer_queue.empty() || !*running_; });
            if (!*running_) {
                break;
            }

            current_buffer = input_data->buffer_queue.front();
            input_data->buffer_queue.pop();
        }

        if (!current_buffer) {
            std::cerr << "Pushed a nullptr Gst buffer" << std::endl;
            continue;
        }

        cuda_buffers.emplace_back(getCudaBuffer(current_buffer));

        if (cuda_buffers.back().device_ptr == nullptr) {
            std::cerr << "Failed to get CUDA buffer from GstBuffer." << std::endl;
            gst_buffer_unref(cuda_buffers.back().gst_buffer);
            cuda_buffers.pop_back();
            continue;
        }

        if (cuda_buffers.size() > model_context_.CONTEXT_BUFFER_SIZE) {
            gst_buffer_unref(cuda_buffers.front().gst_buffer);
            cuda_buffers.erase(cuda_buffers.begin());
        }

        if (cuda_buffers.size() < model_context_.CONTEXT_BUFFER_SIZE) {
            continue;
        }

        // buffer are ordered from oldest to newest. 
        for (size_t i = 0; i < model_context_.CONTEXT_BUFFER_SIZE; ++i) {
            void* dest_ptr = static_cast<char*>(input_buf_) + i * model_context_.INPUT_SIZE * model_context_.SIZE_OF_DATA_IN_BYTES;
            if (cudaMemcpy(dest_ptr, cuda_buffers[i].device_ptr, model_context_.INPUT_SIZE * model_context_.SIZE_OF_DATA_IN_BYTES, cudaMemcpyDeviceToDevice) != cudaSuccess) {
                std::cerr << "Failed to copy input tensor to context buffer." << std::endl;
            }
        }

        context_->setTensorAddress("input", input_buf_);
        context_->setTensorAddress("output", output_buf_);
        context_->enqueueV3(stream_);

           // for debugging. 
        if (!save_path.empty()) {
            // need to add a conditional here for float16. 
            std::vector<float> input_tensor(model_context_.INPUT_SIZE);
            cudaMemcpy(input_tensor.data(), cuda_buffers.back().device_ptr, model_context_.INPUT_SIZE * model_context_.SIZE_OF_DATA_IN_BYTES, cudaMemcpyDeviceToHost);

            std::string file_path = save_path + "/input_tensor_" + std::to_string(frame_num) + ".bin";
            std::ofstream ofs(file_path, std::ios::binary);
            if (ofs) {
                ofs.write(reinterpret_cast<const char *>(input_tensor.data()), model_context_.INPUT_SIZE * model_context_.SIZE_OF_DATA_IN_BYTES);
                ofs.close();
            } else {
                std::cerr << "Failed to open file for writing input tensor: " << file_path << std::endl;
            }
        }

        // current we only need run the model 200-400ish ms. so the blocking sync  and sync copy is fine. 
        // the model runs in aournd 15-20ms, fp32 on jetson orin nano 15v mode.
        cudaStreamSynchronize(stream_);
        {
            std::unique_lock<std::mutex> lock(result_data->mtx);
            if (result_data->output_data.capacity() < model_context_.OUTPUT_SIZE) {
                result_data->output_data.reserve(model_context_.OUTPUT_SIZE);
                std::cerr << "Warning: result_data output_data vector capacity has been modified from " << model_context_.OUTPUT_SIZE; 
                std::cerr << " reallocating..." << std::endl;
            }

            // else the size == 0, because we subvert the internal update with the direct memcpy.
            result_data->output_data.resize(model_context_.OUTPUT_SIZE);
            if (cudaMemcpy(result_data->output_data.data(), output_buf_, model_context_.OUTPUT_SIZE * model_context_.SIZE_OF_DATA_IN_BYTES, cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::cerr << "cuda Memcpy failed!" << std::endl;
                continue;
            }
        }

        result_data->cv.notify_one();

        // current this feature is not working, the solution is in task: Logging and Debugging: Camera to waypoint generation latency
        // GstClockTime pts = GST_BUFFER_PTS(buffer);
        // uint64_t pts_nano = static_cast<uint64_t>(pts) + *base_time;
        // std::cout << "Buffer PTS: " << pts << std::endl;
        // uint64_t pts_second = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        // std::cout << "Buffer created" << pts_nano << " ns, Inference completed at " << pts_second << " ns." << std::endl;

        // if (pts_second - pts_nano > MAX_INFER_TIME_NS) {
        //     std::cerr << "Warning: Inference time exceeded max limit. PTS diff (ns): " << (pts_second - pts_nano) << std::endl;
        // }

        frame_num++;
    }

    for (const auto& cuda_buf : cuda_buffers) {
        gst_buffer_unref(cuda_buf.gst_buffer);
    }
    // free cuda resources
    cudaFree(output_buf_);
    cudaFree(input_buf_);
    cudaStreamDestroy(stream_);
}

CudaBuffer CudaManager::getCudaBuffer(GstBuffer* gst_buffer) {

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
    if (!batch_meta) {
        std::cerr << "Failed to get batch meta from buffer." << std::endl;
        return CudaBuffer{gst_buffer, nullptr, 0};
    } 

    // note that batch_user_meta_list is just a GList of NvDsUserMeta, double linked list.
    for (NvDsMetaList *l_user = batch_meta->batch_user_meta_list; l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta *user_meta = static_cast<NvDsUserMeta *>(l_user->data);
        if (user_meta && user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META) {
            GstNvDsPreProcessBatchMeta *preprocess_meta = static_cast<GstNvDsPreProcessBatchMeta *>(user_meta->user_meta_data);

            if (!preprocess_meta || !preprocess_meta->tensor_meta->raw_tensor_buffer) {
                std::cerr << "Preprocess meta or tensor buffer is null." << std::endl;
                continue;
            }
            
            auto input_size = preprocess_meta->tensor_meta->buffer_size;
            if (input_size != model_context_.INPUT_SIZE * model_context_.SIZE_OF_DATA_IN_BYTES) {
                std::cerr << "Unexpected input tensor size: " << input_size << std::endl;
                continue;
            }
            return CudaBuffer{gst_buffer, preprocess_meta->tensor_meta->raw_tensor_buffer, input_size};
        }
    }
    return CudaBuffer{gst_buffer, nullptr, 0}; // Placeholder return, update as needed
}

// we could speed this up by memory mapping the file if needed.
// as cuda engine requires pointer and the size. 
std::vector<char> CudaManager::readModelFromFile(const std::string& model_path) {
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
