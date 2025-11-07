#include "CudaManager.hpp"
#include "Coms.hpp"

#include <iostream>
#include <fstream>


bool CudaManager::setup() {
    logger_ = std::make_unique<Logger>();

    runtime_.reset(createInferRuntime(logger_.get()));
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT Runtime." << std::endl;
        return false;
    }

    std::vector<char> model_data = readModelFromFile(model_path_);
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

    if (cudaMalloc(&output_buf_, output_size_ * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate output buffer on GPU for the output tensor." << std::endl;
        return false;
    }
    std::cout << "Successfully loaded model and created TensorRT engine." << std::endl;
    return true;
}


void CudaManager::infer(ProcessData* input_data, 
                        ResultData* result_data, 
                        const int INPUT_SIZE, 
                        const int OUTPUT_SIZE, 
                        const int SIZE_OF_DATA_IN_BYTES,
                        bool *running_) {
    while (*running_) {
        GstBuffer *buffer = nullptr;
        {
            std::unique_lock<std::mutex> lock(input_data->mtx);
            input_data->cv.wait(lock, [input_data, running_]() { return !input_data->buffer_queue.empty() || !*running_; });
            if (!*running_) {
                break;
            }

            buffer = input_data->buffer_queue.front();
            input_data->buffer_queue.pop();
        }

        if (!buffer) {
            std::cerr << "Pushed a nullptr Gst buffer" << std::endl;
            continue;
        }

        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
        if (!batch_meta) {
            std::cerr << "Failed to get batch meta from buffer." << std::endl;
            gst_buffer_unref(buffer);
            continue;
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
                if (input_size != INPUT_SIZE * SIZE_OF_DATA_IN_BYTES) {
                    std::cerr << "Unexpected input tensor size: " << input_size << std::endl;
                    continue;
                }

                context_->setTensorAddress("input", preprocess_meta->tensor_meta->raw_tensor_buffer);
                context_->setTensorAddress("output", output_buf_);
                context_->enqueueV3(stream_);

                // current we only need run the model 200-400ish ms. so the blocking sync  and sync copy is fine. 
                // the model runs in aournd 15-20ms, fp32 on jetson orin nano 15v mode.
                cudaStreamSynchronize(stream_);
                {
                    std::unique_lock<std::mutex> lock(result_data->mtx);
                    if (result_data->output_data.capacity() < OUTPUT_SIZE) {
                        result_data->output_data.reserve(OUTPUT_SIZE);
                        std::cerr << "Warning: result_data output_data vector capacity has been modified from " << OUTPUT_SIZE; 
                        std::cerr << " reallocating..." << std::endl;
                    }

                    if (cudaMemcpy(result_data->output_data.data(), output_buf_, OUTPUT_SIZE * SIZE_OF_DATA_IN_BYTES, cudaMemcpyDeviceToHost) != cudaSuccess) {
                        std::cerr << "cuda Memcpy failed!" << std::endl;
                        continue;
                    }
                    result_data->cv.notify_one();
                }
            } 
        }

        // important to unref, we reffed when pushing the queue, to ensure the DS/Gst pipline backend doesn't free it too early.
        // I'm actually, not sure if it's needed but better safe than sorry.
        gst_buffer_unref(buffer);
    }
    // free cuda resources
    cudaFree(output_buf_);
    cudaStreamDestroy(stream_);
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