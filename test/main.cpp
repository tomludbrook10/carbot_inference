#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <iostream>
#include <nvdsmeta.h>
#include "nvdspreprocess_meta.h"

#include <NvInfer.h>
#include <gstnvdsmeta.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <thread>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <memory>

#define SURFACE_MEMORY 0 
#define GPU 1
#define MODEL_PATH "/home/tom/models/carbot_v1/modelv1.engine"

struct ProcessData {
    std::queue<GstBuffer *> buffer_queue;
    std::mutex mtx;
    std::condition_variable cv;
};

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

class CudaManager {
public:

    bool isRunning() const { return running_; }

    bool setup() {
        runtime_.reset(createInferRuntime(logger));
        if (!runtime_) {
            std::cerr << "Failed to create TensorRT Runtime." << std::endl;
            return false;
        }

        std::vector<char> model_data = readModelFromFile(MODEL_PATH);
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

        output_size_ = 16; // [1, 8, 2]
        if (cudaMalloc(&output_buf_, output_size_ * sizeof(float)) != cudaSuccess) {
            std::cerr << "Failed to allocate output buffer on GPU for the output tensor." << std::endl;
            return false;
        }
        return true;
    }

    void stop() { running_ = false; }

    void infer(ProcessData* data) {
        while (isRunning()) {
            GstBuffer *buffer = nullptr;
            {
                std::unique_lock<std::mutex> lock(data->mtx);
                data->cv.wait(lock, [data, this]() { return !data->buffer_queue.empty() || !isRunning(); });

                buffer = data->buffer_queue.front();
                data->buffer_queue.pop();
            }

            if (!buffer) {
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
                    if (preprocess_meta && preprocess_meta->tensor_meta->raw_tensor_buffer) {
                        context_->setTensorAddress("input", preprocess_meta->tensor_meta->raw_tensor_buffer);
                        context_->setTensorAddress("output", output_buf_);
                        context_->enqueueV3(stream_);
                        cudaStreamSynchronize(stream_);

                        std::vector<float> output_data(output_size_);
                        if (cudaMemcpy(output_data.data(), output_buf_, output_size_ * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                            std::cerr << "cuda Memcpy failed!" << std::endl;
                        } else {
                            for (const auto &val : output_data) {
                                std::cout << val << " ";
                            }
                            std::cout << std::endl;
                        }
                    } else {
                        std::cerr << "Preprocess meta or tensor buffer is null." << std::endl;
                    }
                }
            }
            gst_buffer_unref(buffer);
        }
        cudaFree(output_buf_);
        cudaStreamDestroy(stream_);
    }

private:
    bool running_ = true;
    cudaStream_t stream_;
    cudaError_t err_;
    void* output_buf_;
    size_t output_size_; 
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // we could speed this up by memory mapping the file if needed.
    // as cuda engine requires pointer and the size. 
    std::vector<char> readModelFromFile(const std::string& modelPath) {
        // add the std::ios::ate flag to open at the end of the file to get the size
        std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "Failed to open model file: " << modelPath << std::endl;
            return {};
        }

        // reduce allocations. 
        const std::streamsize size = file.tellg();
        std::vector<char> buffer(size);

        // reset the pointer to the beginning of the file
        file.seekg(0, std::ios::beg); // std::ios::beg is the beginning of the file and 0 offset. 
        if (!file.read(buffer.data(), size)) {
            std::cerr << "Failed to read model file: " << modelPath << std::endl;
            return {};
        }
        return buffer;
    }
};

void writeFloatsBinary(const std::vector<float>& data, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    outFile.close();
}

static GstFlowReturn on_new_sample(GstElement *appsink, gpointer udata) {
    GstSample *sample;
    ProcessData *data = static_cast<ProcessData *>(udata);

    if (data == nullptr) {
        std::cerr << "Process data is null." << std::endl;
        return GST_FLOW_ERROR;
    }

    g_signal_emit_by_name(appsink, "pull-sample", &sample);
    if (!sample) {
        std::cerr << "Failed to pull sample from appsink." << std::endl;
        return GST_FLOW_ERROR;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        std::cerr << "Failed to get buffer from sample." << std::endl;
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    gst_buffer_ref(buffer);
    {
        std::unique_lock lock(data->mtx);
        data->buffer_queue.push(buffer);
        data->cv.notify_one();
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

int main() {
    GstElement *pipeline, *source, *network_preprocessor, *sink;
    GstElement *streammux;
    GstElement *camera_capsfilter;
    GstCaps *camera_caps;

    gst_init(nullptr, nullptr);

    pipeline = gst_pipeline_new("test-pipeline");
    source = gst_element_factory_make("nvarguscamerasrc", "source");
    streammux = gst_element_factory_make("nvstreammux", "streammux");
    network_preprocessor = gst_element_factory_make("nvdspreprocess", "network-preprocessor");
    sink = gst_element_factory_make("appsink", "sink");
    camera_capsfilter = gst_element_factory_make("capsfilter", "camera-capsfilter");

    if (!pipeline || !source || !streammux || !network_preprocessor || !sink) {
        g_printerr("Not all elements could be created.\n");
        return -1;
    }

    g_object_set(G_OBJECT(network_preprocessor), "config-file", "/home/tom/carbot_inference/config_preprocess.txt", nullptr);
    camera_caps = gst_caps_from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=1/1, format=NV12");
    g_object_set(G_OBJECT(camera_capsfilter), "caps", camera_caps, nullptr);
    gst_caps_unref(camera_caps);

    
    g_object_set(G_OBJECT(streammux), "width", 1920, "height", 1080, "batch-size", 1, nullptr);

    gst_bin_add_many(GST_BIN(pipeline), source, camera_capsfilter, streammux, network_preprocessor, sink, nullptr);
    gst_element_link_many(source, camera_capsfilter, nullptr);


    // https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html
    // we have to link the pads manually for streammux, by requesting a new pad from the muxer.
    GstPad *mux_sink0 = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *caps_src_pad = gst_element_get_static_pad(camera_capsfilter, "src");
    if (gst_pad_link(caps_src_pad, mux_sink0) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link capsfilter to streammux.\n");
        return -1;
    }

    gst_object_unref(caps_src_pad);
    gst_object_unref(mux_sink0);
    gst_element_link_many(streammux, network_preprocessor, sink, nullptr);

    std::unique_ptr<CudaManager> cuda_manager = std::make_unique<CudaManager>();
    if (!cuda_manager->setup()) {
        std::cerr << "Failed to set up CudaManager." << std::endl;
        gst_object_unref(pipeline);
        return -1;
    }

    ProcessData process_data;
    std::thread infer_thread(&CudaManager::infer, cuda_manager.get(), &process_data);

    g_object_set(G_OBJECT(sink), "emit-signals", TRUE, "sync", FALSE, nullptr);
    g_signal_connect(sink, "new-sample", G_CALLBACK(on_new_sample), &process_data);


    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    // Let the pipeline run for 5 seconds
    g_usleep(30 * G_USEC_PER_SEC);

    cuda_manager->stop();
    if (infer_thread.joinable()) {
        infer_thread.join();
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}