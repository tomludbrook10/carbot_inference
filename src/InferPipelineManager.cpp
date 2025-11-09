#include "ProcessData.hpp"
#include "InferPipelineManager.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <iostream>
#include <string>

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

InferPipelineManager::InferPipelineManager(const std::string& config_file_path)
    : pipeline_(nullptr), config_file_path_(config_file_path) {}

bool InferPipelineManager::setup() {
    GstElement *source, *network_preprocessor, *sink;
    GstElement *streammux;
    GstElement *camera_capsfilter;
    GstCaps *camera_caps;

    gst_init(nullptr, nullptr);

    pipeline_ = gst_pipeline_new("infer-pipeline");

    if (!pipeline_) {
        g_printerr("Failed to create GST pipeline.\n");
        return false;
    }

    source = gst_element_factory_make("nvarguscamerasrc", "source");
    streammux = gst_element_factory_make("nvstreammux", "streammux");
    network_preprocessor = gst_element_factory_make("nvdspreprocess", "network-preprocessor");
    sink = gst_element_factory_make("appsink", "sink");
    camera_capsfilter = gst_element_factory_make("capsfilter", "camera-capsfilter");

    if (!source || !streammux || !network_preprocessor || !sink || !camera_capsfilter) {
        g_printerr("Not all elements could be created.\n");
        gst_object_unref(pipeline_);
        return false;
    }

    // camera caps. 
    camera_caps = gst_caps_from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=1/1, format=NV12");
    g_object_set(G_OBJECT(camera_capsfilter), "caps", camera_caps, nullptr);
    gst_caps_unref(camera_caps);

    g_object_set(G_OBJECT(network_preprocessor), "config-file", config_file_path_.c_str(), nullptr);
    g_object_set(G_OBJECT(streammux), "width", WIDTH, "height", HEIGHT, "batch-size", 1, nullptr);

    gst_bin_add_many(GST_BIN(pipeline_), source, camera_capsfilter, streammux, network_preprocessor, sink, nullptr);


    // https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html
    // we have to link the pads manually for streammux, by requesting a new pad from the muxer.
    GstPad *mux_sink0 = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *caps_src_pad = gst_element_get_static_pad(camera_capsfilter, "src");
    if (gst_pad_link(caps_src_pad, mux_sink0) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link capsfilter to streammux.\n");
        return false;
    }
    gst_object_unref(caps_src_pad);
    gst_object_unref(mux_sink0);

    if (!gst_element_link_many(streammux, network_preprocessor, sink, nullptr)) {
        g_printerr("Elements could not be linked.\n");
        gst_object_unref(pipeline_);
        return false;
    }

    process_data_ = std::make_unique<ProcessData>();

    g_object_set(G_OBJECT(sink), "emit-signals", true, "sync", false, nullptr);
    g_signal_connect(sink, "new-sample", G_CALLBACK(on_new_sample), process_data_.get());

    std::cout << "GST Infer pipeline setup completed." << std::endl;

    cuda_manager_ = std::make_unique<CudaManager>();
    if (!cuda_manager_->setup(model_path_)) {
        std::cerr << "Failed to set up CudaManager." << std::endl;
        gst_object_unref(pipeline_);
        return false;
    }

    result_data_ =  std::make_unique<ResultData>();
    result_data_->output_data.reserve(OUTPUT_SIZE);
    return true;
}

void InferPipelineManager::start_pipeline_async() {
    if (!pipeline_) {
        std::cerr << "Pipeline is not set up. Call setup() before starting the pipeline." << std::endl;
        return;
    }


    gst_pipeline_thread_ = std::thread(&InferPipelineManager::start_pipeline, this);
    cuda_infer_thread_ = std::thread(&CudaManager::infer, 
                                    cuda_manager_.get(), 
                                    process_data_.get(), 
                                    result_data_.get(), 
                                    INPUT_SIZE, 
                                    OUTPUT_SIZE, 
                                    SIZE_OF_DATA_IN_BYTES);
}

void InferPipelineManager::stop_pipeline() { 
    // first stop the gst pipeline.
    gst_element_send_event(pipeline_, gst_event_new_eos());
    if (gst_pipeline_thread_.joinable()) {
        gst_pipeline_thread_.join();
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }

    cuda_manager_running_ = false;
    if (cuda_infer_thread_.joinable()) {
        cuda_infer_thread_.join();
    }

    std::cout << "Inference pipeline stopped." << std::endl;
}


void InferPipelineManager::start_pipeline() {
    GstBus *bus;
    GstMessage *msg;

    auto ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Unable to set the pipeline to the playing state.\n");
        return;
    }

    bus = gst_element_get_bus(pipeline_);
    msg = gst_bus_timed_pop_filtered(bus, 
                                    GST_CLOCK_TIME_NONE,
                                    static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    if (msg != NULL) {
        GError *err;
        gchar *dbg;
        switch (GST_MESSAGE_TYPE (msg)) {
            case GST_MESSAGE_ERROR:
                gst_message_parse_error (msg, &err, &dbg);
                g_printerr ("Error received from element %s: %s\n",
                            GST_OBJECT_NAME (msg->src), err->message);
                g_error_free (err);
                g_free (dbg);
                break;
            case GST_MESSAGE_EOS:
                g_print ("End‐of‐Stream reached.\n");
                break;
            default:
                /* Unexpected */
                break;
        }
        gst_message_unref (msg);
    }

    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(bus);
    std::cout << "GST Infer pipeline stopped." << std::endl;
}

void InferPipelineManager::getLatestResult(std::vector<float>& output_data) {
    std::unique_lock<std::mutex> lock(result_data_->mtx);
    result_data_->cv.wait(lock, [this]() { return !result_data_->output_data.empty(); || !cuda_manager_running_; });
    if (!cuda_manager_running_) {
        output_data.clear();
        return;
    }
    // since the floats are trivially copyable, this is as fast memcpy.
    output_data = result_data_->output_data;
    result_data_->output_data.clear();
}



