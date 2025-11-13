#pragma once

#include <gst/gst.h>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <vector>

struct ProcessData {
    std::queue<GstBuffer *> buffer_queue;
    std::mutex mtx;
    std::condition_variable cv;
};

struct ResultData {
    std::vector<float> output_data;
    std::mutex mtx;
    std::condition_variable cv;
};

struct CudaBuffer {
    GstBuffer* gst_buffer;
    void* device_ptr;
    size_t size;
};

struct ModelContext {
    int NUM_WAYPOINTS = 4;
    int CONTEXT_BUFFER_SIZE = 5; // current observation + 4 previous frames.
    int OUTPUT_SIZE = NUM_WAYPOINTS * 2;
    const int INPUT_SIZE = 3 * 256 * 256;
    const int SIZE_OF_DATA_IN_BYTES = 4; // float.
};

