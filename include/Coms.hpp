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
    int NUM_WAYPOINTS;
    int CONTEXT_BUFFER_SIZE;
    int INPUT_WIDTH = 280;
    int INPUT_HEIGHT = 280;
    int CHANNELS = 3;
    int INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * CHANNELS;
    int OUTPUT_SIZE = NUM_WAYPOINTS * 2; // x,y for each waypoint.
    int SIZE_OF_DATA_IN_BYTES = 4; // float.
    int NUM_BINS = 19;

    ModelContext(const int num_waypoints = 6, const int context_buffer_size = 5,
                  const int num_bins = 19)
        : NUM_WAYPOINTS(num_waypoints), CONTEXT_BUFFER_SIZE(context_buffer_size), NUM_BINS(num_bins)
        {
            INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * CHANNELS;
            OUTPUT_SIZE = NUM_WAYPOINTS * 2 * NUM_BINS;
        }

};

