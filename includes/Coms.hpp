#pragma once

#include <gst/gst.h>
#include <mutex>
#include <queue>
#include <condition_variable>

struct ProcessData {
    std::queue<GstBuffer *> buffer_queue;
    std::mutex mtx;
    std::condition_variable cv;
};

struct ResultData{ 
    std::vector<float> output_data;
    std::mutex mtx;
    std::condition_variable cv;
};