#include "UndiscretisePredict.hpp"



std::vector<float> UndiscretisePredict::undiscretiseNSample(std::vector<float> logits) {
    
    if (logits.size() != NUM_WAYPOINTS * NUM_BINS * 2) {
        return {};
    }

    std::vector<float> waypoints;
    waypoints.reserve(NUM_WAYPOINTS * 2);

    // inputs are structured as [y_logits, x_logits, y_logits, x_logits, ...]
    // Sampling, just taking the max. 
    for (int i = 0; i < NUM_WAYPOINTS; i++) {
        int offset = i * NUM_BINS * 2;
        int y_max_index = 0;
        int x_max_index = 0;

        for (int j = 0; j < NUM_BINS; j++) {
            if (logits[offset + j] > logits[offset + y_max_index]) {
                y_max_index = j;
            }

            if (logits[offset + NUM_BINS + j] > logits[offset + NUM_BINS + x_max_index]) {
                x_max_index = j;
            }
        }

        // Undiscretise, int -> bin center value.
        waypoints.push_back(x_bins[i][x_max_index]);
        waypoints.push_back(y_bins[i][y_max_index]);
    }
    return waypoints;
}