#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <torch/torch.h>

#include "bare_env.h"
#include "vec_arena.h"
#include "execution_core.h"

// Forward declarations
namespace lb {
    namespace cfr {
        struct StackFrame;
        struct Worker;
    }
}

class CFRManager {
public:
    CFRManager(int num_players, int max_depth = 100, int num_workers = 1024);
    ~CFRManager();

    // Load the Advantage Network (used for Opponent nodes)
    void load_advantage_network(const std::unordered_map<std::string, torch::Tensor>& state_dict);
    void finalize_model_loading();

    // Run CFR traversals (External Sampling)
    // Returns number of nodes visited or some stat
    int run_cfr_traversal(int num_traversals);

    // Collect generated samples from buffers
    // Returns tuple of tensors: (obs, act, agent, pos, mask, target, weight)
    // One for Regret (Advantage) buffer, one for Strategy (Policy) buffer
    std::tuple<
        std::vector<torch::Tensor>, // Regret samples
        std::vector<torch::Tensor>  // Strategy samples
    > collect_samples();

private:
    int num_players_;
    int max_depth_;
    int num_workers_;
    
    // Workers state
    std::vector<std::unique_ptr<lb::cfr::Worker>> workers_;
    
    // Model (Advantage Network)
    std::unique_ptr<execution_core::NeuralInferenceOrchestrator> orchestrator_;
    
    // Buffers (C++ side storage before transfer to Python)
    // We store flattened data to minimize overhead
    struct SampleBuffer {
        std::vector<float> obs;
        std::vector<int64_t> actions; // action sequence
        std::vector<int64_t> agents;
        std::vector<int64_t> positions;
        std::vector<int64_t> padding_mask; // 0 or 1
        std::vector<float> targets; // Regret or Strategy
        std::vector<float> weights;
        
        void clear() {
            obs.clear();
            actions.clear();
            agents.clear();
            positions.clear();
            padding_mask.clear();
            targets.clear();
            weights.clear();
        }
    };
    
    SampleBuffer regret_buffer_;
    SampleBuffer strategy_buffer_;
    
    // Model loading state (reused from EvalManager logic roughly)
    std::unordered_map<std::string, torch::Tensor> staged_state_dict_;
    c10::Dict<std::string, torch::Tensor> batched_weights_;
    std::unordered_map<int, int> policy_id_to_idx_; // Always 0 for single model
    bool weights_finalized_ = false;
    
    // Helpers
    void step_workers();
    void run_inference_batch();
};
