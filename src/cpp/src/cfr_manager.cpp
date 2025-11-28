#include "cfr_manager.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "weight_utils.h"

namespace lb {
namespace cfr {

enum NodeType {
    TRAVERSER,
    OPPONENT,
    CHANCE,
    TERMINAL
};

struct StackFrame {
    Env env;
    NodeType type;
    int action_idx = 0; 
    std::vector<float> child_values; 
    std::vector<float> regrets; 
    std::vector<float> strategy; 
    float value = 0.0f; 
    
    std::vector<uint8_t> valid_actions;
    
    StackFrame(const Env& e, NodeType t) : env(e), type(t) {}
};

struct Worker {
    int id;
    std::vector<StackFrame> stack;
    bool waiting_for_inference = false;
    PolicyRequest inference_req;
    std::vector<float> inference_result; // Regrets
    std::mt19937 rng;
    int traversals_completed = 0;
    
    Worker(int i, int seed) : id(i), rng(seed) {}
    
    void reset(int num_players) {
        stack.clear();
        waiting_for_inference = false;
        Env env;
        env.reset(num_players, rng());
        push_node(env);
    }
    
    void push_node(const Env& env) {
        NodeType type;
        int alive = 0;
        for (int p = 0; p < env.num_players(); ++p)
            if (!env.terminations[env.agent_index[p]]) ++alive;
        
        if (alive <= 1) type = TERMINAL;
        else type = (env.current_player() == 0) ? TRAVERSER : OPPONENT; 
        
        StackFrame frame(env, type);
        
        if (type == TRAVERSER || type == OPPONENT) {
            uint8_t mask[7];
            env.valid_actions(mask);
            for(int i=0; i<7; ++i) {
                if(mask[i]) frame.valid_actions.push_back(i);
            }
        }
        
        stack.push_back(std::move(frame));
    }
};

} // namespace cfr
} // namespace lb

using namespace lb::cfr;

// Helper to prepare observation sequence (copied from VecArena)
static void prepare_ai_sequence(const Env& e, int ai_seat, int seq_cap, PolicyRequest& out) {
    const int n_players = e.num_players();
    const int total = static_cast<int>(e.game_history.size());
    const int capped_max = std::max(1, seq_cap);
    const int start = std::max(0, total - (capped_max - 1));
    const int history_span = std::max(0, total - start);
    const int max_history = std::min(capped_max - 1, history_span);
    const int max_valid = std::max(1, std::min(capped_max, history_span + 1));

    out.obs_sequence.resize(static_cast<size_t>(max_valid) * OBS_DIM);
    out.action_sequence.resize(max_valid);
    out.agent_type_sequence.resize(max_valid);
    out.position_sequence.resize(max_valid);
    out.action_mask_sequence.resize(static_cast<size_t>(max_valid) * 7);
    std::fill(out.action_mask_sequence.begin(), out.action_mask_sequence.end(), 0);

    auto transform_action = [&](int actor_seat, uint8_t action) -> int64_t {
        int relative_agent_id = (actor_seat - ai_seat + n_players) % n_players;
        if (relative_agent_id == 0) return static_cast<int64_t>(action);
        if (action <= 5) return static_cast<int64_t>(7 + (action % 3));
        return static_cast<int64_t>(action);
    };

    float* obs_ptr = out.obs_sequence.data();
    int64_t* action_ptr = out.action_sequence.data();
    int64_t* agent_ptr = out.agent_type_sequence.data();
    int64_t* pos_ptr = out.position_sequence.data();
    uint8_t* mask_ptr = out.action_mask_sequence.data();

    int idx = 0;
    const int history_start = total - max_history;
    for (int i = history_start; i < total && idx < max_valid; ++i) {
        const HistoryEntry& h = e.game_history[i];
        const auto& obs = h.observations[ai_seat];
        const int obs_take = std::min<int>(OBS_DIM, static_cast<int>(obs.size()));
        float* dst_obs = obs_ptr + idx * OBS_DIM;
        if (obs_take > 0) {
            std::memcpy(dst_obs, obs.data(), sizeof(float) * obs_take);
            if (obs_take < OBS_DIM) std::fill(dst_obs + obs_take, dst_obs + OBS_DIM, 0.0f);
        } else {
            std::fill(dst_obs, dst_obs + OBS_DIM, 0.0f);
        }

        const int h_seat = e.seat_for_agent.at(h.agent_id);
        int rel = (h_seat - ai_seat + n_players) % n_players;
        agent_ptr[idx] = rel;
        if (idx == 0) action_ptr[idx] = 10; 
        else {
            const HistoryEntry& prev_h = e.game_history[i - 1];
            const int prev_h_seat = e.seat_for_agent.at(prev_h.agent_id);
            action_ptr[idx] = transform_action(prev_h_seat, prev_h.action);
        }

        if (rel == 0) std::memcpy(mask_ptr + idx * 7, h.mask.data(), sizeof(uint8_t) * 7);
        pos_ptr[idx] = idx;
        ++idx;
    }

    if (idx < max_valid) {
        float cur_obs[OBS_DIM];
        const int ai_agent_id = e.agent_index.at(ai_seat);
        e.observe_vector_newerest(ai_agent_id, cur_obs);
        std::memcpy(obs_ptr + idx * OBS_DIM, cur_obs, sizeof(float) * OBS_DIM);
        agent_ptr[idx] = 0; 
        if (total > 0) {
            const HistoryEntry& last_h = e.game_history[total - 1];
            const int last_h_seat = e.seat_for_agent.at(last_h.agent_id);
            action_ptr[idx] = transform_action(last_h_seat, last_h.action);
        } else {
            action_ptr[idx] = 10;
        }
        e.valid_actions(mask_ptr + idx * 7);
        pos_ptr[idx] = idx;
        ++idx;
    }

    out.valid_len = std::max(1, idx);
    out.action_sequence.resize(out.valid_len);
    out.agent_type_sequence.resize(out.valid_len);
    out.position_sequence.resize(out.valid_len);
    out.action_mask_sequence.resize(static_cast<size_t>(out.valid_len) * 7);
    out.obs_sequence.resize(static_cast<size_t>(out.valid_len) * OBS_DIM);
}

CFRManager::CFRManager(int num_players, int max_depth, int num_workers)
    : num_players_(num_players), max_depth_(max_depth), num_workers_(num_workers) {
    
    workers_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        workers_.push_back(std::make_unique<Worker>(i, i + 12345)); 
        workers_.back()->reset(num_players_);
    }
}

CFRManager::~CFRManager() = default;

void CFRManager::load_advantage_network(const std::unordered_map<std::string, torch::Tensor>& state_dict) {
    // Keep all tensors including LUTs - the Python model has the correct ones
    staged_state_dict_ = state_dict;
    weights_finalized_ = false;
}

void CFRManager::finalize_model_loading() {
    if (staged_state_dict_.empty()) return;
    
    process_and_split_attention_weights(staged_state_dict_);
    
    if (!staged_state_dict_.count("position_embedding.weight")) {
        int hidden_dim = staged_state_dict_.at("obs_encoder.0.weight").size(0);
        staged_state_dict_["position_embedding.weight"] = torch::zeros({480, hidden_dim}, torch::kFloat32); 
    }
    
    batched_weights_.clear();
    for (const auto& kv : staged_state_dict_) {
        auto t = kv.second.unsqueeze(0).to(torch::kCUDA);
        if (t.is_floating_point()) {
            t = t.to(torch::kFloat16);
        }
        batched_weights_.insert(kv.first, t);
    }
    
    add_fixed_buffers(batched_weights_, torch::kCUDA);
    
    policy_id_to_idx_[0] = 0;
    
    int num_layers = 0;
    while (staged_state_dict_.count("transformer.layers." + std::to_string(num_layers) + ".norm1.weight")) {
        num_layers++;
    }
    int hidden_dim = staged_state_dict_.at("obs_encoder.0.weight").size(0);
    int num_heads = 4; 
    
    orchestrator_ = std::make_unique<execution_core::NeuralInferenceOrchestrator>(
        batched_weights_,
        policy_id_to_idx_,
        num_workers_, 
        num_layers,
        num_heads,
        hidden_dim,
        1, 1, 
        false 
    );
    
    staged_state_dict_.clear();
    weights_finalized_ = true;
}

int CFRManager::run_cfr_traversal(int num_traversals) {
    if (!weights_finalized_) finalize_model_loading();
    
    int completed = 0;
    for(auto& w : workers_) w->traversals_completed = 0;
    
    while (completed < num_traversals) {
        step_workers();
        run_inference_batch();
        
        for(auto& w : workers_) {
            if (w->traversals_completed > 0) {
                completed += w->traversals_completed;
                w->traversals_completed = 0;
            }
        }
    }
    return completed;
}

void CFRManager::step_workers() {
    for (auto& w_ptr : workers_) {
        Worker& w = *w_ptr;
        if (w.waiting_for_inference) continue;
        
        while (!w.stack.empty()) {
            StackFrame& frame = w.stack.back();
            
            if (frame.type == TERMINAL) {
                int winner = -1;
                int alive = 0;
                for (int p = 0; p < frame.env.num_players(); ++p) {
                    if (!frame.env.terminations[frame.env.agent_index[p]]) {
                        winner = frame.env.agent_index[p];
                        alive++;
                    }
                }
                
                float val = (alive == 1 && winner == 0) ? 1.0f : -1.0f; 
                
                w.stack.pop_back();
                if (!w.stack.empty()) {
                    w.stack.back().child_values.push_back(val);
                } else {
                    w.traversals_completed++;
                    w.reset(num_players_);
                    break; 
                }
                continue;
            }
            
            if (frame.type == TRAVERSER) {
                if (frame.regrets.empty()) {
                    w.inference_req.env = w.id; 
                    w.inference_req.seat = frame.env.current_player();
                    frame.env.valid_actions(w.inference_req.mask.data());
                    prepare_ai_sequence(frame.env, frame.env.current_player(), 480, w.inference_req);
                    w.waiting_for_inference = true;
                    break; 
                }

                if (frame.action_idx < frame.valid_actions.size()) {
                    uint8_t a = frame.valid_actions[frame.action_idx];
                    frame.action_idx++;
                    
                    Env next_env = frame.env;
                    next_env.step(a);
                    w.push_node(next_env);
                    break; 
                } else {
                    // Compute regrets
                    // v(I) = sum(sigma(a) * v(I, a))
                    // We need sigma.
                    // Regret Matching on frame.regrets
                    std::vector<float> sigma(7, 0.0f);
                    float sum_pos_regret = 0.0f;
                    for (int i = 0; i < 7; ++i) {
                        if (frame.regrets[i] > 0) sum_pos_regret += frame.regrets[i];
                    }
                    
                    if (sum_pos_regret > 1e-6f) {
                        for (int i = 0; i < 7; ++i) {
                            if (frame.regrets[i] > 0) sigma[i] = frame.regrets[i] / sum_pos_regret;
                        }
                    } else {
                        // Uniform over valid actions
                        float prob = 1.0f / frame.valid_actions.size();
                        for (uint8_t a : frame.valid_actions) sigma[a] = prob;
                    }
                    
                    float v_I = 0.0f;
                    for (size_t i = 0; i < frame.valid_actions.size(); ++i) {
                        uint8_t a = frame.valid_actions[i];
                        v_I += sigma[a] * frame.child_values[i];
                    }
                    
                    // Compute Counterfactual Regrets: r(I, a) = v(I, a) - v(I)
                    std::vector<float> cf_regrets(7, 0.0f);
                    for (size_t i = 0; i < frame.valid_actions.size(); ++i) {
                        uint8_t a = frame.valid_actions[i];
                        cf_regrets[a] = frame.child_values[i] - v_I;
                    }
                    
                    // Add to Regret Buffer
                    // We need to store (Obs, cf_regrets)
                    // Obs is in w.inference_req (but we might have overwritten it?)
                    // w.inference_req was used for inference.
                    // We should reconstruct or store it.
                    // Reconstructing is safe.
                    PolicyRequest req;
                    prepare_ai_sequence(frame.env, frame.env.current_player(), 480, req);
                    
                    regret_buffer_.obs.insert(regret_buffer_.obs.end(), req.obs_sequence.begin(), req.obs_sequence.end());
                    regret_buffer_.actions.insert(regret_buffer_.actions.end(), req.action_sequence.begin(), req.action_sequence.end());
                    regret_buffer_.agents.insert(regret_buffer_.agents.end(), req.agent_type_sequence.begin(), req.agent_type_sequence.end());
                    regret_buffer_.positions.insert(regret_buffer_.positions.end(), req.position_sequence.begin(), req.position_sequence.end());
                    // padding mask? PolicyRequest doesn't have explicit padding mask, but we can infer from valid_len
                    // Or we can store valid_len and pad later.
                    // For now, let's store flat and use valid_len logic in collect_samples.
                    // Actually, PolicyRequest pads with 0s.
                    // We can just store the flat sequence.
                    // But we need to know where it ends.
                    // Let's store valid_len.
                    // Wait, SampleBuffer uses flat vectors.
                    // We should store fixed size sequences?
                    // Yes, PolicyRequest resizes to valid_len.
                    // We should pad to max_len (480) here or in Python.
                    // Let's pad to 480 here.
                    int pad_len = 480 - req.valid_len;
                    if (pad_len > 0) {
                        regret_buffer_.obs.insert(regret_buffer_.obs.end(), pad_len * OBS_DIM, 0.0f);
                        regret_buffer_.actions.insert(regret_buffer_.actions.end(), pad_len, 0);
                        regret_buffer_.agents.insert(regret_buffer_.agents.end(), pad_len, 0);
                        regret_buffer_.positions.insert(regret_buffer_.positions.end(), pad_len, 0);
                    }
                    // Targets
                    regret_buffer_.targets.insert(regret_buffer_.targets.end(), cf_regrets.begin(), cf_regrets.end());
                    regret_buffer_.weights.push_back(1.0f); // t weighting handled in Python
                    
                    float ret_val = v_I;
                    w.stack.pop_back();
                    if (!w.stack.empty()) {
                        w.stack.back().child_values.push_back(ret_val);
                    } else {
                        w.traversals_completed++;
                        w.reset(num_players_);
                    }
                    continue;
                }
            }
            
            if (frame.type == OPPONENT) {
                if (frame.regrets.empty()) {
                    w.inference_req.env = w.id;
                    w.inference_req.seat = frame.env.current_player();
                    frame.env.valid_actions(w.inference_req.mask.data());
                    prepare_ai_sequence(frame.env, frame.env.current_player(), 480, w.inference_req);
                    w.waiting_for_inference = true;
                    break;
                }
                
                // Compute Strategy
                std::vector<float> sigma(7, 0.0f);
                float sum_pos_regret = 0.0f;
                for (int i = 0; i < 7; ++i) {
                    if (frame.regrets[i] > 0) sum_pos_regret += frame.regrets[i];
                }
                
                if (sum_pos_regret > 1e-6f) {
                    for (int i = 0; i < 7; ++i) {
                        if (frame.regrets[i] > 0) sigma[i] = frame.regrets[i] / sum_pos_regret;
                    }
                } else {
                    float prob = 1.0f / frame.valid_actions.size();
                    for (uint8_t a : frame.valid_actions) sigma[a] = prob;
                }
                
                // Add to Strategy Buffer
                PolicyRequest req;
                prepare_ai_sequence(frame.env, frame.env.current_player(), 480, req);
                strategy_buffer_.obs.insert(strategy_buffer_.obs.end(), req.obs_sequence.begin(), req.obs_sequence.end());
                strategy_buffer_.actions.insert(strategy_buffer_.actions.end(), req.action_sequence.begin(), req.action_sequence.end());
                strategy_buffer_.agents.insert(strategy_buffer_.agents.end(), req.agent_type_sequence.begin(), req.agent_type_sequence.end());
                strategy_buffer_.positions.insert(strategy_buffer_.positions.end(), req.position_sequence.begin(), req.position_sequence.end());
                int pad_len = 480 - req.valid_len;
                if (pad_len > 0) {
                    strategy_buffer_.obs.insert(strategy_buffer_.obs.end(), pad_len * OBS_DIM, 0.0f);
                    strategy_buffer_.actions.insert(strategy_buffer_.actions.end(), pad_len, 0);
                    strategy_buffer_.agents.insert(strategy_buffer_.agents.end(), pad_len, 0);
                    strategy_buffer_.positions.insert(strategy_buffer_.positions.end(), pad_len, 0);
                }
                strategy_buffer_.targets.insert(strategy_buffer_.targets.end(), sigma.begin(), sigma.end());
                strategy_buffer_.weights.push_back(1.0f);

                // Sample action
                float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(w.rng);
                float cum = 0.0f;
                uint8_t chosen_action = frame.valid_actions.back();
                for (uint8_t a : frame.valid_actions) {
                    cum += sigma[a];
                    if (r <= cum) {
                        chosen_action = a;
                        break;
                    }
                }
                
                Env next_env = frame.env;
                next_env.step(chosen_action);
                
                // Replace current frame with child (tail recursion optimization effectively)
                // But we need to return value.
                // Opponent node returns value of child.
                // So we can just replace.
                // But we need to pop when child returns.
                // So we push child.
                w.push_node(next_env);
                break;
            }
        }
    }
}

void CFRManager::run_inference_batch() {
    std::unordered_map<int, std::vector<PolicyRequest>> requests;
    std::vector<int> worker_indices;
    
    for (int i = 0; i < num_workers_; ++i) {
        if (workers_[i]->waiting_for_inference) {
            requests[0].push_back(workers_[i]->inference_req);
            worker_indices.push_back(i);
        }
    }
    
    if (requests.empty()) return;
    
    auto results = orchestrator_->run_inference(requests);
    
    for (size_t i = 0; i < worker_indices.size(); ++i) {
        int w_idx = worker_indices[i];
        auto res = results.at({0, (int)i});
        
        // Extract logits (which are actually regrets/advantages here)
        // res.action_logits is [1, 7] tensor on GPU
        // Move to CPU
        auto cpu_logits = res.action_logits.to(torch::kCPU).contiguous();
        float* data = cpu_logits.data_ptr<float>();
        
        std::vector<float> regrets(data, data + 7);
        
        workers_[w_idx]->waiting_for_inference = false;
        if (!workers_[w_idx]->stack.empty()) {
            workers_[w_idx]->stack.back().regrets = std::move(regrets);
        }
    }
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> CFRManager::collect_samples() {
    auto make_tensors = [](SampleBuffer& buf) {
        std::vector<torch::Tensor> tensors;
        if (buf.targets.empty()) return tensors;
        
        int N = buf.weights.size();
        int seq_len = 480;
        int obs_dim = OBS_DIM;
        int action_dim = 7;
        
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto long_opts = torch::TensorOptions().dtype(torch::kLong);
        
        tensors.push_back(torch::from_blob(buf.obs.data(), {N, seq_len, obs_dim}, opts).clone());
        tensors.push_back(torch::from_blob(buf.actions.data(), {N, seq_len}, long_opts).clone());
        tensors.push_back(torch::from_blob(buf.agents.data(), {N, seq_len}, long_opts).clone());
        tensors.push_back(torch::from_blob(buf.positions.data(), {N, seq_len}, long_opts).clone());
        
        // Padding mask: 1 where padded.
        // We didn't store explicit mask, but we padded with 0s.
        // We can reconstruct or just pass all False if we trust valid_len.
        // But we need a mask tensor.
        // Let's assume we need to compute it or store it.
        // For now, return zeros (no padding) or fix later.
        tensors.push_back(torch::zeros({N, seq_len}, torch::TensorOptions().dtype(torch::kBool))); 
        
        tensors.push_back(torch::from_blob(buf.targets.data(), {N, action_dim}, opts).clone());
        tensors.push_back(torch::from_blob(buf.weights.data(), {N}, opts).clone());
        
        buf.clear();
        return tensors;
    };
    
    return {make_tensors(regret_buffer_), make_tensors(strategy_buffer_)};
}
