#pragma once

struct Role {
    int policy_id{ -1 };
    // Removed trajectory_id - we use agent_index (the map key) instead!
};
