#pragma once

struct Role {
    int policy_id{ -1 };
    int trajectory_id{ -1 };  // Stable unique ID for training trajectory (-1 for bots/historical)
};
