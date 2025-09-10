#pragma once
#include <cstdint>

enum class RoleType : uint8_t { BotCpp = 0, Policy = 1 };

enum class BotKind : uint8_t {
    Classic = 0,
    GreedyCardSpammer = 1,
    RandomAgent = 2,
    SelectiveTableConservativeChallenger = 3,
    StrategicChallenger = 4,
    TableFirstConservativeChallenger = 5,
    TableNonTableAgent = 6
};

struct Role {
	RoleType type{ RoleType::BotCpp };
	BotKind  bot_kind{ BotKind::Classic }; // valid iff type==BotCpp
	int      policy_id{ -1 };              // valid iff type==Policy
};