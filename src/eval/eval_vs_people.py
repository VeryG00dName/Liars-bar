
import os
import time
import json
import socket
import threading
import copy
import torch
import numpy as np
import logging
import pyautogui
from typing import Dict, Any, Optional, List, Tuple

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.agents.agent_factory import AgentFactory
from src.model.memory import delete_opponent_memory
from src.env.liars_deck_env_utils_2 import decode_action
from pettingzoo.utils import agent_selector

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AR_Game_Interface")

# Card mapping between game and environment
GAME_TO_ENV_CARD = {
    1: "King",    # King
    2: "Queen",   # Queen
    3: "Ace",     # Ace
    4: "Joker",   # Joker
}

# Map from human-readable to internal card representation
CARD_TO_INT = {
    "King": 0,
    "Queen": 1,
    "Ace": 2,
    "Joker": "Joker"
}

# Command delay in seconds
COMMAND_DELAY = 0.3

class ARGameInterface:
    """
    Interface to enable an Autoregressive AI agent to play Liar's Deck against
    real human players by connecting to the game via a mod.
    """
    def __init__(self, checkpoint_path: str, device: torch.device, host='127.0.0.1', port=5005):
        # Initialize environment
        self.env = LiarsDeckEnv(num_players=3)
        
        # Initialize AI agent
        self.agent_factory = AgentFactory(device)
        self.ai_agent = self.agent_factory.create_agent_from_checkpoint(
            checkpoint_path=checkpoint_path,
            player_id_prefix="AR_Player",
            agent_key="player_0"  # Key in checkpoint - you might need to adjust this
        )
        
        # Initialize game state tracking
        self.current_game_state = None
        self.previous_game_state = None
        self.ai_player_name = "VeryGoodName"  # The name of the AI player in the game (same as in game_controller)
        self.player_name_to_env_id = {}  # Will map real player names to environment IDs
        self.env_id_to_player_name = {}  # Reverse mapping
        self.player_positions = []  # List of player names in turn order
        self.host = host
        self.port = port
        self.state_lock = threading.Lock()
        self.last_hands = {}  # To track changes in hands
        self.ai_needs_to_act = False
        self.last_claimed_cards = []
        self.claimed_card_count = 0
        self.is_first_turn_in_round = True
        self.ai_turn_triggered = False
        # Initialize communication
        self.socket_thread = None
        self.running = True
        
        logger.info(f"AR Game Interface initialized with AI agent from {checkpoint_path}")
    
    def start(self):
        """Start the game interface including socket listener and main loop."""
        # Start socket listener in a thread
        self.socket_thread = threading.Thread(target=self.state_listener, daemon=True)
        self.socket_thread.start()
        
        # Start main loop
        self.main_loop()
    
    def state_listener(self):
        """Listen for game state from the mod via socket connection."""
        logger.info(f"Starting state listener on {self.host}:{self.port}")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    with client_socket:
                        data = client_socket.recv(8192)
                        if not data:
                            continue
                        
                        raw_json = data.decode('utf-8-sig').strip()
                        parsed = json.loads(raw_json)
                        
                        # Update the shared game state
                        with self.state_lock:
                            self.previous_game_state = self.current_game_state
                            self.current_game_state = parsed
                            self.process_new_game_state()
                except Exception as e:
                    logger.error(f"Error in state listener: {e}", exc_info=True)
    
    def process_new_game_state(self):
        """Process a new game state and update the environment."""
        if self.current_game_state is None:
            return
        
        # Check if state is meaningful
        if not self.is_valid_game_state(self.current_game_state):
            return
            
        # If this is the first valid state, initialize player mappings
        if not self.player_name_to_env_id:
            self.initialize_player_mappings()
        
        # Check if it's the AI's turn based on better detection logic
        just_became_our_turn = self.is_ai_turn_simple() and not self.ai_turn_triggered

        # Determine if it's the start of a new round (all players have full hands)
        is_new_round, challenger_id, penalized_player_id = self.detect_new_round()
        
        if is_new_round:
            self.is_first_turn_in_round = True
            logger.info("Detected new round")
            
            if challenger_id:
                # First, simulate the challenge action that ended the previous round
                logger.info(f"Simulating challenge by {self.env_id_to_player_name.get(challenger_id, challenger_id)}")
                
                # Reset the environment but don't reset the agent yet
                self.env.reset()
                
                # Update environment state to match before the challenge
                self.update_env_pre_challenge_state()
                
                # Now simulate the challenge action
                if challenger_id in self.env.agents:
                    # Make sure the challenger is the current agent
                    self.env.agents = [challenger_id] + [a for a in self.env.agents if a != challenger_id]
                    self.env._agent_selector = agent_selector(self.env.agents)
                    self.env.agent_selection = self.env._agent_selector.next()
                    
                    # Simulate the challenge action
                    self.env.step(6)  # Challenge action
                    logger.info(f"Simulated challenge (action 6) by {challenger_id}")
            
            # Now reset the agent for the new round
            self.ai_agent.reset()
            
            # Update env state for the new round
            self.update_env_for_new_round(penalized_player_id)
        else:
            # Update the environment state based on the current game state
            self.update_env_state()
        
        # Determine if AI needs to act
        if just_became_our_turn:
            logger.info("It's the AI's turn, preparing to act")
            print("\n" + "="*50)
            print("AI TURN DETECTED - PREPARING TO ACT")
            print("="*50 + "\n")
            self.ai_needs_to_act = True
            self.ai_turn_triggered = True
        else:
            self.ai_needs_to_act = False
            # reset for next cycle as soon as it's no longer our turn
            if not self.is_ai_turn_simple():
                self.ai_turn_triggered = False
    
    def get_game_state_dump(self):
        """Returns a human-readable dump of the current game state for debugging."""
        if not self.current_game_state:
            return "No game state available"
        
        output = []
        output.append("\n=== GAME STATE DUMP ===")
        
        # Game type
        game_type = self.current_game_state.get('game_type', 'Unknown')
        output.append(f"Game Type: {game_type}")
        
        # Round card
        round_card = self.get_round_card()
        output.append(f"Round Card: {round_card} ({GAME_TO_ENV_CARD.get(round_card, 'Unknown')})")
        
        # Action possibilities
        can_play = self.current_game_state.get('can_play', False)
        can_challenge = self.current_game_state.get('can_challenge', False)
        output.append(f"Can Play: {can_play}, Can Challenge: {can_challenge}")
        
        # Last played
        last_round = self.current_game_state.get('last_round', {})
        player = last_round.get('player', 'None')
        cards = last_round.get('cards', [])
        output.append(f"Last played by: {player}, Cards: {cards}")
        
        # Players
        output.append("\nPlayers:")
        hands = self.current_game_state.get('hands', {})
        for player_name, player_info in hands.items():
            # Handle both formats
            if isinstance(player_info, dict):
                cards = player_info.get('cards', [])
                is_dead = player_info.get('is_dead', False)
                bullet_pos = player_info.get('bullet_position', -1)
                shots_fired = player_info.get('shots_fired', -1)
                is_my_turn = player_info.get('is_my_turn', False)
                active_card = player_info.get('active_card_index', -1)
                status = "DEAD" if is_dead else "ALIVE"
                turn = "-> MY TURN" if is_my_turn else ""
                output.append(f"  {player_name} [{status}] {turn}")
                output.append(f"    Cards: {cards}")
                output.append(f"    Bullet Pos: {bullet_pos}, Shots Fired: {shots_fired}")
                output.append(f"    Active Card: {active_card}")
            else:
                # Simple format
                cards = player_info
                output.append(f"  {player_name}")
                output.append(f"    Cards: {cards}")
        
        # Environment state
        if self.env and hasattr(self.env, 'agent_selection'):
            output.append("\nEnvironment State:")
            output.append(f"  Current Agent: {self.env.agent_selection}")
            output.append(f"  Agents Order: {self.env.agents}")
            output.append(f"  Table Card: {self.env.table_card}")
            
            # Add player mappings
            output.append("\nPlayer Mappings:")
            for player_name, env_id in self.player_name_to_env_id.items():
                output.append(f"  {player_name} -> {env_id}")
        
        output.append("=== END DUMP ===\n")
        return "\n".join(output)

    def dump_game_state(self):
        """Print the current game state for debugging."""
        dump = self.get_game_state_dump()
        print(dump)
        return dump
    def update_env_pre_challenge_state(self):
        """Update environment state to match the state right before a challenge."""
        # Create a new state dictionary for the environment
        state_dict = self.env.get_state()
        
        # Use previous_game_state to set the state before the challenge
        if not self.previous_game_state:
            logger.warning("No previous game state available for pre-challenge setup")
            return
            
        # Update player hands from previous state
        hands = self.previous_game_state.get('hands', {})
        for player_name, player_info in hands.items():
            if player_name in self.player_name_to_env_id:
                env_id = self.player_name_to_env_id[player_name]
                
                # Get cards from previous game state
                if isinstance(player_info, dict):
                    cards = player_info.get('cards', [])
                    
                    # Update penalty thresholds (bullet position)
                    bullet_pos = player_info.get('bullet_position', 3)
                    state_dict['penalty_thresholds'][env_id] = bullet_pos if bullet_pos > 0 else 3
                    
                    # Update penalties (shots fired)
                    shots_fired = player_info.get('shots_fired', 0)
                    state_dict['penalties'][env_id] = shots_fired if shots_fired >= 0 else 0
                    
                    # Check if player is dead
                    is_dead = player_info.get('is_dead', False)
                    state_dict['terminations'][env_id] = is_dead
                else:
                    cards = player_info
                
                # Convert card numbers to environment card names
                env_cards = []
                for card in cards:
                    if card in GAME_TO_ENV_CARD:
                        env_cards.append(GAME_TO_ENV_CARD[card])
                
                state_dict['players_hands'][env_id] = env_cards
        
        # Update table card
        round_card = self.get_round_card_from_state(self.previous_game_state)
        if round_card in GAME_TO_ENV_CARD:
            state_dict['table_card'] = GAME_TO_ENV_CARD[round_card]
            state_dict['table_card_idx'] = CARD_TO_INT[GAME_TO_ENV_CARD[round_card]]
        
        # Get last played cards from previous state
        last_round = self.previous_game_state.get('last_round', {})
        player_name = last_round.get('player', None)
        played_cards = last_round.get('cards', [])
        
        if player_name and played_cards and player_name in self.player_name_to_env_id:
            env_id = self.player_name_to_env_id[player_name]
            
            # Convert played cards to environment format
            env_played_cards = []
            for card in played_cards:
                if card in GAME_TO_ENV_CARD:
                    env_played_cards.append(GAME_TO_ENV_CARD[card])
            
            # Update last played cards and action
            state_dict['last_played_cards'][env_id] = env_played_cards
            state_dict['last_action_agent'] = env_id
            state_dict['last_action'] = len(env_played_cards)  # Number of cards played
            
            # Determine if it was a bluff
            if round_card in GAME_TO_ENV_CARD:
                expected_card = GAME_TO_ENV_CARD[round_card]
                # A play is a bluff if not all cards match the expected card or are Jokers
                is_bluff = not all(card == expected_card or card == "Joker" for card in env_played_cards)
                state_dict['last_action_bluff'] = is_bluff
        
        # Set agent order based on whose turn it would have been for the challenge
        # We'll determine this from the active player in the previous state
        active_player = None
        for player_name, player_info in hands.items():
            if player_name in self.player_name_to_env_id:
                # Check if it was this player's turn
                is_my_turn = False
                if isinstance(player_info, dict):
                    is_my_turn = player_info.get('is_my_turn', False)
                
                if is_my_turn:
                    active_player = self.player_name_to_env_id[player_name]
                    break
        
        # If we found the active player, make them the current agent
        if active_player:
            active_agents = [active_player]
            # Add other agents in the standard turn order
            for env_id in ["player_0", "player_1", "player_2"]:
                if env_id != active_player and env_id in self.env_id_to_player_name:
                    active_agents.append(env_id)
            
            state_dict['agents'] = active_agents
            state_dict['agent_selection'] = active_agents[0]
        
        # Set the environment state
        self.env.set_state(state_dict)
        
    def get_round_card_from_state(self, state):
        """Get the round card from a specific game state."""
        if state:
            return state.get('last_round', {}).get('actual_card', 0)
        return 0
    
    def is_valid_game_state(self, state):
        """Check if the game state is valid and has the necessary information."""
        # Must have hands and at least one player
        if 'hands' not in state or not state['hands']:
            return False
            
        # Should have game type
        if 'game_type' not in state:
            return False
            
        return True
    
    def initialize_player_mappings(self):
        """Initialize mappings between player names and environment IDs based on the order from the mod."""
        hands = self.current_game_state.get('hands', {})

        # Get player names in the order they appear in the data
        player_names = list(hands.keys())

        # Initialize mappings
        self.player_name_to_env_id = {}
        self.env_id_to_player_name = {}

        for i, player_name in enumerate(player_names):
            env_id = f"player_{i}"
            self.player_name_to_env_id[player_name] = env_id
            self.env_id_to_player_name[env_id] = player_name

        # Save turn order as player names
        self.player_positions = player_names.copy()

        logger.info(f"Player mappings initialized: {self.player_name_to_env_id}")
        logger.info(f"Player positions (as received from mod): {self.player_positions}")
    
    def detect_new_round(self):
        """
        Detect if a new round has started.
        Returns:
            tuple: (is_new_round, challenger_id, penalized_player_id)
        """
        if self.previous_game_state is None:
            # If this is the first state, consider it a new round
            return True, None, None
        
        current_hands = self.current_game_state.get('hands', {})
        previous_hands = self.previous_game_state.get('hands', {})
        
        # Check if all players now have 5 cards (full hand)
        all_full_hands = True
        for player, player_info in current_hands.items():
            cards = player_info.get('cards', []) if isinstance(player_info, dict) else player_info
            if len(cards) != 5:
                all_full_hands = False
                break
        
        # Check if previous hands were not full
        previous_not_full = False
        for player, player_info in previous_hands.items():
            cards = player_info.get('cards', []) if isinstance(player_info, dict) else player_info
            if len(cards) < 5:
                previous_not_full = True
                break
        
        # If all hands are now full and at least one wasn't full before, it's a new round
        is_new_round = all_full_hands and previous_not_full
        
        if is_new_round:
            # Try to figure out who made the challenge and who got penalized
            challenger_id = None
            penalized_player_id = None
            
            # Check for penalties to identify who was penalized
            for player_name, player_info in current_hands.items():
                if player_name in self.player_name_to_env_id:
                    current_shots = 0
                    previous_shots = 0
                    
                    # Get current penalties (shots fired)
                    if isinstance(player_info, dict):
                        current_shots = player_info.get('shots_fired', 0)
                    
                    # Get previous penalties
                    prev_info = previous_hands.get(player_name, {})
                    if isinstance(prev_info, dict):
                        previous_shots = prev_info.get('shots_fired', 0)
                    
                    # If penalties increased, this player was penalized
                    if current_shots > previous_shots:
                        penalized_player_id = self.player_name_to_env_id[player_name]
                        logger.info(f"Detected player {player_name} was penalized (shots: {previous_shots} -> {current_shots})")
            
            # Guess the challenger - the player whose turn came right before the penalized player
            if penalized_player_id:
                # Look at the turn order
                agent_ids = ["player_0", "player_1", "player_2"]
                
                # Find the penalized player's position
                if penalized_player_id in agent_ids:
                    penalized_idx = agent_ids.index(penalized_player_id)
                    # Challenger is the player before in the turn order (circular)
                    challenger_idx = (penalized_idx - 1) % len(agent_ids)
                    challenger_id = agent_ids[challenger_idx]
                    
                    if challenger_id in self.env_id_to_player_name:
                        logger.info(f"Detected player {self.env_id_to_player_name[challenger_id]} likely made the challenge")
            
            return is_new_round, challenger_id, penalized_player_id
        
        return False, None, None
    
    def update_env_for_new_round(self, penalized_player_id=None):
        """
        Update the environment for a new round.
        
        Args:
            penalized_player_id: The player who got a penalty and should go first
        """
        # Create a new state dictionary for the environment
        state_dict = self.env.get_state()
        
        # Update table card
        round_card = self.get_round_card()
        if round_card in GAME_TO_ENV_CARD:
            state_dict['table_card'] = GAME_TO_ENV_CARD[round_card]
            state_dict['table_card_idx'] = CARD_TO_INT[GAME_TO_ENV_CARD[round_card]]
        
        # Update player hands
        hands = self.current_game_state.get('hands', {})
        for player_name, player_info in hands.items():
            if player_name in self.player_name_to_env_id:
                env_id = self.player_name_to_env_id[player_name]
                
                # Get cards from game state
                if isinstance(player_info, dict):
                    cards = player_info.get('cards', [])
                else:
                    cards = player_info
                
                # Convert card numbers to environment card names
                env_cards = []
                for card in cards:
                    if card in GAME_TO_ENV_CARD:
                        env_cards.append(GAME_TO_ENV_CARD[card])
                
                # Ensure we have 5 cards (add placeholders if needed)
                while len(env_cards) < 5:
                    env_cards.append("King")  # Default placeholder
                
                state_dict['players_hands'][env_id] = env_cards
                
                # Update penalties and thresholds
                if isinstance(player_info, dict):
                    # Update penalty thresholds (bullet position)
                    bullet_pos = player_info.get('bullet_position', 3)
                    state_dict['penalty_thresholds'][env_id] = bullet_pos if bullet_pos > 0 else 3
                    
                    # Update penalties (shots fired)
                    shots_fired = player_info.get('shots_fired', 0)
                    state_dict['penalties'][env_id] = shots_fired if shots_fired >= 0 else 0
                    
                    # Check if player is dead
                    is_dead = player_info.get('is_dead', False)
                    state_dict['terminations'][env_id] = is_dead
        
        # Reset last action data
        state_dict['last_action'] = None
        state_dict['last_action_agent'] = None
        state_dict['last_action_bluff'] = None
        state_dict['last_played_cards'] = {env_id: [] for env_id in self.env.possible_agents}
        
        # Reset round state
        state_dict['round_eliminated'] = {env_id: False for env_id in self.env.possible_agents}
        
        # Get active players (non-terminated)
        active_player_ids = []
        for env_id in ["player_1", "player_2", "player_0"]:
            if env_id in self.env_id_to_player_name and not state_dict['terminations'].get(env_id, False):
                active_player_ids.append(env_id)
        
        # If we have a penalized player, they should go first
        if penalized_player_id and penalized_player_id in active_player_ids:
            # Find the penalized player and rotate the list to put them first
            penalized_idx = active_player_ids.index(penalized_player_id)
            active_player_ids = active_player_ids[penalized_idx:] + active_player_ids[:penalized_idx]
            logger.info(f"Penalized player {self.env_id_to_player_name.get(penalized_player_id, penalized_player_id)} goes first")
        else:
            # Check if any player is marked as having their turn in the game
            current_turn_player = None
            for player_name, player_info in hands.items():
                if player_name in self.player_name_to_env_id:
                    is_my_turn = False
                    if isinstance(player_info, dict):
                        is_my_turn = player_info.get('is_my_turn', False)
                    
                    if is_my_turn:
                        current_turn_player = self.player_name_to_env_id[player_name]
                        break
            
            # If we found a current turn player, adjust order
            if current_turn_player and current_turn_player in active_player_ids:
                current_idx = active_player_ids.index(current_turn_player)
                active_player_ids = active_player_ids[current_idx:] + active_player_ids[:current_idx]
                logger.info(f"Current turn player {self.env_id_to_player_name.get(current_turn_player, current_turn_player)} goes first")
        
        # Update agents and agent_selector
        state_dict['agents'] = active_player_ids
        state_dict['agent_selection'] = active_player_ids[0] if active_player_ids else None
        
        # Set the environment state
        self.env.set_state(state_dict)
        logger.info(f"Environment updated for new round with table card: {state_dict['table_card']}")
        logger.info(f"Turn order for this round: {[self.env_id_to_player_name.get(pid, pid) for pid in state_dict['agents']]}")
    
    def update_env_state(self):
        """Update the environment state based on the current game state."""
        # Get the current environment state
        state_dict = self.env.get_state()
        
        # Update hands and penalties
        hands = self.current_game_state.get('hands', {})
        for player_name, player_info in hands.items():
            if player_name in self.player_name_to_env_id:
                env_id = self.player_name_to_env_id[player_name]
                
                # Get cards from game state
                if isinstance(player_info, dict):
                    cards = player_info.get('cards', [])
                    
                    # Update penalty thresholds (bullet position)
                    bullet_pos = player_info.get('bullet_position', 3)
                    state_dict['penalty_thresholds'][env_id] = bullet_pos if bullet_pos > 0 else 3
                    
                    # Update penalties (shots fired)
                    shots_fired = player_info.get('shots_fired', 0)
                    state_dict['penalties'][env_id] = shots_fired if shots_fired >= 0 else 0
                    
                    # Check if player is dead
                    is_dead = player_info.get('is_dead', False)
                    state_dict['terminations'][env_id] = is_dead
                else:
                    cards = player_info
                
                # Convert card numbers to environment card names
                env_cards = []
                for card in cards:
                    if card in GAME_TO_ENV_CARD:
                        env_cards.append(GAME_TO_ENV_CARD[card])
                
                state_dict['players_hands'][env_id] = env_cards
        
        # Check for last played cards
        last_round = self.current_game_state.get('last_round', {})
        player_name = last_round.get('player', None)
        played_cards = last_round.get('cards', [])
        
        # Only update if there's a meaningful change in played cards
        if player_name and played_cards and (played_cards != self.last_claimed_cards or len(played_cards) != self.claimed_card_count):
            self.last_claimed_cards = played_cards.copy()
            self.claimed_card_count = len(played_cards)
            
            if player_name in self.player_name_to_env_id:
                env_id = self.player_name_to_env_id[player_name]
                
                # Convert played cards to environment format
                env_played_cards = []
                for card in played_cards:
                    if card in GAME_TO_ENV_CARD:
                        env_played_cards.append(GAME_TO_ENV_CARD[card])
                
                # First, update the state dictionary
                state_dict['last_played_cards'][env_id] = env_played_cards
                state_dict['last_action_agent'] = env_id
                state_dict['last_action'] = len(env_played_cards)  # Number of cards played
                
                # Determine if it was a bluff
                round_card = self.get_round_card()
                if round_card in GAME_TO_ENV_CARD:
                    expected_card = GAME_TO_ENV_CARD[round_card]
                    # A play is a bluff if not all cards match the expected card or are Jokers
                    is_bluff = not all(card == expected_card or card == "Joker" for card in env_played_cards)
                    state_dict['last_action_bluff'] = is_bluff
                
                # Set the environment state
                self.env.set_state(state_dict)
                
                # Now properly simulate the action in the environment
                action = self.determine_action_from_played_cards(env_played_cards, state_dict['table_card'])
                
                # Make sure the environment's agent_selection is set to this player
                current_selection = self.env.agent_selection
                if current_selection != env_id:
                    # Find this player in agents list
                    if env_id in self.env.agents:
                        idx = self.env.agents.index(env_id)
                        # Temporarily reorder agents to put this player first
                        self.env.agents = [env_id] + [a for a in self.env.agents if a != env_id]
                        self.env._agent_selector = agent_selector(self.env.agents)
                        self.env.agent_selection = self.env._agent_selector.next()
                
                # Now step the environment with the action
                self.env.step(action)
                logger.info(f"Simulated action {action} for player {player_name} in environment")
        
        # No need to set state again as we've already updated it and simulated actions
    
    def determine_action_from_played_cards(self, played_cards, round_card):
        """Determine the environment action number (0-6) from played cards."""
        # Determine action type and count
        count = len(played_cards)
        
        # Determine if it's a table card (matching) play or non-table card play
        if all(card == round_card or card == "Joker" for card in played_cards):
            card_category = 'table'
        else:
            card_category = 'non-table'
        
        # Map to action number (0-5)
        if card_category == 'table':
            action = count - 1  # 1→0, 2→1, 3→2
        else:
            action = count + 2  # 1→3, 2→4, 3→5
        
        logger.info(f"Determined action {action} for {count} {card_category} cards")
        return action
        
    def apply_challenge_action(self, challenger_id):
        """Properly simulate a challenge action in the environment."""
        # Make sure the challenger is the current agent_selection
        current_selection = self.env.agent_selection
        if current_selection != challenger_id:
            # Find this player in agents list
            if challenger_id in self.env.agents:
                # Temporarily reorder agents to put this player first
                self.env.agents = [challenger_id] + [a for a in self.env.agents if a != challenger_id]
                self.env._agent_selector = agent_selector(self.env.agents)
                self.env.agent_selection = self.env._agent_selector.next()
        
        # Now step the environment with the challenge action (6)
        self.env.step(6)
        logger.info(f"Simulated challenge action (6) for agent {challenger_id} in environment")
    
    def apply_challenge_to_env(self, challenger_id, state_dict):
        """Apply a challenge action to the environment."""
        action = 6  # Challenge action
        
        # Update last_agent_action for the challenger
        state_dict['last_agent_action'][challenger_id] = action
        
        # In a real challenge, a new round would start
        # For simplicity, we'll just update agent selection
        if challenger_id in state_dict['agents']:
            idx = state_dict['agents'].index(challenger_id)
            # Move to next agent (wrap around if needed)
            next_idx = (idx + 1) % len(state_dict['agents'])
            state_dict['agent_selection'] = state_dict['agents'][next_idx]
        
        logger.info(f"Applied challenge (action 6) for agent {challenger_id}")
        return action
    
    def get_round_card(self):
        """Get the current round card from the game state."""
        if self.current_game_state:
            return self.current_game_state.get('last_round', {}).get('actual_card', 0)
        return 0
    
    def is_ai_turn_simple(self) -> bool:
        """
        Simple turn detector: if the last player to act was the one
        immediately before our AI in self.player_positions, then it's AI's turn.
        """
        if not self.previous_game_state:
           return False

        last_player = self.previous_game_state.get('last_round', {}).get('player')
        if not last_player:
            return False

        # Make sure our ordering is set up
        if self.ai_player_name not in self.player_positions:
            return False

        idx_ai = self.player_positions.index(self.ai_player_name)
        idx_prev = (idx_ai - 1) % len(self.player_positions)
        prev_player = self.player_positions[idx_prev]

        return last_player == prev_player
    
    def get_ai_action(self):
        """Get an action from the AI agent."""
        if self.ai_agent is None:
            logger.error("AI agent not initialized")
            return None
        
        # Get the environment ID for the AI player
        ai_env_id = self.player_name_to_env_id.get(self.ai_player_name)
        if not ai_env_id:
            logger.error("AI player environment ID not found")
            return None
        
        # Get observation and info for the AI
        observation = self.env.observe(ai_env_id, newer=True)
        info = self.env.infos.get(ai_env_id, {})
        
        # Get action from AI
        try:
            action = self.ai_agent.get_action(self.env, ai_env_id, observation, info)
            logger.info(f"AI selected action: {action}")
            
            # Log the decoded action for clarity
            action_type, card_category, count = decode_action(action)
            logger.info(f"Decoded as: type={action_type}, category={card_category}, count={count}")
            
            return action
        except Exception as e:
            logger.error(f"Error getting AI action: {e}", exc_info=True)
            return 0  # Default to playing 1 table card
    
    def execute_ai_action_in_game(self, action):
        """Execute the AI's action in the real game."""
        # Focus the game window
        self.focus_game_window()
        time.sleep(COMMAND_DELAY)
        
        # Decode the action
        action_type, card_category, count = decode_action(action)
        
        # Execute based on action type
        if action_type == "Challenge":
            logger.info("Executing Challenge action in game")
            self.challenge_play()
            return True
        
        elif action_type == "Play":
            logger.info(f"Executing Play action: {count} {card_category} card(s)")
            
            # Get the round card
            round_card = self.get_round_card()
            round_card_env = GAME_TO_ENV_CARD.get(round_card, "King")
            
            # Get AI's cards
            my_cards = []
            ai_hands = self.current_game_state.get('hands', {}).get(self.ai_player_name, {})
            if isinstance(ai_hands, dict):
                my_cards = ai_hands.get('cards', [])
            else:
                my_cards = ai_hands
            
            # Define card filters based on action
            if card_category == 'table':
                # Cards that match round card or Joker (4)
                card_filter = lambda card: card == round_card or card == 4
            else:
                # Cards that don't match round card and are not Jokers
                card_filter = lambda card: card != round_card and card != 4
            
            # Find appropriate cards
            card_indices = self.find_card_indices(card_filter, count, my_cards)
            
            if len(card_indices) < count:
                logger.warning(f"Not enough {'matching' if card_category == 'table' else 'non-matching'} cards (need {count}, found {len(card_indices)})")
                # Try to find any cards if specific category not available
                any_indices = list(range(min(count, len(my_cards))))
                return self.select_and_play_cards(any_indices)
            
            # Select and play these cards
            return self.select_and_play_cards(card_indices)
        
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return False
    
    def find_card_indices(self, card_type_filter, num_needed, my_cards):
        """Find indices of cards that match the filter function."""
        if not my_cards:
            return []
        
        matching_indices = [i for i, card in enumerate(my_cards) if card_type_filter(card)]
        
        # Return up to num_needed indices
        return matching_indices[:num_needed]
    
    def focus_game_window(self):
        """Focus the game window by its title."""
        try:
            # Try to focus the game window - adjust the window title as needed
            window = pyautogui.getWindowsWithTitle("Liar's Bar")
            if window:
                window[0].activate()
                time.sleep(COMMAND_DELAY)  # Wait for window to activate
                return True
            else:
                logger.warning("Game window not found. Make sure the game is running.")
                logger.warning(f"Waiting {COMMAND_DELAY} seconds to allow you to click on the game window...")
                time.sleep(COMMAND_DELAY)
                return True  # Continue anyway to allow manual focus
        except Exception as e:
            logger.error(f"Error focusing game window: {e}")
            logger.warning(f"Waiting {COMMAND_DELAY} seconds to allow you to click on the game window...")
            time.sleep(COMMAND_DELAY)
            return True  # Continue anyway to allow manual focus
    
    def press_key(self, key, times=1, delay=0.1):
        """Press a key a specified number of times."""
        for _ in range(times):
            pyautogui.press(key)
            time.sleep(delay)
    
    def challenge_play(self):
        """Challenge the last play by pressing X."""
        if not self.focus_game_window():
            return "Failed to focus game window"
        
        pyautogui.press('x')
        return True
    
    def play_cards(self):
        """Play selected cards by pressing E."""
        if not self.focus_game_window():
            return "Failed to focus game window"
        
        pyautogui.press('e')
        return True
    
    def select_current_card(self):
        """Select the currently hovered card by pressing space."""
        if not self.focus_game_window():
            return "Failed to focus game window"
        
        pyautogui.press('space')
        return True
    
    def move_left(self, times=1):
        """Move to the left card by pressing A."""
        if not self.focus_game_window():
            return "Failed to focus game window"
        
        self.press_key('a', times)
        return True
    
    def move_right(self, times=1):
        """Move to the right card by pressing D."""
        if not self.focus_game_window():
            return "Failed to focus game window"
        
        self.press_key('d', times)
        return True
    
    def select_and_play_cards(self, card_indices):
        """Select the cards at the given indices and play them."""
        if not card_indices:
            return False
        
        # Show debug info
        logger.info(f"Selecting cards at indices: {card_indices}")
        
        # Get AI's cards from game state
        my_cards = []
        ai_hands = self.current_game_state.get('hands', {}).get(self.ai_player_name, {})
        if isinstance(ai_hands, dict):
            my_cards = ai_hands.get('cards', [])
        else:
            my_cards = ai_hands
        
        # Validate indices
        for idx in card_indices:
            if idx < 0 or idx >= len(my_cards):
                logger.warning(f"Invalid card index: {idx}, available: 0-{len(my_cards)-1}")
                return False
        
        if not self.focus_game_window():
            return False
        
        # First turn behavior is different: starts with rightmost card
        if self.is_first_turn_in_round:
            # First turn - UI starts with rightmost card, need to press 'a' to move left
            for idx in card_indices:
                moves_needed = len(my_cards) - 1 - idx
                if moves_needed > 0:
                    logger.info(f"First turn: Moving left {moves_needed} times to reach card {idx}")
                    self.press_key('a', moves_needed)
                
                time.sleep(0.2)  # Small delay after movement
                
                # Select the card
                self.select_current_card()
                time.sleep(0.2)
                
                # Go back to first card position to prepare for next selection if needed
                if len(card_indices) > 1 and idx < card_indices[-1]:
                    # Move all the way to the right
                    self.press_key('d', moves_needed)
                time.sleep(0.2)
        else:
            # Subsequent turns - UI starts with leftmost card, need to press 'd' to move right
            for idx in card_indices:
                moves_needed = idx
                if moves_needed > 0:
                    logger.info(f"Subsequent turn: Moving right {moves_needed} times to reach card {idx}")
                    self.press_key('d', moves_needed)
                
                time.sleep(0.2)  # Small delay after movement
                
                # Select the card
                self.select_current_card()
                time.sleep(0.2)
                
                # Go back to first card position to prepare for next selection if needed
                if len(card_indices) > 1 and idx < card_indices[-1]:
                    # Move all the way to the left
                    self.press_key('a', moves_needed)
                time.sleep(0.2)
        
        # Play the selected cards
        self.play_cards()
        
        # After successfully playing a card, it's definitely not the first turn anymore
        self.is_first_turn_in_round = False
        
        return True
    
    def main_loop(self):
        """Main loop to process game state and execute AI actions."""
        logger.info("Starting main loop")
        last_state_dump_time = 0
        dump_interval = 5  # seconds
        
        try:
            while self.running:
                # Periodically dump game state for debugging
                current_time = time.time()
                if current_time - last_state_dump_time > dump_interval:
                    if self.current_game_state:
                        self.dump_game_state()
                        last_state_dump_time = current_time
                
                # Check if it's AI's turn to act
                if self.ai_needs_to_act:
                    logger.info("AI preparing to act")
                    print("\n" + "="*50)
                    print("AI IS MAKING A MOVE NOW")
                    print("="*50 + "\n")
                    
                    # Dump state again when AI is acting
                    self.dump_game_state()
                    
                    # Get AI action
                    action = self.get_ai_action()
                    
                    if action is not None:
                        logger.info(f"AI decided on action: {action}")
                        
                        # Decode the action for clearer logging
                        action_type, card_category, count = decode_action(action)
                        logger.info(f"Decoded as: type={action_type}, category={card_category}, count={count}")
                        
                        # Execute the action in the game
                        success = self.execute_ai_action_in_game(action)
                        if success:
                            logger.info("AI action executed successfully")
                            
                            # Now properly simulate the action in our environment
                            env_id = self.player_name_to_env_id.get(self.ai_player_name)
                            if env_id:
                                # Make sure AI is the current agent selection
                                if self.env.agent_selection != env_id:
                                    # Temporarily reorder agents to put AI first
                                    self.env.agents = [env_id] + [a for a in self.env.agents if a != env_id]
                                    self.env._agent_selector = agent_selector(self.env.agents)
                                    self.env.agent_selection = self.env._agent_selector.next()
                                
                                # Step the environment with the AI's action
                                self.env.step(action)
                                logger.info(f"Simulated AI action {action} in environment")
                        else:
                            logger.warning("Failed to execute AI action")
                        
                        # Reset flag after acting
                        self.ai_needs_to_act = False
                    else:
                        logger.error("AI returned None for action")
                
                # Sleep to prevent high CPU usage
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("Main loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info("Main loop stopped")

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/checkpoint_1.36m_both_AR_best.pth"
    
    # Configure logging to file and console
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.FileHandler("ar_interface.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting AR interface with device: {device}")
    interface = ARGameInterface(checkpoint_path, device)
    
    # Add a keyboard interrupt handler
    try:
        interface.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error running interface: {e}", exc_info=True)
    finally:
        logger.info("Interface stopped")