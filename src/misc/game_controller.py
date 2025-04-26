import socket
import json
import time
import threading
import pyautogui
from typing import Dict, List, Any, Optional, Tuple

# Configure pyautogui safety settings
pyautogui.PAUSE = 0.1  # Add a short pause between commands
pyautogui.FAILSAFE = True  # Move mouse to upper left to abort

# Game state storage
current_game_state = None
state_lock = threading.Lock()
my_player_name = "VeryGoodName"  # Hard-coded to VeryGoodName

# Command delay in seconds
COMMAND_DELAY = 2.0

# Keep track of whether it's the first turn in the game
is_first_turn = True

# Card type names
CARD_NAMES = {
    1: "King",
    2: "Queen", 
    3: "Ace",
    4: "Joker",   # It's Joker, not Jack
    -1: "Devil"
}

def format_card(card_num: int, game_type: str = "Default") -> str:
    """Convert card number to readable format with colors."""
    card_text = ""
    color = ""
    
    if card_num == 1: 
        card_text = "King"
        color = "yellow"
    elif card_num == 2: 
        card_text = "Queen"
        color = "magenta"
    elif card_num == 3: 
        card_text = "Ace"
        color = "green" 
    elif card_num == 4: 
        card_text = "Joker"  # Changed from Jack to Joker
        color = "cyan"
    elif card_num == -1: 
        card_text = "Devil"
        color = "red"
    else:
        card_text = f"Unknown({card_num})"
        color = "white"
    
    # Special case for Chaos mode
    if game_type == "Chaos":
        if card_num == 3:
            card_text = "Chaos"
        elif card_num == 4:
            card_text = "Master"
    
    return f"\033[{get_color_code(color)}m{card_text}\033[0m"

def get_color_code(color: str) -> str:
    """Return ANSI color code for terminal output."""
    colors = {
        "red": "91",
        "green": "92", 
        "yellow": "93",
        "blue": "94",
        "magenta": "95",
        "cyan": "96",
        "white": "97"
    }
    return colors.get(color, "97")  # Default to white

def print_game_state(data: Dict[str, Any], previous_data: Optional[Dict[str, Any]] = None) -> None:
    """Pretty print the game state data with colored output."""
    global my_player_name
    
    print("\n" + "="*50)
    print(f"🎮 Game Type: \033[96m{data.get('game_type', 'Unknown')}\033[0m")
    
    # Display action possibilities
    can_play = data.get('can_play', False)
    can_challenge = data.get('can_challenge', False)
    
    if can_play:
        print(f"✅ \033[92mYou can play cards\033[0m")
    else:
        print(f"❌ \033[91mYou cannot play cards\033[0m")
        
    if can_challenge:
        print(f"✅ \033[92mYou can challenge\033[0m")
    else:
        print(f"❌ \033[91mYou cannot challenge\033[0m")
    
    print("-"*50)
    
    hands = data.get('hands', {})
    
    if hands:
        print("👥 Players:")
        for player_name, player_info in hands.items():
            # Handle both old and new format
            if isinstance(player_info, list):
                cards = player_info
                is_dead = False
                bullet_pos = -1
                shots_fired = -1
                selected = []
                is_my_turn = False
                active_card = -1
            else:
                cards = player_info.get('cards', [])
                is_dead = player_info.get('is_dead', False)
                bullet_pos = player_info.get('bullet_position', -1)
                shots_fired = player_info.get('shots_fired', -1)
                selected = player_info.get('selected', [False] * len(cards))
                is_my_turn = player_info.get('is_my_turn', False)
                active_card = player_info.get('active_card_index', -1)
            
            status = "\033[91m[DEAD]\033[0m" if is_dead else "\033[92m[ALIVE]\033[0m"
            turn_indicator = "➡️ " if is_my_turn else "  "
            me_indicator = "🧑 " if player_name == my_player_name else "  "
            
            print(f"{me_indicator}{turn_indicator}{status} \033[93m{player_name}\033[0m:")
            
            if bullet_pos >= 0:
                print(f"    🔫 Bullet Position: {bullet_pos}, Shots Fired: {shots_fired}")
            
            if cards:
                # Display cards with selection status
                card_strings = []
                for i, (card, is_selected) in enumerate(zip(cards, selected)):
                    card_str = format_card(card, data.get('game_type'))
                    
                    if i == active_card:
                        if is_selected:
                            card_strings.append(f"[\033[93m{i}\033[0m]({card_str})* 👈")  # Active and selected
                        else:
                            card_strings.append(f"[\033[93m{i}\033[0m]({card_str}) 👈")  # Active but not selected
                    else:
                        if is_selected:
                            card_strings.append(f"[{i}]({card_str})*")  # Selected
                        else:
                            card_strings.append(f"[{i}]({card_str})")  # Normal
                
                card_display = ", ".join(card_strings)
                print(f"    🃏 Current Hand: {card_display}")
            else:
                print(f"    🃏 Current Hand: None")
    
    last_round = data.get('last_round', {})
    if last_round:
        player = last_round.get('player', 'None')
        cards = last_round.get('cards', [])
        round_card = last_round.get('actual_card', 0)
        
        if player != "None" or cards or round_card > 0:
            print("-"*50)
            print(f"🎮 Round Information:")
            
            if round_card > 0:
                print(f"  🎯 Round Card: {format_card(round_card, data.get('game_type'))}") 
                
            if player != "None" and cards:
                print(f"  🎭 Last played by: \033[93m{player}\033[0m")
                card_str = ", ".join(format_card(card, data.get('game_type')) for card in cards)
                print(f"  🃏 Played cards: {card_str}")
    
    print("="*50)
    
    # Print available AI actions
    round_card = data.get('last_round', {}).get('actual_card', 0)
    if round_card > 0:
        print(f"Available AI Actions:")
        print(f"  Action 0: Play 1 card matching {format_card(round_card)} or Joker")
        print(f"  Action 1: Play 2 cards matching {format_card(round_card)} or Joker")
        print(f"  Action 2: Play 3 cards matching {format_card(round_card)} or Joker")
        print(f"  Action 3: Play 1 card NOT matching {format_card(round_card)} (excluding Jokers)")
        print(f"  Action 4: Play 2 cards NOT matching {format_card(round_card)} (excluding Jokers)")
        print(f"  Action 5: Play 3 cards NOT matching {format_card(round_card)} (excluding Jokers)")
        print(f"  Action 6: Challenge last play")
        print("-"*50)

def has_meaningful_change(new_data: Dict[str, Any], old_data: Dict[str, Any]) -> bool:
    """Check if there's a meaningful change between two data states."""
    if old_data is None:
        return True
    
    # Check game type change
    if new_data.get('game_type') != old_data.get('game_type'):
        return True
    
    # Check play/challenge state changes
    if new_data.get('can_play') != old_data.get('can_play'):
        return True
        
    if new_data.get('can_challenge') != old_data.get('can_challenge'):
        return True
    
    # Check last round changes
    new_round = new_data.get('last_round', {})
    old_round = old_data.get('last_round', {})
    
    if new_round.get('player') != old_round.get('player'):
        return True
    
    if new_round.get('cards') != old_round.get('cards'):
        return True
        
    if new_round.get('actual_card') != old_round.get('actual_card'):
        return True
    
    # Check player hands
    new_hands = new_data.get('hands', {})
    old_hands = old_data.get('hands', {})
    
    # Check for added or removed players
    if set(new_hands.keys()) != set(old_hands.keys()):
        return True
    
    # Check for changes in player data
    for player, new_info in new_hands.items():
        if player not in old_hands:
            return True
        
        old_info = old_hands[player]
        
        # Handle both formats (list or dict)
        if isinstance(new_info, list) and isinstance(old_info, list):
            if new_info != old_info:
                return True
        elif isinstance(new_info, dict) and isinstance(old_info, dict):
            # Check card changes
            if new_info.get('cards') != old_info.get('cards'):
                return True
            # Check selection changes
            if new_info.get('selected') != old_info.get('selected'):
                return True
            # Check death status changes
            if new_info.get('is_dead') != old_info.get('is_dead'):
                return True
            # Check bullet position changes
            if new_info.get('bullet_position') != old_info.get('bullet_position'):
                return True
            # Check shots fired changes
            if new_info.get('shots_fired') != old_info.get('shots_fired'):
                return True
            # Check turn changes
            if new_info.get('is_my_turn') != old_info.get('is_my_turn'):
                return True
            # Check active card changes
            if new_info.get('active_card_index') != old_info.get('active_card_index'):
                return True
        else:
            # Format change is definitely a meaningful change
            return True
    
    return False

def state_listener(host='127.0.0.1', port=5005):
    """Listen for game state from the mod."""
    global current_game_state
    
    print(f"🔌 Listening on {host}:{port} for game state...\n")
    
    previous_data = None
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        
        while True:
            client_socket, addr = server_socket.accept()
            with client_socket:
                data = client_socket.recv(8192)
                if not data:
                    continue
                
                try:
                    raw_json = data.decode('utf-8-sig').strip()
                    parsed = json.loads(raw_json)
                    
                    # Update the shared game state
                    with state_lock:
                        current_game_state = parsed
                    
                    # Only print if there's a meaningful change
                    if has_meaningful_change(parsed, previous_data):
                        print_game_state(parsed, previous_data)
                        previous_data = parsed
                        
                except Exception as e:
                    print(f"⚠️ Failed to parse JSON: {e}")
                    print(f"📦 Raw: {data[:200]}...")  # Show first 200 bytes

def get_my_cards():
    """Get the current player's cards."""
    global current_game_state, my_player_name
    
    with state_lock:
        if not current_game_state:
            return []
        
        hands = current_game_state.get('hands', {})
        
        # Look for VeryGoodName player's cards
        if my_player_name in hands:
            player_info = hands[my_player_name]
            if isinstance(player_info, dict):
                return player_info.get('cards', [])
            elif isinstance(player_info, list):
                return player_info
    
    return []

def get_round_card():
    """Get the current round card."""
    global current_game_state
    
    with state_lock:
        if not current_game_state:
            return 0
        
        round_card = current_game_state.get('last_round', {}).get('actual_card', 0)
        return round_card

def focus_game_window():
    """Focus the game window by its title."""
    try:
        # Try to focus the game window - adjust the window title as needed
        window = pyautogui.getWindowsWithTitle("Liar's Bar")
        if window:
            window[0].activate()
            time.sleep(COMMAND_DELAY)  # Wait for window to activate
            return True
        else:
            print("⚠️ Game window not found. Make sure the game is running.")
            print(f"⏳ Waiting {COMMAND_DELAY} seconds to allow you to click on the game window...")
            time.sleep(COMMAND_DELAY)
            return True  # Continue anyway to allow manual focus
    except Exception as e:
        print(f"⚠️ Error focusing game window: {e}")
        print(f"⏳ Waiting {COMMAND_DELAY} seconds to allow you to click on the game window...")
        time.sleep(COMMAND_DELAY)
        return True  # Continue anyway to allow manual focus

def press_key(key, times=1, delay=0.1):
    """Press a key a specified number of times."""
    for _ in range(times):
        pyautogui.press(key)
        time.sleep(delay)

def challenge_play():
    """Challenge the last play by pressing X."""
    if not focus_game_window():
        return "Failed to focus game window"
    
    pyautogui.press('x')
    return "Challenged last play"

def play_cards():
    """Play selected cards by pressing E."""
    if not focus_game_window():
        return "Failed to focus game window"
    
    pyautogui.press('e')
    return "Played selected cards"

def select_current_card():
    """Select the currently hovered card by pressing space."""
    if not focus_game_window():
        return "Failed to focus game window"
    
    pyautogui.press('space')
    return "Selected current card"

def move_left(times=1):
    """Move to the left card by pressing A."""
    if not focus_game_window():
        return "Failed to focus game window"
    
    press_key('a', times)
    return f"Moved left {times} times"

def move_right(times=1):
    """Move to the right card by pressing D."""
    if not focus_game_window():
        return "Failed to focus game window"
    
    press_key('d', times)
    return f"Moved right {times} times"

def reset_first_turn():
    """Reset the first turn flag for a new game."""
    global is_first_turn
    is_first_turn = True
    return "First turn flag reset for new game"

def toggle_first_turn():
    """Toggle the first turn flag."""
    global is_first_turn
    is_first_turn = not is_first_turn
    return f"First turn flag set to {is_first_turn}"

def find_card_indices(card_type_filter, num_needed):
    """Find indices of cards that match the filter function."""
    my_cards = get_my_cards()
    
    if not my_cards:
        return []
    
    matching_indices = [i for i, card in enumerate(my_cards) if card_type_filter(card)]
    
    # Return up to num_needed indices
    return matching_indices[:num_needed]

def select_and_play_cards(card_indices):
    """Select the cards at the given indices and play them."""
    global is_first_turn  # Declare global at the beginning of the function
    
    if not card_indices:
        return "No cards to play"
        
    # Get my cards using improved player detection
    my_cards = get_my_cards()
    
    if not my_cards:
        return "No cards available - can't find your cards"
    
    # Show debug info
    print(f"DEBUG: Your cards: {my_cards}")
    print(f"DEBUG: Selecting cards at indices: {card_indices}")
    
    # Validate indices
    for idx in card_indices:
        if idx < 0 or idx >= len(my_cards):
            return f"Invalid card index: {idx}, available: 0-{len(my_cards)-1}"
    
    if not focus_game_window():
        return "Failed to focus game window"
    
    # Now select each card
    for idx in card_indices:
        # Calculate moves needed based on whether it's first turn or not
        if is_first_turn:
            # First turn - UI starts with rightmost card, need to press 'a' to move left
            moves_needed = len(my_cards) - 1 - idx
            if moves_needed > 0:
                print(f"First turn: Moving left {moves_needed} times to reach card {idx}")
                press_key('a', moves_needed)
        else:
            # Subsequent turns - UI starts with leftmost card, need to press 'd' to move right
            moves_needed = idx
            if moves_needed > 0:
                print(f"Subsequent turn: Moving right {moves_needed} times to reach card {idx}")
                press_key('d', moves_needed)
        
        time.sleep(0.2)  # Small delay after movement
        
        # Select the card
        pyautogui.press('space')
        time.sleep(0.2)
        
        # Go back to first card position to prepare for next selection if needed
        if len(card_indices) > 1 and idx < card_indices[-1]:
            if is_first_turn:
                # Move all the way to the right
                press_key('d', moves_needed)
            else:
                # Move all the way to the left
                press_key('a', moves_needed)
            time.sleep(0.2)
    
    # Play the selected cards
    pyautogui.press('e')
    
    # After successfully playing a card, it's definitely not the first turn anymore
    if is_first_turn:
        is_first_turn = False
        print("First turn completed - subsequent turns will start with leftmost card active")
    
    return f"Selected and played cards at indices: {card_indices}"

def play_ai_action(action_number):
    """Convert AI action number to game action."""
    # Validate action number
    if action_number < 0 or action_number > 6:
        return f"Invalid action number: {action_number}, must be between 0-6"
    
    # Action 6 is challenge
    if action_number == 6:
        return challenge_play()
    
    # For other actions, we need to find matching or non-matching cards
    round_card = get_round_card()
    if round_card == 0:
        return "Cannot determine the round card"
    
    # Determine if we're looking for matching or non-matching cards
    is_matching = action_number < 3  # Actions 0,1,2 are matching cards
    
    # Determine number of cards needed
    num_cards = (action_number % 3) + 1  # 0->1, 1->2, 2->3, 3->1, 4->2, 5->3
    
    print(f"Action {action_number}: {'Matching' if is_matching else 'Non-matching'} cards, need {num_cards}")
    print(f"Round card: {round_card} ({CARD_NAMES.get(round_card, 'Unknown')})")
    
    # Define card filters
    if is_matching:
        # Cards that match round card or Joker (4)
        card_filter = lambda card: card == round_card or card == 4
    else:
        # Cards that don't match round card and are NOT Jokers (save Jokers)
        card_filter = lambda card: card != round_card and card != 4
    
    # Find appropriate cards
    card_indices = find_card_indices(card_filter, num_cards)
    
    if len(card_indices) < num_cards:
        return f"Not enough {'matching' if is_matching else 'non-matching'} cards (need {num_cards}, found {len(card_indices)})"
    
    # Select and play these cards
    return select_and_play_cards(card_indices)

def play_card_at_index(card_index):
    """Select and play a card at the specified index (0 is leftmost)."""
    return select_and_play_cards([card_index])

def simple_ai_play():
    """A simple AI to demonstrate playing cards."""
    global current_game_state
    
    with state_lock:
        if not current_game_state:
            print("No game state available")
            return "No game state available"
        
        # Check if it's our turn
        can_play = current_game_state.get('can_play', False)
        can_challenge = current_game_state.get('can_challenge', False)
        
        if can_challenge:
            # 20% chance to challenge
            import random
            if random.random() < 0.2:
                print("AI decides to challenge...")
                return challenge_play()
        
        if can_play:
            round_card = get_round_card()
            my_cards = get_my_cards()
            
            if not my_cards:
                return "No cards to play"
            
            # Check for matching cards
            matching_cards = [i for i, card in enumerate(my_cards) if card == round_card or card == 4]
            
            if matching_cards:
                # Play a matching card
                idx = matching_cards[0]
                print(f"AI plays matching card at index {idx}")
                return play_card_at_index(idx)
            else:
                # Play first non-matching card that's not a Joker
                non_matching = [i for i, card in enumerate(my_cards) if card != round_card and card != 4]
                if non_matching:
                    idx = non_matching[0]
                    print(f"AI plays non-matching card at index {idx}")
                    return play_card_at_index(idx)
                else:
                    # If all else fails, play the first card
                    print("AI plays first card")
                    return play_card_at_index(0)
    
    return "No action taken"

def command_loop():
    """Interactive command loop for player or AI control."""
    print("\n🎮 Game Controller Ready")
    print("Commands:")
    print("  move <n> - Move left n times")
    print("  moveright <n> - Move right n times")
    print("  select - Select current card")
    print("  play - Play selected cards")
    print("  challenge - Challenge the last play")
    print("  play <index> - Play card at index (starting from 0 on the left)")
    print("  action <n> - Execute AI action number (0-6)")
    print("  firstturn - Toggle whether it's the first turn of game (changes UI behavior)")
    print("  reset - Reset first turn flag for new game")
    print("  ai - Let the AI play one move")
    print("  help - Show this help")
    print("  exit - Exit the controller")
    print("\nℹ️ First turn status: ACTIVE (UI starts with rightmost card active)")
    print(f"ℹ️ Your player name: {my_player_name}")
    
    while True:
        try:
            cmd = input("\nCommand> ").strip()
            
            if not cmd:
                continue
                
            if cmd.lower() == "exit":
                break
                
            elif cmd.lower() == "help":
                print("Commands:")
                print("  move <n> - Move left n times")
                print("  moveright <n> - Move right n times")
                print("  select - Select current card")
                print("  play - Play selected cards")
                print("  challenge - Challenge the last play")
                print("  play <index> - Play card at index (starting from 0 on the left)")
                print("  action <n> - Execute AI action number (0-6):")
                print("    - 0: Play 1 matching card")
                print("    - 1: Play 2 matching cards")
                print("    - 2: Play 3 matching cards")
                print("    - 3: Play 1 non-matching card (not Joker)")
                print("    - 4: Play 2 non-matching cards (not Jokers)")
                print("    - 5: Play 3 non-matching cards (not Jokers)")
                print("    - 6: Challenge")
                print("  firstturn - Toggle whether it's the first turn of game (changes UI behavior)")
                print("  reset - Reset first turn flag for new game")
                print("  ai - Let the AI play one move")
                print("  help - Show this help")
                print("  exit - Exit the controller")
                print(f"\nℹ️ First turn status: {'ACTIVE' if is_first_turn else 'INACTIVE'}")
                print(f"ℹ️ Your player name: {my_player_name}")
            
            elif cmd.lower().startswith("move "):
                try:
                    times = int(cmd.split(" ")[1])
                    result = move_left(times)
                    print(result)
                except ValueError:
                    print("Invalid number of times")
                except IndexError:
                    print("Usage: move <times>")
            
            elif cmd.lower().startswith("moveright "):
                try:
                    times = int(cmd.split(" ")[1])
                    result = move_right(times)
                    print(result)
                except ValueError:
                    print("Invalid number of times")
                except IndexError:
                    print("Usage: moveright <times>")
            
            elif cmd.lower() == "select":
                result = select_current_card()
                print(result)
            
            elif cmd.lower() == "play":
                result = play_cards()
                print(result)
            
            elif cmd.lower().startswith("play "):
                try:
                    index = int(cmd.split(" ")[1])
                    result = play_card_at_index(index)
                    print(result)
                except ValueError:
                    print("Invalid card index")
                except IndexError:
                    print("Usage: play <index>")
            
            elif cmd.lower().startswith("action "):
                try:
                    action = int(cmd.split(" ")[1])
                    result = play_ai_action(action)
                    print(result)
                except ValueError:
                    print("Invalid action number")
                except IndexError:
                    print("Usage: action <number>")
            
            elif cmd.lower() == "challenge":
                result = challenge_play()
                print(result)
            
            elif cmd.lower() == "firstturn":
                result = toggle_first_turn()
                print(result)
                print(f"ℹ️ First turn status: {'ACTIVE' if is_first_turn else 'INACTIVE'}")
                
            elif cmd.lower() == "reset":
                result = reset_first_turn()
                print(result)
                print(f"ℹ️ First turn status: {'ACTIVE' if is_first_turn else 'INACTIVE'}")
                
            elif cmd.lower() == "ai":
                result = simple_ai_play()
                print(result)
                
            else:
                print(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    try:
        # Start the state listener in a separate thread
        listener_thread = threading.Thread(target=state_listener, daemon=True)
        listener_thread.start()
        
        # Start the command loop
        command_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Controller stopped by user")
    except Exception as e:
        print(f"⚠️ Error: {e}")