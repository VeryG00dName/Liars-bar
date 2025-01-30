# src/env/liars_deck_env_utils_2.py
import numpy as np

TABLE_CARD_MAP = {"King": 0, "Queen": 1, "Ace": 2}

def create_deck(np_random):
    ranks = ['Ace', 'King', 'Queen']
    deck = ranks * 6 + ['Joker', 'Joker']
    np_random.shuffle(deck)
    return deck

def encode_hand(hand, table_card):
    """
    Encodes the agent's hand by counting table and non-table cards.
    
    Args:
        hand (list): List of cards in the agent's hand.
        table_card (str): The current table card.
    
    Returns:
        np.ndarray: Array containing counts of table and non-table cards.
    """
    # Initialize counts
    table_card_count = 0
    non_table_card_count = 0
    
    for card in hand:
        if card == "Joker" or card == table_card:
            table_card_count += 1
        else:
            non_table_card_count += 1
    
    # Normalize counts (optional)
    # Assuming maximum hand size is 5
    table_card_count_norm = table_card_count / 5.0
    non_table_card_count_norm = non_table_card_count / 5.0
    
    return np.array([table_card_count_norm, non_table_card_count_norm], dtype=np.float32)

def decode_action(action):
    """
    Decodes the discrete action into its components.
    
    Args:
        action (int or None): Discrete action index or None.
    
    Returns:
        tuple: (action_type, card_category, count)
    """
    if action is None:
        return 'Invalid', None, None

    if 0 <= action <= 5:
        return 'Play', 'table' if action < 3 else 'non-table', (action % 3) + 1
    elif action == 6:
        return 'Challenge', None, None
    return 'Invalid', None, None

def select_cards_to_play(current_hand, card_category, count, table_card, np_random):
    if card_category == 'table':
        valid_cards = [card for card in current_hand if card == table_card or card == 'Joker']
    else:
        valid_cards = [card for card in current_hand if card != table_card and card != 'Joker']

    if len(valid_cards) >= count:
        return np_random.choice(valid_cards, size=count, replace=False).tolist()
    return []

def validate_claim(claimed_cards, table_card):
    if not claimed_cards:
        return False
    return all(card == table_card or card == 'Joker' for card in claimed_cards)

def simulate_russian_roulette(np_random):
    return 3  # Placeholder: replace with actual logic if needed
