import os
import pickle
import random
import time
from collections import Counter
from src.training.train_transformer import balance_training_data
# Load the dataset
DATA_PATH = "opponent_training_data.pkl"

if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found!")
    exit()

with open(DATA_PATH, "rb") as f:
    training_data = pickle.load(f)

# Shuffle the data
training_data = balance_training_data(training_data)
random.shuffle(training_data)

# Define a simple function to display an event
def display_event(event):
    print(f"\nEvent Details:")
    print(f"  - Response: {event.get('response', 'Unknown')}")
    print(f"  - Triggering Action: {event.get('triggering_action', 'Unknown')}")
    print(f"  - Penalties: {event.get('penalties', 0)}")
    print(f"  - Card Count: {event.get('card_count', 0)}")

# Main loop to test your classification ability
def classify_strategies():
    correct_guesses = 0
    total_guesses = 0
    print("\nWelcome to the Strategy Classification Test!")
    print("You will be shown a sequence of game events and try to guess the strategy.")

    for i, (memory, actual_strategy) in enumerate(training_data[:10]):  # Show 10 samples for testing
        print("\n" + "="*50)
        print(f"\nStrategy Sample {i+1}/{10}")
        
        # Show the first few events from the sequence
        for j, event in enumerate(memory[:20]):  # Show only the first 5 events to make it manageable
            print(f"\nEvent {j+1}:")
            display_event(event)

        # Ask for the guess
        print("\nAvailable strategies:")
        unique_strategies = list(set(label for _, label in training_data))
        for idx, strat in enumerate(unique_strategies):
            print(f"  {idx+1}. {strat}")

        user_guess = input("\nWhich strategy do you think this belongs to? (Enter number or exact name): ").strip()

        # Convert number input to a strategy name
        if user_guess.isdigit():
            user_guess = unique_strategies[int(user_guess) - 1]

        # Show the correct answer
        print(f"\nCorrect Answer: {actual_strategy}")
        if user_guess.lower() == actual_strategy.lower():
            print("✅ Correct!")
            correct_guesses += 1
        else:
            print("❌ Incorrect.")

        total_guesses += 1
        time.sleep(1)

    # Final results
    print("\n" + "="*50)
    print("\n🎯 Test Complete!")
    print(f"Final Score: {correct_guesses}/{total_guesses} ({(correct_guesses/total_guesses)*100:.2f}%)")
    print("If this was hard, your feature representation might not be strong enough for the model either.")

# Run the classification test
if __name__ == "__main__":
    classify_strategies()
