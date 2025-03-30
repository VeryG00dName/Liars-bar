#!/usr/bin/env python3
import pickle
import pprint

def main():
    # Path to the pickle file (adjust the path if needed)
    filename = "ps_data/ps_data_checkpoint_70000.pkl"
    
    # Load the data from the pickle file
    try:
        with open(filename, "rb") as file:
            data = pickle.load(file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Check if the data is a list and has elements
    if isinstance(data, list) and len(data) > 0:
        print("Displaying two sample entries from the data:\n")
        for i, sample in enumerate(data[:5]):
            print(f"Sample {i+1}:")
            pprint.pprint(sample)
            print("\n" + "-"*40 + "\n")
    else:
        print("The data format is not as expected or the file is empty.")

if __name__ == "__main__":
    main()
