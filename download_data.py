"""Script to download the Tiny Shakespeare dataset."""
import os
import requests

# URL for the Tiny Shakespeare dataset
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "input.txt")

def download_shakespeare_data():
    """Downloads the Tiny Shakespeare dataset and saves it to data/input.txt."""
    print(f"Attempting to download Shakespeare dataset from {DATA_URL}")

    try:
        # Create the data directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created directory: {OUTPUT_DIR}")

        # Download the data
        response = requests.get(DATA_URL, timeout=30) # Added timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        # Save the data to the file
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"Successfully downloaded and saved dataset to {OUTPUT_FILE}")
        print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
    except OSError as e:
        print(f"Error creating directory or writing file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_shakespeare_data()
