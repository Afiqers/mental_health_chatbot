import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
import time
import os

def translate_to_malay(text):
    try:
        # Google Translate API has limits, we truncate to 500 chars to be safe and fast
        translated = GoogleTranslator(source='en', target='ms').translate(text[:500])
        time.sleep(0.1)  # small delay to prevent rate limiting
        return translated
    except Exception as e:
        return text  # fallback to original English if it fails

def main():
    input_file = "data/mental_heath_unbalanced.csv"
    output_file = "data/malay_dataset_translated.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please ensure it's downloaded.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # The dataset has 48k rows. We take a balanced sample of 500 per class (2000 total)
    # Translating 48k rows via free API takes days and gets blocked.
    print("Sampling 500 examples per class to translate...")
    df_sampled = df.groupby('status').apply(lambda x: x.sample(n=min(500, len(x)), random_state=42)).reset_index(drop=True)
    
    # Create the label mapping as expected by our backend
    # The original dataset labels are: Suicidal, Depression, Anxiety, Normal
    # We will map Suicidal -> STRESS (or we could update the backend, but let's stick to the 4 classes)
    # Actually, we can keep the original labels and just update the backend later.
    
    print(f"Translating {len(df_sampled)} rows to Malay... This will take a few minutes.")
    tqdm.pandas()
    df_sampled['text_ms'] = df_sampled['text'].progress_apply(translate_to_malay)
    
    # Save the new dataset
    df_sampled.to_csv(output_file, index=False)
    print(f"Done! Translated dataset saved to {output_file}")

if __name__ == "__main__":
    main()
