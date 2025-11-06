from datasets import load_dataset, concatenate_datasets
import pandas as pd

# 1. The Confirmed Subjects from the paper's legend
target_subjects = [
    'global_facts',
    'medical_genetics',
    'college_biology',
    'abstract_algebra',
    'management',
    'college_chemistry'
]

all_datasets = []

# 2. Loop through the subjects and load the 'dev' split for training/development
for subject in target_subjects:
    # --- CHANGE 'train' to 'dev' HERE ---
    print(f"Loading the 'dev' split for: {subject}...")
    
    # Load the specific subject/config and the 'dev' split
    subject_data = load_dataset('cais/mmlu', subject, split='dev')
    
    # Add the 'subject' column for later analysis
    subject_data = subject_data.add_column('subject_domain', [subject] * len(subject_data))
    
    all_datasets.append(subject_data)

# 3. Concatenate all training data into one dataset object
raw_training_dataset = concatenate_datasets(all_datasets)

print("\n--- Download Complete ---")
print(f"Total number of training sequences: {len(raw_training_dataset)}")
print(f"Subjects downloaded: {set(raw_training_dataset['subject_domain'])}")