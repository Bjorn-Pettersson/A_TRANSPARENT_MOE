from datasets import load_dataset, concatenate_datasets
import os

# --- 1. Set Custom Cache Directory ---
custom_cache_path = r'D:\OneDrive - University of Copenhagen\SNU\ADVANCED_DEEP_LEARNING\TRANSPARENT_MOE\MMLU_EXPERIMENTS'
os.environ['HF_DATASETS_CACHE'] = custom_cache_path

print(f"Hugging Face datasets cache set to: {custom_cache_path}")
print("Starting download of the massive AUXILIARY_TRAIN set...")
print("---" * 15)

# --- 2. Load the entire (subject-agnostic) auxiliary_train split ---
# We pass 'auxiliary_train' as the configuration name (second argument)
# and 'train' as the split name (the only split in this configuration)
try:
    raw_fine_tuning_dataset = load_dataset('cais/mmlu', 'auxiliary_train', split='train')
    
    # You can inspect the first few subjects if you want, but this set is massive.
    print("\n✅ Fine-Tuning Data Download Complete")
    print(f"Total FINE-TUNING sequences ready for SFT: {len(raw_fine_tuning_dataset)}")

except Exception as e:
    print(f"\n❌ ERROR during Fine-Tuning Data download: {e}")


# --- 3. Run the TEST set download (to confirm cache is working) ---
# Your previous attempt to download the TEST split was successful, 
# so we run it again to ensure the path is working and files are local.

target_subjects = [
    'global_facts',
    'medical_genetics',
    'college_biology',
    'abstract_algebra',
    'management',
    'college_chemistry'
]

test_datasets = []
total_test_samples = 0
print("\nStarting download of TEST data (for final evaluation) to confirm path...")

for subject in target_subjects:
    # This correctly uses the subject name as the config
    subject_data = load_dataset('cais/mmlu', subject, split='test')
    subject_data = subject_data.add_column('subject_domain', [subject] * len(subject_data))
    test_datasets.append(subject_data)
    total_test_samples += len(subject_data)
    # This will likely load from your cache now
    # print(f"  > Loaded test split for: {subject}") 


raw_test_dataset = concatenate_datasets(test_datasets)
print("\n✅ Test Data Load Complete")
print(f"Total TEST sequences (for final evaluation): {total_test_samples}")