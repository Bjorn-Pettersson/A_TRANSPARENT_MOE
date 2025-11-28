"""
MMLU Dataset Downloader and Viewer
Download and explore the MMLU benchmark dataset with pandas
"""

import pandas as pd
from datasets import load_dataset

def download_mmlu():
    """Download the MMLU dataset from Hugging Face"""
    print("Downloading MMLU dataset...")
    print("This may take a few minutes on first run...")
    
    # Load the dataset
    dataset = load_dataset("cais/mmlu", "all")
    
    return dataset

def dataset_to_dataframe(dataset, split='test'):
    """Convert dataset to pandas DataFrame"""
    print(f"\nConverting {split} split to DataFrame...")
    
    # Convert to pandas
    df = pd.DataFrame(dataset[split])
    
    # Rename columns for clarity
    df = df.rename(columns={
        'question': 'Question',
        'subject': 'Category',
        'choices': 'Choices',
        'answer': 'Answer'
    })
    
    # Convert answer index to letter (0->A, 1->B, etc.)
    df['Answer_Letter'] = df['Answer'].apply(lambda x: chr(65 + x))
    
    return df

def display_stats(df):
    """Display dataset statistics"""
    print("\n" + "="*60)
    print("MMLU DATASET STATISTICS")
    print("="*60)
    print(f"Total questions: {len(df)}")
    print(f"Number of categories: {df['Category'].nunique()}")
    print(f"\nQuestions per category:")
    print(df['Category'].value_counts().sort_index())
    print("="*60)

def filter_by_category(df, category):
    """Filter DataFrame by category"""
    if category not in df['Category'].values:
        print(f"\nCategory '{category}' not found!")
        print("\nAvailable categories:")
        print(df['Category'].unique())
        return None
    
    filtered_df = df[df['Category'] == category].copy()
    print(f"\nFiltered to category: {category}")
    print(f"Number of questions: {len(filtered_df)}")
    return filtered_df

def save_to_csv(df, filename='mmlu_data.csv'):
    """Save DataFrame to CSV"""
    df.to_csv(filename, index=False)
    print(f"\nData saved to {filename}")

def main():
    # Download dataset
    dataset = download_mmlu()
    
    # Convert to DataFrame
    df = dataset_to_dataframe(dataset, split='test')
    
    # Display statistics
    display_stats(df)
    
    # Show first few rows
    print("\nFirst 5 questions:")
    print(df[['Category', 'Question', 'Answer_Letter']].head())
    
    # Example: Filter by category
    print("\n" + "="*60)
    print("EXAMPLE: Filtering by 'clinical_knowledge'")
    print("="*60)
    clinical_df = filter_by_category(df, 'clinical_knowledge')
    
    if clinical_df is not None:
        print("\nFirst 3 clinical knowledge questions:")
        for idx, row in clinical_df.head(3).iterrows():
            print(f"\nQ: {row['Question']}")
            for i, choice in enumerate(row['Choices']):
                print(f"  {chr(65+i)}. {choice}")
            print(f"Answer: {row['Answer_Letter']}")
    
    # Save full dataset
    save_to_csv(df, 'mmlu_full_test.csv')
    
    # Save filtered dataset
    if clinical_df is not None:
        save_to_csv(clinical_df, 'mmlu_clinical_knowledge.csv')
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    print("""
# Filter by any category:
biology_df = filter_by_category(df, 'high_school_biology')

# Search in questions:
keyword_df = df[df['Question'].str.contains('protein', case=False)]

# Get specific categories:
stem_categories = ['physics', 'chemistry', 'biology', 'computer_science']
stem_df = df[df['Category'].isin(stem_categories)]

# Export filtered data:
save_to_csv(biology_df, 'biology_questions.csv')
    """)
    
    return df

if __name__ == "__main__":
    df = main()
    print("\nDataFrame stored in variable 'df' for interactive use!")