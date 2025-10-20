import os
from reference_maker import build_references_from_dataset

def main():
    print("Building reference models from dataset...")
    
    # Path to the dataset directory with exercise videos
    dataset_path = "../data/final_kaggle_with_additional_video"
    
    # Output directory for reference models
    output_dir = "models/references"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        print("Please make sure your data directory contains the exercise videos")
        return
    
    print(f"Processing videos from: {dataset_path}")
    print(f"Saving reference models to: {output_dir}")
    
    # Build references from the dataset
    build_references_from_dataset(dataset_path, output_dir)
    
    print("Reference model building complete!")

if __name__ == "__main__":
    main()