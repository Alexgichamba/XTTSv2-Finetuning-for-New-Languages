import os
import pandas as pd
import argparse
from tqdm import tqdm

def convert_dataset_by_language(metadata_path, audio_base_path, output_base_dir, split_type, use_symlinks=True):
    """
    Convert dataset from original format to the required format, organizing by language.
    
    Args:
        metadata_path: Path to the metadata CSV file
        audio_base_path: Path containing all audio files for this split
        output_base_dir: Base output directory
        split_type: Either 'train' or 'eval' to determine output filename
        use_symlinks: If True, create symbolic links instead of copying files
    """
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Group by language
    languages = df['language'].unique()
    
    # Create multilingual dataset with all languages combined
    multilingual_dir = os.path.join(output_base_dir, 'multilingual')
    os.makedirs(os.path.join(multilingual_dir, 'wavs'), exist_ok=True)
    multilingual_metadata = []
    
    # Process all data for multilingual dataset
    for _, row in tqdm(df.iterrows(), 
                      total=len(df), 
                      desc=f"Processing multilingual {split_type} data"):
        # Get source audio path - handle corpus subdirectory if present
        if 'corpus' in row and pd.notna(row['corpus']):
            src_audio_path = os.path.join(audio_base_path, row['corpus'], row['audio_path'])
        else:
            src_audio_path = os.path.join(audio_base_path, row['audio_path'])
        
        # Define target audio filename (keeping original extension)
        filename = os.path.basename(row['audio_path'])
        target_audio_path = os.path.join(multilingual_dir, 'wavs', filename)
        
        # Create symbolic link or copy file
        if os.path.exists(src_audio_path):
            # Remove target if it already exists
            if os.path.exists(target_audio_path) or os.path.islink(target_audio_path):
                os.remove(target_audio_path)
            
            if use_symlinks:
                # Create symbolic link (using absolute paths for reliability)
                src_absolute = os.path.abspath(src_audio_path)
                os.symlink(src_absolute, target_audio_path)
            else:
                # Copy file (original behavior)
                import shutil
                shutil.copy2(src_audio_path, target_audio_path)
            
            # Add entry to multilingual metadata
            multilingual_metadata.append({
                'audio_file': f'wavs/{filename}',
                'text': row['transcription'],
                'speaker_name': f'@{row["speaker_id"]}'
            })
        else:
            print(f"Warning: Could not find audio file {src_audio_path}")
    
    # Save multilingual metadata
    multilingual_df = pd.DataFrame(multilingual_metadata)
    multilingual_df.to_csv(os.path.join(multilingual_dir, f'metadata_{split_type}.csv'), 
                          sep='|', index=False, header=True)
    
    action = "symlinked" if use_symlinks else "copied"
    print(f"Created multilingual dataset with {len(multilingual_df)} entries ({action} files)")
    
    # Process individual language datasets
    for language in languages:
        # Create language-specific output directory
        output_dir = os.path.join(output_base_dir, language)
        os.makedirs(os.path.join(output_dir, 'wavs'), exist_ok=True)
        
        # Filter data for this language
        language_df = df[df['language'] == language]
        
        # Create new metadata dataframe with required format
        new_metadata = []
        
        for _, row in tqdm(language_df.iterrows(), 
                          total=len(language_df), 
                          desc=f"Processing {language} {split_type} data"):
            # Get source audio path - handle corpus subdirectory if present
            if 'corpus' in row and pd.notna(row['corpus']):
                src_audio_path = os.path.join(audio_base_path, row['corpus'], row['audio_path'])
            else:
                src_audio_path = os.path.join(audio_base_path, row['audio_path'])
            
            # Define target audio filename (keeping original extension)
            filename = os.path.basename(row['audio_path'])
            target_audio_path = os.path.join(output_dir, 'wavs', filename)
            
            # Create symbolic link or copy file
            if os.path.exists(src_audio_path):
                # Remove target if it already exists
                if os.path.exists(target_audio_path) or os.path.islink(target_audio_path):
                    os.remove(target_audio_path)
                
                if use_symlinks:
                    # Create symbolic link (using absolute paths for reliability)
                    src_absolute = os.path.abspath(src_audio_path)
                    os.symlink(src_absolute, target_audio_path)
                else:
                    # Copy file (original behavior)
                    import shutil
                    shutil.copy2(src_audio_path, target_audio_path)
                
                # Add entry to new metadata
                new_metadata.append({
                    'audio_file': f'wavs/{filename}',
                    'text': row['transcription'],
                    'speaker_name': f'@{row["speaker_id"]}'
                })
            else:
                print(f"Warning: Could not find audio file {src_audio_path}")
        
        # Create new metadata dataframe
        new_df = pd.DataFrame(new_metadata)
        
        # Save metadata with pipe separator and no index
        output_file = f'metadata_{split_type}.csv'
        
        new_df.to_csv(os.path.join(output_dir, output_file), 
                      sep='|', 
                      index=False, 
                      header=True)
        
        action = "symlinked" if use_symlinks else "copied"
        print(f"Converted {len(new_df)} entries to {language}/{output_file} ({action} files)")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to required format by language')
    parser.add_argument('--train_audio_path', help='Base path containing training audio files')
    parser.add_argument('--train_metadata', help='Path to the training metadata CSV file')
    parser.add_argument('--eval_audio_path', help='Base path containing evaluation audio files')
    parser.add_argument('--eval_metadata', help='Path to the evaluation metadata CSV file')
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    parser.add_argument('--copy_files', action='store_true', 
                       help='Copy files instead of creating symbolic links (default: use symlinks)')
    
    args = parser.parse_args()
    
    use_symlinks = not args.copy_files
    
    if use_symlinks:
        print("Using symbolic links (saves disk space)")
    else:
        print("Copying files (original behavior)")
    
    # Process training data if provided
    if args.train_metadata and args.train_audio_path:
        convert_dataset_by_language(args.train_metadata, args.train_audio_path, 
                                   args.output_dir, 'train', use_symlinks)
    
    # Process evaluation data if provided
    if args.eval_metadata and args.eval_audio_path:
        convert_dataset_by_language(args.eval_metadata, args.eval_audio_path, 
                                   args.output_dir, 'eval', use_symlinks)
    
    print(f"Dataset conversion complete. Output in {args.output_dir}")

if __name__ == "__main__":
    main()