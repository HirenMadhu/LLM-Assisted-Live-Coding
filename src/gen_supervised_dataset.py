import os
import numpy as np
import torch
from datasets import Dataset, Features, Value, Sequence, ClassLabel

from wavEmbed import embed_wav
from codeEmbed import embed_code
from gen_recordings import create_recording


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


if __name__ == "__main__":
    cwd = os.getcwd()
    case_path = "../"
    folders = [
        "s1-mel",
        "s2-drum",
        "s3-bass",
    ]
    subfolds = ["3-cand"]
    temp_dir = "../temp"
    filename_type = ".rb"

    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Process each case folder and subfolder combination
    for folder in folders:
        for subfold in subfolds:
            case_dir = os.path.join(case_path, folder, subfold)
            print(f"\n{'='*50}")
            print(f"Processing case folder: {case_dir}")
            print(f"{'='*50}")

            # Check if case directory exists
            if not os.path.exists(case_dir):
                print(f"Warning: Directory {case_dir} does not exist! Skipping...")
                continue

            # Extract case name as folder-subfolder
            case_name = f"{folder}-{subfold}"

            codes = []
            code_embs = []
            wav_embs = []
            labels = []

            dataset_name = f"label_embs_{case_name}.arrow"

            print(f"Processing files from {case_dir}")

            # Collect all unique labels first
            all_labels = set()
            pi_files = []

            for filename in os.listdir(case_dir):
                if filename.endswith(filename_type):
                    code_name = filename.split(filename_type)[0]
                    # Add case prefix to label
                    label_with_case = f"{case_name}_{code_name}"
                    all_labels.add(label_with_case)
                    pi_files.append((case_dir, filename, code_name, label_with_case))

            if len(pi_files) == 0:
                print(f"No .pi files found in {case_dir}. Skipping...")
                continue

            # Convert to sorted list for consistent label encoding
            label_names = sorted(list(all_labels))
            print(
                f"Found {len(pi_files)} .pi files with {len(label_names)} unique labels"
            )
            print(f"Labels: {label_names}")

            # Process each .pi file
            for dirpath, filename, code_name, label_with_case in pi_files:
                print(f"Processing: {filename}")

                # Create recordings path
                recordings_path = os.path.join(cwd, temp_dir)
                if not os.path.exists(recordings_path):
                    os.makedirs(recordings_path)

                wav_path = os.path.join(recordings_path, f"{code_name}.wav")

                try:
                    # Generate recording
                    create_recording(dirpath, filename, wav_path)

                    # Generate code embedding
                    code_path = os.path.join(dirpath, filename)
                    with open(code_path, "r") as file:
                        code_str = file.read()
                    code_emb = embed_code(code_str)

                    # Generate wav embedding
                    wav_emb = embed_wav(wav_path, limit_dim=True)

                    # Convert to numpy if needed
                    if isinstance(code_emb, torch.Tensor):
                        code_emb = to_numpy(code_emb)

                    if isinstance(wav_emb, torch.Tensor):
                        wav_emb = to_numpy(wav_emb)

                    # Append to lists
                    codes.append(code_str)
                    code_embs.append(code_emb)
                    wav_embs.append(wav_emb)
                    labels.append(label_with_case)

                    print(f"Successfully processed: {filename}")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

                # Clean up temporary wav file
                if os.path.exists(wav_path):
                    os.remove(wav_path)

            if len(codes) == 0:
                print(f"No files were successfully processed for {case_name}!")
                continue

            # Stack into arrays
            code_embs = np.stack(code_embs, axis=0)
            wav_embs = np.stack(wav_embs, axis=0)

            # Build the HF Dataset with labels
            features = Features(
                {
                    "code": Value("string"),
                    "code_embedding": Sequence(feature=Value("float32"), length=768),
                    "wav_embedding": Sequence(feature=Value("float32"), length=768),
                    "label": ClassLabel(names=label_names),
                }
            )

            ds = Dataset.from_dict(
                {
                    "code": codes,
                    "code_embedding": code_embs,
                    "wav_embedding": wav_embs,
                    "label": labels,
                },
                features=features,
            )

            # Save dataset
            dataset_path = os.path.join(case_dir, dataset_name)
            ds.save_to_disk(dataset_path)
            print(
                f"Done! Dataset for {case_name} has {len(ds)} examples and is saved to {dataset_path}"
            )

            # Print dataset statistics
            print(f"\nDataset Statistics for {case_name}:")
            print(f"Total examples: {len(ds)}")
            print(f"Number of unique labels: {len(label_names)}")
            print(f"Code embedding shape: {code_embs.shape}")
            print(f"Wav embedding shape: {wav_embs.shape}")

            # Print label distribution
            from collections import Counter

            label_counts = Counter(labels)
            print(f"\nLabel distribution for {case_name}:")
            for label, count in sorted(label_counts.items()):
                print(f"  {label}: {count} examples")

    print(f"\n{'='*50}")
    print("All folder-subfolder combinations processed!")
    print(f"{'='*50}")

    # Clean up temp directory
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
