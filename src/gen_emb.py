import os
import numpy as np
import torch
from datasets import Dataset, Features, Value, Sequence

from wavEmbed import embed_wav
from codeEmbed import embed_code
from gen_recordings import create_recording

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


if __name__ == "__main__":
    codes      = []
    code_embs  = []
    wav_embs   = []

    dataset_name = "sonicpi_embeddings.arrow"

    cwd = os.getcwd()
    dataset_dir = "../datasets/"
    temp_dir = "../temp"

    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for filename in filenames:  

            # get code name without .pi extension
            code_name = filename.split(".pi")[0]

            dir_name = os.path.basename(dirpath)

            recordings_path = os.path.join(cwd, temp_dir, dir_name)
            if not os.path.exists(recordings_path):
                    os.makedirs(recordings_path)

            wav_path = os.path.join(
                    cwd, recordings_path, f"{code_name}.wav"
                )

            create_recording(dirpath, filename, wav_path)

            wav_path = os.path.join(
                cwd, temp_dir, dir_name, f"{code_name}.wav"
            )

            # generate code_embedding
            code_path = os.path.join(dirpath, filename)
            with open(code_path, "r") as file:
                code_str = file.read()
            code_emb = embed_code(code_str)

            # generate wav_embedding
            wav_emb = embed_wav(wav_path, limit_dim=True)

            if isinstance(code_emb, torch.Tensor):
                code_emb = to_numpy(code_emb)

            if isinstance(wav_emb, torch.Tensor):
                wav_emb = to_numpy(wav_emb)

            # append to lists
            codes.append(code_str)
            code_embs.append(code_emb)
            wav_embs.append(wav_emb)

        # empty temp directory for efficiency
        for dirpath, dirnames, filenames in os.walk(temp_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)

    # Stack into arrays
    code_embs = np.stack(code_embs, axis=0)
    wav_embs = np.stack(wav_embs, axis=0)

    # Build the HF Dataset
    features = Features({
        "code":           Value("string"),
        "code_embedding": Sequence(feature=Value("float32"), length=768),
        "wav_embedding":  Sequence(feature=Value("float32"), length=768),
    })
    ds = Dataset.from_dict({
        "code":           codes,
        "code_embedding": code_embs,
        "wav_embedding":  wav_embs
    }, features=features)

    dataset_path = os.path.join(cwd, "..", dataset_name)
    ds.save_to_disk(dataset_path)
    print(f"Done! Dataset has {len(ds)} examples and is saved to {dataset_path}")

    # Delete temp directory
    for dirpath, dirnames, filenames in os.walk(temp_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            os.remove(file_path)
    os.rmdir(temp_dir)
            
            

