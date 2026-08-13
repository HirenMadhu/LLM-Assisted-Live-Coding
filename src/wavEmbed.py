from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
import soundfile as sf
import numpy as np
import librosa


model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()  # Set the model to evaluation mode
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def load_audio(wav_file_path):
    # Load audio file
    audio, sample_rate = sf.read(wav_file_path)
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
    
    return audio, 16000  # Always return 16kHz sample rate

def embed_wav(wav_file_path, limit_dim=False):
    # Load and process audio
    audio, sample_rate = load_audio(wav_file_path)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling to get a single vector for the entire audio
        embeddings = outputs.last_hidden_state.mean(dim=1)

    if limit_dim:
        embeddings = embeddings.mean(dim=0)  
    
    return embeddings

def compute_wav_similarity(wav_file1, wav_file2):
    # Get embeddings for both files
    emb1 = embed_wav(wav_file1)
    emb2 = embed_wav(wav_file2)
    
    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
    return float(similarity[0]) 