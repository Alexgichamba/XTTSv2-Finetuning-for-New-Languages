import torch
import torchaudio
import argparse
import os
import sys
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Text-to-Speech using XTTS model')
    parser.add_argument('--text', '-t', type=str, required=True, 
                        help='Text to synthesize')
    parser.add_argument('--speaker', '-s', type=str, required=True,
                        help='Path to speaker audio file')
    parser.add_argument('--language', '-l', type=str, required=True,
                        help='Language code (e.g., "multi", "en", "es", etc.)')
    parser.add_argument('--output', '-o', type=str, default='output.wav',
                        help='Output audio file name (default: output.wav)')
    parser.add_argument('--model-checkpoint', type=str, 
                        default='../export_checkpoint/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--model-config', type=str,
                        default='../export_checkpoint/XTTS_v2.0_original_model_files/config.json',
                        help='Path to model config file')
    parser.add_argument('--model-vocab', type=str,
                        default='../export_checkpoint/XTTS_v2.0_original_model_files/vocab.json',
                        help='Path to model vocabulary file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.speaker):
        print(f"Error: Speaker audio file not found: {args.speaker}")
        sys.exit(1)
    
    if not os.path.exists(args.model_checkpoint):
        print(f"Error: Model checkpoint not found: {args.model_checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.model_config):
        print(f"Error: Model config not found: {args.model_config}")
        sys.exit(1)
    
    if not os.path.exists(args.model_vocab):
        print(f"Error: Model vocab not found: {args.model_vocab}")
        sys.exit(1)
    
    # Device configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    config = XttsConfig()
    config.load_json(args.model_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=args.model_checkpoint, 
                               vocab_path=args.model_vocab, use_deepspeed=False)
    XTTS_MODEL.to(device)
    
    print("Model loaded successfully!")
    
    # Get conditioning latents from speaker audio
    print("Processing speaker audio...")
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=args.speaker,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )
    
    # Tokenize text into sentences
    tts_texts = sent_tokenize(args.text)
    print(f"Processing {len(tts_texts)} sentences...")
    
    # Generate audio for each sentence
    wav_chunks = []
    for text in tqdm(tts_texts, desc="Generating audio"):
        wav_chunk = XTTS_MODEL.inference(
            text=text,
            language=args.language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=10,
            top_p=0.3,
        )
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))
    
    # Concatenate all audio chunks
    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()
    
    # Save the output wav
    print(f"Saving audio to: {args.output}")
    torchaudio.save(
        args.output,
        out_wav,
        XTTS_MODEL.config.audio.output_sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    
    print("Audio generation completed successfully!")

if __name__ == "__main__":
    main()