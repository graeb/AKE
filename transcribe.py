import torch
import whisperx
import gc


AUDIO_PATH = r"/home/oskar52/code/video_transcription/audio/shitcoin.mp3"
HF_TOKEN = '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'


def transcribe_with_diarization(audio_file: str) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1  # reduce if low on GPU mem
    compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("base", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    gc.collect()
    torch.cuda.empty_cache()
    del model

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
        )
    result = whisperx.align(
        result["segments"],
        model_a, metadata, audio, device,
        return_char_alignments=False
        )

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio_file)
    # diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result["segments"]


def transcribe(audio_file: str) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float32"

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("small", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    return result["segments"]
