# Voicemeeter Auto Transcriber with Speaker Diarization

Automatically monitor Voicemeeter recordings, transcribe audio using Faster-Whisper, identify speakers using PyAnnote, and generate clean Markdown transcripts for Obsidian.

## Features

- Folder watcher for new Voicemeeter recordings
- Automatically waits for recording completion
- GPU-accelerated transcription using Faster-Whisper
- Speaker diarization with PyAnnote
- Consecutive speaker segments grouped together
- Automatic retry handling
- Crash recovery support
- Persistent processing database
- Markdown transcript output
- Obsidian-friendly formatting
- CUDA acceleration for RTX GPUs

## Example Workflow

1. Start recording in Voicemeeter
2. Recording is saved to the monitored folder
3. Script detects the new WAV file
4. Script waits until recording activity stops
5. Speaker diarization runs on GPU
6. Whisper transcribes speech
7. Speaker labels are applied
8. Transcript is saved as Markdown
9. Recording status is stored in the database

No manual intervention required.

---

## Output Example

```markdown
# Meeting Recording

## SPEAKER_00

[00:00] Good morning everyone.

## SPEAKER_01

[00:05] Morning.

## SPEAKER_00

[00:08] Let's review the project status.

---

## Transcript Statistics

- Duration: 32m 18s
- Language: en
- Speakers: 2
