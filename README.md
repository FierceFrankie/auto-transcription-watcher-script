# Voicemeeter Auto Transcriber with Speaker Diarization

Automatically monitor Voicemeeter recordings, transcribe audio using Faster-Whisper (or Parakeet), identify speakers using PyAnnote, and generate clean Markdown transcripts for Obsidian.

## Features

- Folder watcher for new Voicemeeter recordings
- Automatically waits for recording completion
- GPU-accelerated transcription using Faster-Whisper
- Optional NVIDIA Parakeet ASR backend
- Speaker diarization with PyAnnote
- Consecutive speaker segments grouped together
- **Stable speaker IDs preserved** (`speaker_id`)
- **Custom speaker name mapping** from `config/speakers.json`
- **Fallback behavior**: if no mapping exists, uses raw `speaker_id`
- **Draft transcript JSON output** for correction/re-render workflows
- **Re-render without re-transcription** (update names and regenerate Markdown)
- **Speaker auto-suggestions scaffold** from:
  - prior confirmed mappings (`config/speaker_history.json`)
  - attendee list (`config/attendees.txt`)
- Automatic retry handling
- Crash recovery support
- Persistent processing database
- Markdown transcript output
- Obsidian-friendly formatting
- CUDA acceleration for RTX GPUs

## New Speaker Mapping + Correction Workflow

### 1) First pass transcription

When a new `.wav` is processed, the script now outputs:

- `*_transcript.md` (human-readable transcript)
- `*_transcript_draft.json` (stable diarized segments with immutable `speaker_id`s)

The draft JSON includes:

- segment-level `speaker_id`, `start`, `end`, `text`
- current `speaker_map`
- optional speaker name `suggestions`

### 2) Review / rename speakers

Edit `config/speakers.json` and map stable IDs to real names.

```json
{
  "speaker_0": "Alice Johnson",
  "speaker_1": "Bob Lee"
}
```

### 3) Re-render only (no re-transcription)

Run in re-render mode against a draft file. This skips diarization/ASR and only regenerates Markdown with updated names.

---

## Configuration Files

### `config/speakers.json`
Canonical speaker display names for stable IDs.

```json
{
  "speaker_0": "Alice Johnson",
  "speaker_1": "Bob Lee"
}
```

### `config/speaker_history.json`
Maintains previously confirmed mappings and confidence values for future auto-suggestion.

Example shape:

```json
{
  "speaker_0": {
    "canonical_name": "Alice Johnson",
    "confidence": 1.0
  }
}
```

### `config/attendees.txt`
Optional attendee list (one name per line) used as low-confidence suggestions.

```text
Alice Johnson
Bob Lee
Charlie Kim
```

---

## Environment Variables

- `HF_TOKEN` (required)
- `ASR_BACKEND` (`whisper` or `parakeet`, default `whisper`)
- `SPEAKER_MAP_FILE` (optional, default `config/speakers.json`)
- `SPEAKER_HISTORY_FILE` (optional, default `config/speaker_history.json`)
- `ATTENDEE_LIST_FILE` (optional, default `config/attendees.txt`)
- `RE_RENDER_ONLY` (`1` to enable re-render mode)
- `RE_RENDER_SOURCE` (path to `*_transcript_draft.json` when re-rendering)

---

## Example Workflow

1. Start recording in Voicemeeter
2. Recording is saved to the monitored folder
3. Script detects the new WAV file
4. Script waits until recording activity stops
5. Speaker diarization runs on GPU
6. ASR transcribes speech
7. Stable `speaker_id`s are attached to transcript segments
8. Draft JSON is created (`*_transcript_draft.json`)
9. Speaker mapping is applied (`config/speakers.json`)
10. Markdown transcript is generated
11. Recording status is stored in the database

Optional correction loop:

12. Edit `config/speakers.json`
13. Re-render from draft JSON (no re-transcription)

---

## Re-render Examples

### PowerShell

```powershell
$env:RE_RENDER_ONLY="1"
$env:RE_RENDER_SOURCE="C:\Users\Frank\Documents\Voicemeeter\Meeting_2026-09-01_transcript_draft.json"
python .\watch_voicemeeter_diarization.py
```

### CMD

```cmd
set RE_RENDER_ONLY=1
set RE_RENDER_SOURCE=C:\Users\Frank\Documents\Voicemeeter\Meeting_2026-09-01_transcript_draft.json
python watch_voicemeeter_diarization.py
```

### Bash

```bash
RE_RENDER_ONLY=1 \
RE_RENDER_SOURCE="/path/to/Meeting_2026-09-01_transcript_draft.json" \
python watch_voicemeeter_diarization.py
```

---

## Output Example (Markdown)

```markdown
# Meeting Recording

## Alice Johnson

[00:00] Good morning everyone.

## Bob Lee

[00:05] Morning.

## Alice Johnson

[00:08] Let's review the project status.

---

## Transcript Statistics

- Duration: 32m 18s
- Language: en
- Speakers: 2
```

## Output Example (Draft JSON)

```json
{
  "source_audio": "C:\\Users\\Frank\\Documents\\Voicemeeter\\Meeting_2026-09-01.wav",
  "generated_at": "2026-09-01T06:56:00.000000",
  "segments": [
    {
      "speaker_id": "speaker_0",
      "start": 0.0,
      "end": 4.2,
      "text": "Good morning everyone."
    },
    {
      "speaker_id": "speaker_1",
      "start": 4.3,
      "end": 6.1,
      "text": "Morning."
    }
  ],
  "speaker_map": {
    "speaker_0": "Alice Johnson",
    "speaker_1": "Bob Lee"
  },
  "suggestions": {
    "speaker_2": {
      "suggested_name": "Charlie Kim",
      "source": "attendees",
      "confidence": 0.5
    }
  }
}
```
