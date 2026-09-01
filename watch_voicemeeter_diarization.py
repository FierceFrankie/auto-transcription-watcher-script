import json
import os
import time
from pathlib import Path
from queue import Queue
from threading import Thread, Lock
from datetime import datetime
import soundfile as sf
import torch

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from faster_whisper import WhisperModel
import nemo.collections.asr as nemo_asr
from pyannote.audio import Pipeline
import whisperx

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

WATCH_FOLDER = Path(
    r"C:\Users\Frank\Documents\Voicemeeter"
)

DATABASE_FILE = Path(
    r"C:\AI\voicemeeter_processed.json"
)

SPEAKER_MAP_FILE = Path(
    os.getenv(
        "SPEAKER_MAP_FILE",
        "config/speakers.json"
    )
)

SPEAKER_HISTORY_FILE = Path(
    os.getenv(
        "SPEAKER_HISTORY_FILE",
        "config/speaker_history.json"
    )
)

ATTENDEE_LIST_FILE = Path(
    os.getenv(
        "ATTENDEE_LIST_FILE",
        "config/attendees.txt"
    )
)

RECORDING_IDLE_SECONDS = 30

MAX_RETRIES = 3
RETRY_DELAY = 30

# ASR backend: "whisper", "parakeet", or "whisperx"
ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper").strip().lower()
if ASR_BACKEND not in {"whisper", "parakeet", "whisperx"}:
    raise RuntimeError(
        "ASR_BACKEND must be 'whisper', 'parakeet', or 'whisperx'"
    )

# --------------------------------------------------
# HF TOKEN CHECK
# --------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable not set"
    )

# --------------------------------------------------
# WORK QUEUE
# --------------------------------------------------

WORK_QUEUE = Queue()

queued_files = set()
active_files = set()

queue_lock = Lock()

# --------------------------------------------------
# PERFORMANCE
# --------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# --------------------------------------------------
# SPEAKER MAPPING / SUGGESTION HELPERS
# --------------------------------------------------

def load_json_file(path, default):
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: could not read {path}: {e}")
        return default


def save_json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_name(value):
    return " ".join(value.strip().split())


def load_speaker_map():
    mapping = load_json_file(SPEAKER_MAP_FILE, {})
    if not isinstance(mapping, dict):
        print(
            f"WARNING: {SPEAKER_MAP_FILE} must be a JSON object; using empty mapping"
        )
        return {}

    cleaned = {}
    for speaker_id, name in mapping.items():
        if not isinstance(speaker_id, str) or not isinstance(name, str):
            continue
        cleaned[speaker_id] = normalize_name(name)

    return cleaned


def load_speaker_history():
    history = load_json_file(SPEAKER_HISTORY_FILE, {})
    if not isinstance(history, dict):
        print(
            f"WARNING: {SPEAKER_HISTORY_FILE} must be a JSON object; using empty history"
        )
        return {}

    cleaned = {}
    for key, value in history.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue

        canonical_name = value.get("canonical_name")
        if not isinstance(canonical_name, str):
            continue

        confidence = value.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        cleaned[key] = {
            "canonical_name": normalize_name(canonical_name),
            "confidence": max(0.0, min(1.0, confidence))
        }

    return cleaned


def load_attendees():
    if not ATTENDEE_LIST_FILE.exists():
        return []

    names = []
    try:
        with open(ATTENDEE_LIST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                name = normalize_name(line)
                if name:
                    names.append(name)
    except Exception as e:
        print(f"WARNING: could not read {ATTENDEE_LIST_FILE}: {e}")

    return names


def suggest_speaker_names(
    stable_speaker_ids,
    speaker_map,
    speaker_history,
    attendees
):
    suggestions = {}

    # First pass: previously confirmed/stored mapping history
    for stable_id in stable_speaker_ids:
        history_entry = speaker_history.get(stable_id)
        if not history_entry:
            continue

        suggested_name = history_entry.get("canonical_name")
        confidence = history_entry.get("confidence", 0.0)

        if suggested_name and confidence >= 0.8:
            suggestions[stable_id] = {
                "suggested_name": suggested_name,
                "source": "history",
                "confidence": confidence
            }

    # Second pass: fallback to attendee list in stable-id order
    available_attendees = [
        a for a in attendees
        if a not in set(speaker_map.values())
    ]

    attendee_index = 0
    for stable_id in stable_speaker_ids:
        if stable_id in speaker_map or stable_id in suggestions:
            continue

        if attendee_index < len(available_attendees):
            suggestions[stable_id] = {
                "suggested_name": available_attendees[attendee_index],
                "source": "attendees",
                "confidence": 0.5
            }
            attendee_index += 1

    return suggestions


def save_draft_speaker_artifacts(
    audio_file,
    diarized_segments,
    speaker_map,
    suggestions
):
    draft = {
        "source_audio": str(audio_file),
        "generated_at": datetime.now().isoformat(),
        "segments": diarized_segments,
        "speaker_map": speaker_map,
        "suggestions": suggestions,
        "how_to_update": (
            "Edit config/speakers.json to set display names, "
            "then run with RE_RENDER_ONLY=1 RE_RENDER_SOURCE=<path to this draft json>."
        )
    }

    draft_file = audio_file.with_name(
        f"{audio_file.stem}_transcript_draft.json"
    )

    with open(draft_file, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=2)

    return draft_file


def render_markdown_from_draft(
    source_audio,
    diarized_segments,
    output_file,
    language,
    speaker_map
):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write("type: meeting-transcript\n")
        f.write("source: voicemeeter\n")
        f.write("---\n\n")

        f.write(f"# {Path(source_audio).stem}\n\n")

        current_speaker = None
        current_start = None
        speaker_text = []

        for segment in diarized_segments:
            speaker_id = segment["speaker_id"]
            display_name = speaker_map.get(
                speaker_id,
                speaker_id
            )

            if current_speaker is None:
                current_speaker = display_name
                current_start = int(segment["start"])

            elif display_name != current_speaker:
                minutes = current_start // 60
                seconds = current_start % 60

                f.write(f"## {current_speaker}\n\n")
                f.write(
                    f"[{minutes:02}:{seconds:02}] "
                    + " ".join(speaker_text)
                    + "\n\n"
                )

                current_speaker = display_name
                current_start = int(segment["start"])
                speaker_text = []

            speaker_text.append(segment["text"].strip())

        if speaker_text:
            minutes = current_start // 60
            seconds = current_start % 60

            f.write(f"## {current_speaker}\n\n")
            f.write(
                f"[{minutes:02}:{seconds:02}] "
                + " ".join(speaker_text)
                + "\n\n"
            )

        speaker_ids = sorted(
            set(seg["speaker_id"] for seg in diarized_segments)
        )

        duration_seconds = 0
        if diarized_segments:
            duration_seconds = int(diarized_segments[-1]["end"])

        f.write("---\n\n")
        f.write("## Transcript Statistics\n\n")

        f.write(
            f"- Duration: {duration_seconds // 60}m "
            f"{duration_seconds % 60}s\n"
        )
        f.write(f"- Language: {language}\n")
        f.write(f"- Speakers: {len(speaker_ids)}\n\n")

        f.write("## Speakers\n\n")
        for speaker_id in speaker_ids:
            display_name = speaker_map.get(speaker_id, speaker_id)
            if display_name == speaker_id:
                f.write(f"- {speaker_id}\n")
            else:
                f.write(f"- {display_name} ({speaker_id})\n")


def refresh_speaker_history(
    speaker_history,
    speaker_map,
    stable_speaker_ids
):
    for speaker_id in stable_speaker_ids:
        if speaker_id in speaker_map:
            speaker_history[speaker_id] = {
                "canonical_name": speaker_map[speaker_id],
                "confidence": 1.0
            }

    save_json_file(SPEAKER_HISTORY_FILE, speaker_history)


def re_render_from_draft_file(draft_path):
    draft_path = Path(draft_path)
    if not draft_path.exists():
        raise FileNotFoundError(f"Draft file not found: {draft_path}")

    draft = load_json_file(draft_path, None)
    if not isinstance(draft, dict):
        raise RuntimeError(f"Invalid draft JSON: {draft_path}")

    source_audio = draft.get("source_audio")
    diarized_segments = draft.get("segments", [])

    if not source_audio or not isinstance(diarized_segments, list):
        raise RuntimeError("Draft missing source_audio or segments")

    speaker_map = load_speaker_map()

    output_file = Path(source_audio).with_name(
        f"{Path(source_audio).stem}_transcript.md"
    )

    render_markdown_from_draft(
        source_audio,
        diarized_segments,
        output_file,
        language="en",
        speaker_map=speaker_map
    )

    stable_speaker_ids = sorted(
        set(seg.get("speaker_id", "UNKNOWN") for seg in diarized_segments)
    )

    speaker_history = load_speaker_history()
    refresh_speaker_history(
        speaker_history,
        speaker_map,
        stable_speaker_ids
    )

    print(f"Re-rendered transcript: {output_file}")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

whisper = None
parakeet = None
whisperx_model = None
whisperx_diarize = None

if ASR_BACKEND == "whisper":
    print("Loading Whisper Large-v3-Turbo...")
    whisper = WhisperModel(
        "large-v3-turbo",
        device="cuda",
        compute_type="float16"
    )
elif ASR_BACKEND == "parakeet":
    print("Loading Parakeet TDT 0.6B V2 (En)...")
    parakeet = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    parakeet = parakeet.to("cuda")
elif ASR_BACKEND == "whisperx":
    print("Loading WhisperX Large-v3...")
    whisperx_model = whisperx.load_model(
        "large-v3",
        "cuda",
        compute_type="float16",
        language="en"
    )
    print("Loading WhisperX diarization...")
    whisperx_diarize = whisperx.DiarizationPipeline(
        use_auth_token=HF_TOKEN,
        device="cuda"
    )

pipeline = None
if ASR_BACKEND != "whisperx":
    print("Loading PyAnnote...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )
    pipeline.to(torch.device("cuda"))

print("Models loaded.")


class Segment:
    def __init__(self, start, end, text):
        self.start = float(start)
        self.end = float(end)
        self.text = text


class TranscriptionInfo:
    def __init__(self, language="en"):
        self.language = language


def transcribe_with_whisper(audio_file):
    segments, info = whisper.transcribe(
        str(audio_file),
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False
    )
    return list(segments), info


def transcribe_with_parakeet(audio_file):
    """
    Convert Parakeet word timestamps into segment-like chunks
    so the existing downstream speaker grouping logic can remain unchanged.
    """
    result = parakeet.transcribe(
        [str(audio_file)],
        timestamps=True
    )[0]

    words = []
    transcript_text = ""

    if isinstance(result, dict):
        words = result.get("words", []) or []
        transcript_text = result.get("text", "") or ""
    else:
        transcript_text = str(result)

    segments = []

    if words:
        current_words = []
        seg_start = None
        seg_end = None
        max_gap_seconds = 0.8

        for w in words:
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", w_start))
            w_text = (w.get("word", "") or "").strip()

            if not w_text:
                continue

            if seg_start is None:
                seg_start = w_start
                seg_end = w_end
                current_words = [w_text]
                continue

            if (w_start - seg_end) <= max_gap_seconds:
                current_words.append(w_text)
                seg_end = w_end
            else:
                segments.append(
                    Segment(
                        seg_start,
                        seg_end,
                        " ".join(current_words)
                    )
                )
                seg_start = w_start
                seg_end = w_end
                current_words = [w_text]

        if current_words:
            segments.append(
                Segment(
                    seg_start if seg_start is not None else 0.0,
                    seg_end if seg_end is not None else 0.0,
                    " ".join(current_words)
                )
            )
    elif transcript_text.strip():
        segments = [Segment(0.0, 0.0, transcript_text.strip())]

    info = TranscriptionInfo(language="en")
    return segments, info


def transcribe_with_whisperx(audio_file):
    audio = whisperx.load_audio(str(audio_file))
    result = whisperx_model.transcribe(audio, batch_size=16)

    diarization = whisperx_diarize(audio)
    result = whisperx.assign_word_speakers(
        diarization,
        result
    )

    diarized_segments = []
    for segment in result.get("segments", []):
        text = (segment.get("text", "") or "").strip()
        if not text:
            continue

        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)

        diarized_segments.append(
            {
                "speaker_id": segment.get("speaker", "UNKNOWN"),
                "start": start,
                "end": end,
                "text": text
            }
        )

    info = TranscriptionInfo(
        language=result.get("language", "en")
    )
    return diarized_segments, info

# --------------------------------------------------
# DATABASE FUNCTIONS
# --------------------------------------------------

def is_completed(file):
    entry = database.get(str(file))
    return (
        entry is not None
        and entry["status"] == "completed"
    )

def get_status(file):
    entry = database.get(str(file))
    if entry is None:
        return None
    return entry["status"]

def load_database():
    if DATABASE_FILE.exists():
        with open(
            DATABASE_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)
    return {}

def save_database():
    with open(
        DATABASE_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            database,
            f,
            indent=2
        )

def update_status(file_path, status):
    with queue_lock:
        database[str(file_path)] = {
            "status": status,
            "updated": datetime.now().isoformat()
        }
        save_database()


database = load_database()

# --------------------------------------------------
# RECORDING COMPLETE DETECTION
# --------------------------------------------------

def wait_for_recording_finished(
    path,
    idle_seconds=RECORDING_IDLE_SECONDS
):
    print(
        f"Waiting for recording to finish: "
        f"{path.name}"
    )

    while True:
        try:
            age = (
                time.time()
                - path.stat().st_mtime
            )

            if age >= idle_seconds:
                print(
                    f"Recording complete: "
                    f"{path.name}"
                )
                return

        except FileNotFoundError:
            pass

        time.sleep(2)

# --------------------------------------------------
# QUEUE MANAGEMENT
# --------------------------------------------------

def queue_file(file):
    file = Path(file)

    with queue_lock:
        if (
            is_completed(file)
            or str(file) in queued_files
            or str(file) in active_files
        ):
            print(
                f"Already queued/processed: "
                f"{file.name}"
            )
            return

        queued_files.add(str(file))

    print(f"Queued: {file.name}")
    WORK_QUEUE.put(file)

# --------------------------------------------------
# TRANSCRIPTION
# --------------------------------------------------

def transcribe_file(audio_file):
    audio_file = Path(audio_file)

    if is_completed(audio_file):
        print(
            f"Skipping already processed: "
            f"{audio_file.name}"
        )
        return

    print(f"\nProcessing: {audio_file.name}")

    try:
        if not audio_file.exists():
            raise FileNotFoundError(
                audio_file
            )

        if audio_file.stat().st_size < 1024:
            raise RuntimeError(
                "Audio file appears incomplete."
            )

        # ------------------------------------------
        # LOAD AUDIO
        # ------------------------------------------

        print("Loading audio...")

        audio, sample_rate = sf.read(
            audio_file
        )

        if len(audio) == 0:
            raise RuntimeError(
                "Audio file contains no samples."
            )

        waveform = torch.tensor(
            audio,
            dtype=torch.float32
        )

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.T

        if ASR_BACKEND == "whisperx":
            print(
                "Running WhisperX transcription + diarization..."
            )
            diarized_segments, info = transcribe_with_whisperx(
                audio_file
            )
        else:
            # ------------------------------------------
            # DIARIZATION
            # ------------------------------------------

            print(
                "Running speaker diarization..."
            )

            diarization = pipeline(
                {
                    "waveform": waveform,
                    "sample_rate": sample_rate
                }
            )

            speaker_regions = []

            annotation = diarization.speaker_diarization

            for turn, _, speaker in annotation.itertracks(
                yield_label=True
            ):
                speaker_regions.append(
                    {
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker": speaker
                    }
                )

            print(
                f"Found "
                f"{len(speaker_regions)} "
                f"speaker segments"
            )

            # ------------------------------------------
            # SPEAKER LOOKUP
            # ------------------------------------------

            def find_speaker(
                start_time,
                end_time
            ):
                best_speaker = "UNKNOWN"
                best_overlap = 0

                for region in speaker_regions:
                    overlap = max(
                        0,
                        min(
                            end_time,
                            region["end"]
                        )
                        - max(
                            start_time,
                            region["start"]
                        )
                    )

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = region["speaker"]

                return best_speaker

            # ------------------------------------------
            # ASR
            # ------------------------------------------

            print(
                "Running transcription..."
            )

            if ASR_BACKEND == "whisper":
                segments, info = transcribe_with_whisper(
                    audio_file
                )
            else:
                segments, info = transcribe_with_parakeet(
                    audio_file
                )

            diarized_segments = []
            for segment in segments:
                speaker_id = find_speaker(
                    segment.start,
                    segment.end
                )

                diarized_segments.append(
                    {
                        "speaker_id": speaker_id,
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": segment.text.strip()
                    }
                )

        stable_speaker_ids = sorted(
            set(seg["speaker_id"] for seg in diarized_segments)
        )

        speaker_map = load_speaker_map()
        speaker_history = load_speaker_history()
        attendees = load_attendees()

        suggestions = suggest_speaker_names(
            stable_speaker_ids,
            speaker_map,
            speaker_history,
            attendees
        )

        draft_file = save_draft_speaker_artifacts(
            audio_file,
            diarized_segments,
            speaker_map,
            suggestions
        )

        output_file = audio_file.with_name(
            f"{audio_file.stem}_transcript.md"
        )

        print(
            f"Saving: {output_file.name}"
        )

        render_markdown_from_draft(
            str(audio_file),
            diarized_segments,
            output_file,
            info.language,
            speaker_map
        )

        refresh_speaker_history(
            speaker_history,
            speaker_map,
            stable_speaker_ids
        )

        print(
            f"Draft speaker file: {draft_file.name}"
        )

        update_status(
            audio_file,
            "completed"
        )

        print(
            f"Completed: "
            f"{output_file.name}"
        )

    except Exception as e:
        print(
            f"ERROR processing "
            f"{audio_file.name}"
        )
        print(e)
        raise

# --------------------------------------------------
# WORKER
# --------------------------------------------------

def transcription_worker():
    while True:
        audio_file = WORK_QUEUE.get()

        try:
            with queue_lock:
                queued_files.discard(
                    str(audio_file)
                )

                active_files.add(
                    str(audio_file)
                )

            update_status(
                audio_file,
                "processing"
            )

            success = False

            for attempt in range(
                1,
                MAX_RETRIES + 1
            ):
                try:
                    print(
                        f"Attempt "
                        f"{attempt}/"
                        f"{MAX_RETRIES}: "
                        f"{audio_file.name}"
                    )

                    wait_for_recording_finished(
                        audio_file
                    )

                    transcribe_file(
                        audio_file
                    )

                    success = True
                    break

                except Exception as e:
                    print(
                        f"Attempt "
                        f"{attempt} failed:"
                    )
                    print(e)

                    if attempt < MAX_RETRIES:
                        print(
                            f"Retrying in "
                            f"{RETRY_DELAY} seconds..."
                        )

                        time.sleep(
                            RETRY_DELAY
                        )

            if not success:
                update_status(
                    audio_file,
                    "failed"
                )

                print(
                    f"FAILED after "
                    f"{MAX_RETRIES} attempts: "
                    f"{audio_file.name}"
                )

        finally:
            with queue_lock:
                active_files.discard(
                    str(audio_file)
                )

            WORK_QUEUE.task_done()

# --------------------------------------------------
# WATCHDOG
# --------------------------------------------------

class VoiceMeeterHandler(
    FileSystemEventHandler
):
    def on_created(
        self,
        event
    ):
        if event.is_directory:
            return

        file = Path(event.src_path)

        if file.suffix.lower() != ".wav":
            return

        print(
            f"Detected recording: "
            f"{file.name}"
        )

        queue_file(file)

# --------------------------------------------------
# START WORKER
# --------------------------------------------------

if os.getenv("RE_RENDER_ONLY", "0").strip() == "1":
    draft_source = os.getenv("RE_RENDER_SOURCE", "").strip()
    if not draft_source:
        raise RuntimeError(
            "RE_RENDER_ONLY=1 requires RE_RENDER_SOURCE=<path to *_transcript_draft.json>"
        )

    re_render_from_draft_file(draft_source)
    raise SystemExit(0)

worker = Thread(
    target=transcription_worker,
    daemon=True
)

worker.start()

# --------------------------------------------------
# STARTUP SCAN
# --------------------------------------------------

print(
    f"Scanning existing WAV files in:\n"
    f"{WATCH_FOLDER}"
)

for wav_file in WATCH_FOLDER.glob("*.wav"):
    status = get_status(wav_file)

    if status == "completed":
        continue

    if status == "processing":
        print(
            f"Recovering interrupted job: "
            f"{wav_file.name}"
        )

    elif status == "failed":
        print(
            f"Retrying failed job: "
            f"{wav_file.name}"
        )

    queue_file(wav_file)

# --------------------------------------------------
# START WATCHER
# --------------------------------------------------

observer = Observer()

observer.schedule(
    VoiceMeeterHandler(),
    str(WATCH_FOLDER),
    recursive=False
)

observer.start()

print(
    f"\nWatching:\n{WATCH_FOLDER}"
)

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping...")
    observer.stop()

observer.join()
