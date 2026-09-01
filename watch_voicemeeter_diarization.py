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

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

WATCH_FOLDER = Path(
    r"C:\Users\Frank\Documents\Voicemeeter"
)

DATABASE_FILE = Path(
    r"C:\AI\voicemeeter_processed.json"
)

RECORDING_IDLE_SECONDS = 30

MAX_RETRIES = 3
RETRY_DELAY = 30

# ASR backend: "whisper" or "parakeet"
ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper").strip().lower()
if ASR_BACKEND not in {"whisper", "parakeet"}:
    raise RuntimeError(
        "ASR_BACKEND must be 'whisper' or 'parakeet'"
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
# LOAD MODELS
# --------------------------------------------------

whisper = None
parakeet = None

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

        # ------------------------------------------
        # OUTPUT
        # ------------------------------------------

        output_file = audio_file.with_name(
            f"{audio_file.stem}_transcript.md"
        )

        print(
            f"Saving: {output_file.name}"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("---\n")
            f.write(
                "type: meeting-transcript\n"
            )
            f.write(
                "source: voicemeeter\n"
            )
            f.write("---\n\n")

            f.write(
                f"# {audio_file.stem}\n\n"
            )

            current_speaker = None
            current_start = None
            speaker_text = []

            for segment in segments:
                speaker = find_speaker(
                    segment.start,
                    segment.end
                )

                if current_speaker is None:
                    current_speaker = speaker
                    current_start = int(
                        segment.start
                    )

                elif speaker != current_speaker:
                    minutes = (
                        current_start // 60
                    )
                    seconds = (
                        current_start % 60
                    )

                    f.write(
                        f"## {current_speaker}\n\n"
                    )

                    f.write(
                        f"[{minutes:02}:{seconds:02}] "
                        + " ".join(
                            speaker_text
                        )
                        + "\n\n"
                    )

                    current_speaker = speaker
                    current_start = int(
                        segment.start
                    )

                    speaker_text = []

                speaker_text.append(
                    segment.text.strip()
                )

            if speaker_text:
                minutes = (
                    current_start // 60
                )
                seconds = (
                    current_start % 60
                )

                f.write(
                    f"## {current_speaker}\n\n"
                )

                f.write(
                    f"[{minutes:02}:{seconds:02}] "
                    + " ".join(
                        speaker_text
                    )
                    + "\n\n"
                )

            speakers = sorted(
                set(
                    region["speaker"]
                    for region
                    in speaker_regions
                )
            )

            duration_seconds = 0

            if segments:
                duration_seconds = int(
                    segments[-1].end
                )

            f.write("---\n\n")
            f.write(
                "## Transcript Statistics\n\n"
            )

            f.write(
                f"- Duration: "
                f"{duration_seconds // 60}m "
                f"{duration_seconds % 60}s\n"
            )

            f.write(
                f"- Language: "
                f"{info.language}\n"
            )

            f.write(
                f"- Speakers: "
                f"{len(speakers)}\n\n"
            )

            f.write(
                "## Speakers\n\n"
            )

            for speaker in speakers:
                f.write(
                    f"- {speaker}\n"
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
