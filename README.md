# Transcription pipeline

Audio in → cleaned, timestamped transcript out → JSON / SRT / TXT plus a few
rule-based insights (stats, keywords, action items).

```
 audio file(s)
      │
      ▼
┌───────────┐   ┌────────────┐   ┌─────────────┐   ┌────────────┐
│ 1 prepare │ → │ 2 transcribe│ → │ 3 postprocess│ → │ 4 export   │
│ validate  │   │ STTBackend  │   │ clean, redact│   │ JSON SRT TXT│
│ → 16k mono│   │ (pluggable) │   │ flag low conf│   │ + insights  │
│ hash, RMS │   └────────────┘   └─────────────┘   └────────────┘
└───────────┘
   silent? → skip model, emit warning
```

## Run it

```bash
# no model, no network: mock backend + synthetic audio
python -m tpipe.samples samples/
python -m tpipe.cli samples/ -o out/ --backend mock

# real transcription (open-source Whisper, runs locally)
pip install faster-whisper
python -m tpipe.cli meeting.mp3 -o out/ --backend whisper --model small --redact-pii

# tests (standard library only)
python -m unittest discover -s tests -t .     # 33 tests
```

Non-16 kHz-mono-WAV inputs (mp3, m4a, stereo, ...) need the `ffmpeg` binary.
Output layout:

```
out/
  run_summary.json            # per-file status, warnings, stage timings
  meeting-3fa9c1d2/           # <name>-<first 8 hex of audio hash>
    transcript.json  transcript.srt  transcript.txt
```

## Assessment answers (Part 1)

**1-3. Accept audio, transcribe, return per-segment timestamps.**
`python -m tpipe.cli talk.mp3 -o out/ --backend whisper` takes WAV/MP3 (and a few
other formats), transcribes with faster-whisper, and writes `transcript.json`
(each segment has `start`, `end`, `text`, `confidence`) plus `transcript.srt`.
`TranscriptionPipeline.run_file(path)` is the same thing as a library call, so
wrapping it in an HTTP endpoint is a thin layer.

**4. Different audio formats.** Everything is normalized once, at the door, to
16 kHz mono 16-bit PCM WAV, the format speech models expect, so no later stage
cares what was uploaded.
- Extension allow-list first (wav, mp3, m4a, flac, ogg, opus, webm, mp4) for a fast, clear rejection of non-audio.
- Already-correct WAVs skip conversion entirely (no ffmpeg needed).
- Everything else goes through `ffmpeg`, which sniffs the real container rather than trusting the extension, downmixes stereo, and resamples. A hard timeout stops a pathological file hanging the worker.
- Validation is by actually decoding: a file named `.wav` that is garbage fails with a readable reason, not a crash. Empty, too-short and too-long files are rejected the same way.
- The *original* file is hashed, so the same recording uploaded as WAV and as MP3 is treated as two inputs, which is the honest behaviour.
- Trade-off: a valid file with a missing or unusual extension is rejected. Probing with `ffprobe` instead would accept it; I chose the explicit allow-list for predictability.

**5. Long audio.** Three layers:
1. *Bounded memory everywhere:* hashing, loudness analysis and chunk writing all stream; faster-whisper yields segments lazily instead of returning one giant list.
2. *Silence-aware chunking:* files longer than 1.5x `--chunk-seconds` (default 600) are split into ~10 minute pieces. Each cut is placed at the quietest 100 ms window within +/-15 s of the target, so it lands in a pause rather than mid-word; that removes the need for overlapping windows and de-duplicating the seam. Chunks are written one at a time and deleted straight after use.
3. *Timestamps stay in file time:* each chunk's segments are shifted by the chunk's start offset, so the output reads as one continuous transcript. The language is auto-detected on the first chunk and reused for the rest (faster, and avoids the detector flip-flopping on a quiet chunk).

Measured with the mock backend (so this is pipeline overhead, not model speed):
a 1-hour stereo 44.1 kHz MP3 produced 6 chunks and correct timestamps to
3599.95 s, in 7.5 s with 37 MB peak memory; nearly all of that time was ffmpeg
decoding the MP3. Real model time will dominate in practice.

## Engineering decisions

**Open-source, local STT (faster-whisper) behind a tiny interface.**
Local means no per-minute cost and audio never leaves the machine, which
matters for meetings and calls. The pipeline depends only on `STTBackend`
(`transcribe(wav, language) -> segments`), so a hosted API or another engine is
one new class. A deterministic `MockBackend` lets the entire pipeline be
developed and tested with no model download, GPU or network.

**Normalize once, up front.** Everything is converted to 16 kHz mono PCM16 WAV
so each backend sees identical input. Files already in that format skip ffmpeg.

**Don't feed silence to the model.** Whisper-style models hallucinate text on
silence. A cheap RMS check skips near-silent files and records `silent_audio`.
For real audio the backend also uses VAD, drops windows Whisper itself scores as
probably-not-speech, and disables `condition_on_previous_text` so one bad window
can't corrupt the rest.

**Clean text with small, explainable heuristics, and keep the evidence.**
Filler removal, stutter collapsing, casing, optional PII redaction (emails,
phone numbers). Each is a pure function and individually toggleable. The
original wording is kept in `raw_text`, *except* when redaction is on, because
keeping it would defeat the point (tested).

**Confidence is surfaced, not hidden.** Segments below a threshold get a
`low_confidence` flag and are counted in the insights, so downstream consumers
can route them to human review instead of trusting everything equally.

**One canonical, versioned output.** `Transcript` (with `schema_version`) is the
single source of truth; SRT/TXT/insights are views over it.

**Reliability.**
- *Failure isolation:* one bad file produces a `failed` result with a reason; the batch continues; CLI exits non-zero.
- *Idempotency:* a fingerprint of audio hash + backend + text-affecting settings lives in `transcript.json`. Re-running unchanged input is a no-op and never calls the model; changing a setting reprocesses.
- *Atomic writes:* temp file + `os.replace`, with JSON written last, so a crash can't leave something that looks complete but isn't.
- *Observability:* per-stage timings and a `run_summary.json`.

## From script to service (Part 2 design notes, not implemented)

`TranscriptionPipeline.run_file()` is already the unit of work: idempotent,
failure-isolated, and resumable at chunk level (once checkpoints are stored).
Turning it into a service is mostly about what surrounds it.

```
client -> API -> object storage (audio)
           |            ^
           v            |
       Postgres     workers (stateless, GPU) <- queue -> dead-letter queue
       (job state)      |
           ^            v
           '---- transcripts (JSON in object storage)
```

**Concurrent uploads.** Never transcribe inside the request. The API stores the
upload, creates a job and returns `202` with a job id; a queue absorbs bursts
and stateless workers scale out on queue depth. Large files go straight to
object storage through presigned, resumable multipart URLs so API servers never
carry the bytes. Protect the system with per-tenant rate limits, concurrency
quotas, size limits and fair scheduling (one customer can't starve the rest).
Duplicate submissions are cheap to detect: an `Idempotency-Key` header plus the
audio hash and settings fingerprint the pipeline already computes. Workers claim
jobs with a lease (compare-and-set on job state) so two workers never run the same
job, and each worker loads the model once and runs a bounded number of jobs in
parallel.

**Storage.** *Audio:* object storage (S3/GCS), immutable and keyed by content
hash (free dedupe), encrypted, with lifecycle rules (retention period, cold
storage, delete on request) because recordings are sensitive. The normalized WAV
is derived data: regenerate it rather than store it. *Transcripts:* the
canonical JSON goes to object storage (versioned by `schema_version`); SRT/TXT
are rendered from it on request, not stored three times. *Job metadata:*
Postgres (id, tenant, status, audio hash, settings fingerprint, attempts, error,
timestamps, pointers to storage) because we need queries, listing and
status. If full-text search is needed, add Postgres `tsvector` or OpenSearch.
Per-chunk partial results are stored under the job so retries can resume.

**Retry and recovery.** Job states: `uploaded -> queued -> processing ->
completed | failed`, with `retrying` and `dead`, persisted in the DB. Failures
are classified: *permanent* (corrupt or unsupported audio, too long) fail
immediately with a clear message and are never retried; *transient* (worker crash,
out-of-memory, network, model timeout) retry with exponential backoff plus jitter
and a capped attempt count. The queue is at-least-once (visibility timeout plus
heartbeat), so a dead worker's job is redelivered; this is safe because processing
is idempotent (fingerprint, atomic writes, deterministic chunk plan), and finished
chunks are skipped on retry. After the cap, jobs go to a dead-letter queue (this
also stops a poison file from crashing workers forever), with an alert and a replay
tool. A reaper requeues jobs stuck in `processing` past their lease. Per-job
timeouts scale with audio duration. Metrics: queue depth, failure rate by reason,
latency, real-time factor.

**API.** REST with the async job pattern, versioned under `/v1`:

| Endpoint | Purpose |
|---|---|
| `POST /v1/uploads` | get presigned URL(s) for large files |
| `POST /v1/transcriptions` | start a job (`upload_id` or `audio_url`, `language`, options); returns `202` + `{id, status}`; accepts `Idempotency-Key` |
| `GET /v1/transcriptions/{id}` | status, progress (chunks done/total), error |
| `GET /v1/transcriptions/{id}/result?format=json\|srt\|txt` | the transcript |
| `GET /v1/transcriptions` | list, filter, paginate |
| `DELETE /v1/transcriptions/{id}` | delete audio and transcript |

Results can also be pushed to a `callback_url` (HMAC-signed, retried with
backoff). API keys or OAuth with per-key limits; consistent error bodies; `413`
too large, `415` unsupported type, `422` bad options, `429` rate limited. A thin
FastAPI layer over `run_file` would do; the API process never loads the model.

## Known limitations (deliberate scope cuts)

- **Regex heuristics are best-effort.** Phone matching is North-American style only; stutter collapsing would merge a legitimate "had had"; action-item detection is keyword-based and per-segment. Fine for a first pass, and the interface lets an LLM replace them.
- **Real-backend path is not exercised by the tests.** `FasterWhisperBackend` was written against the faster-whisper API but never run in the sandbox this was built in (no network, no model). It's a thin adapter; run the CLI with `--backend whisper` on one real file before trusting it.
- **Chunks are processed sequentially, with no per-chunk checkpoint.** A crash on a very long file restarts that file from the beginning. A cut can still fall inside a sentence if the speaker doesn't pause near the boundary; Whisper copes with that far better than with a mid-word cut, but it can still happen.
- **No speaker diarization, no word-level timestamps.**
- **Old outputs are not garbage-collected** when settings change (the folder is overwritten in place, so this only matters if the audio itself changes).

## What I'd do next

1. **Measure quality:** a small labeled set and word error rate (e.g. `jiwer`), tracked per model size and per cleaning step, so "better" is a number. Also check that cleaning never *worsens* WER versus raw.
2. **Scale out:** chunking already exists; next is fanning chunks out to parallel workers via a queue, and checkpointing each finished chunk so a failure at hour 2 resumes instead of restarting. The fingerprint scheme already gives per-file dedup.
3. **Streaming** for live audio (chunked decoding with partial results).
4. **Diarization** (who spoke) and word timestamps for richer captions.
5. **LLM post-step** behind the same `insights` interface for summaries and action items, with the rule-based version as fallback.
6. **Retries/backoff + timeouts** if a hosted backend is added.
