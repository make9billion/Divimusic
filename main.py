from pathlib import Path
from uuid import uuid4
import base64
import json
import shutil
import subprocess
import time
import zipfile
from io import BytesIO

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from mutagen import File as MutagenFile
except ImportError:
    MutagenFile = None

try:
    import librosa
    import numpy as np
except ImportError:
    librosa = None
    np = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'separated'
JOBS_FILE = BASE_DIR / 'jobs.json'
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
OUTPUT_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
JOB_TTL_SECONDS = 60 * 60 * 24 * 7
WHISPER_MODEL_SIZE = "small"
whisper_model = None
SEPARATION_MODES = {
    "vocals": {
        "label": "보컬 분리",
        "model": "htdemucs",
        "extra_args": ["--two-stems", "vocals"],
        "output_root": "htdemucs"
    },
    "4stems": {
        "label": "4트랙 분리",
        "model": "htdemucs",
        "extra_args": [],
        "output_root": "htdemucs"
    },
    "6stems": {
        "label": "6트랙 분리",
        "model": "htdemucs_6s",
        "extra_args": [],
        "output_root": "htdemucs_6s"
    }
}

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path='')
# 모든 도메인에서 접속 가능하도록 설정 (HTML 파일과 서버 주소가 달라도 통신 가능)
CORS(app)

# 파일 저장 및 결과 경로 설정
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


def is_allowed_audio(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def build_safe_upload_name(filename):
    safe_name = secure_filename(filename)
    suffix = Path(safe_name or filename).suffix.lower()
    return f"{uuid4().hex}{suffix}"


def collect_output_files(track_folder):
    output_files = []
    if not track_folder.exists():
        return output_files

    for path in sorted(track_folder.rglob('*')):
        if path.is_file() and path.suffix.lower() in OUTPUT_AUDIO_EXTENSIONS:
            relative_path = path.relative_to(OUTPUT_FOLDER).as_posix()
            output_files.append({
                "name": path.name,
                "url": f"/download/{relative_path}",
                "streamUrl": f"/media/{relative_path}",
                "size": path.stat().st_size
            })
    return output_files


def load_jobs():
    if not JOBS_FILE.exists():
        return {}
    try:
        return json.loads(JOBS_FILE.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def save_jobs(jobs):
    JOBS_FILE.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding='utf-8')


def save_job(job):
    jobs = load_jobs()
    jobs[job["id"]] = job
    save_jobs(jobs)


def cleanup_old_jobs():
    now = time.time()
    jobs = load_jobs()
    changed = False

    for upload_path in UPLOAD_FOLDER.glob('*'):
        if upload_path.is_file() and now - upload_path.stat().st_mtime > 60 * 60:
            upload_path.unlink(missing_ok=True)

    for job_id, job in list(jobs.items()):
        if now - job.get("createdAt", now) <= JOB_TTL_SECONDS:
            continue

        relative_folder = job.get("relativeFolder")
        if relative_folder:
            shutil.rmtree(OUTPUT_FOLDER / relative_folder, ignore_errors=True)
        jobs.pop(job_id, None)
        changed = True

    if changed:
        save_jobs(jobs)


def extract_album_cover(file_path):
    if MutagenFile is None:
        return None

    try:
        audio = MutagenFile(file_path)
        if not audio:
            return None

        pictures = []
        if hasattr(audio, "pictures"):
            pictures.extend(audio.pictures)

        tags = audio.tags or {}
        for key, value in tags.items():
            if key.startswith("APIC"):
                pictures.append(value)
            elif key == "covr":
                image_data = value[0]
                mime = "image/png" if image_data.imageformat == 14 else "image/jpeg"
                encoded = base64.b64encode(bytes(image_data)).decode("ascii")
                return f"data:{mime};base64,{encoded}"

        if not pictures:
            return None

        picture = pictures[0]
        mime = getattr(picture, "mime", None) or "image/jpeg"
        encoded = base64.b64encode(picture.data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception as e:
        print(f"앨범 커버 추출 실패: {e}")
        return None


def analyze_audio_metadata(file_path):
    default_analysis = {
        "bpm": 163,
        "key": "Am",
        "keyRoot": "A",
        "keyMode": "minor",
        "keyIndex": 9,
        "confidence": 0
    }
    if librosa is None or np is None:
        return default_analysis

    try:
        y, sr = librosa.load(str(file_path), sr=22050, mono=True, duration=120)
        if y.size == 0:
            return default_analysis

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(np.asarray(tempo).reshape(-1)[0])))
        if bpm < 60:
            bpm *= 2
        elif bpm > 220:
            bpm = int(round(bpm / 2))

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.maximum(chroma.mean(axis=1), 1e-9)
        chroma_mean = chroma_mean / chroma_mean.sum()

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        scores = []
        for index in range(12):
            major_score = float(np.corrcoef(chroma_mean, np.roll(major_profile, index))[0, 1])
            minor_score = float(np.corrcoef(chroma_mean, np.roll(minor_profile, index))[0, 1])
            scores.append((major_score, index, "major"))
            scores.append((minor_score, index, "minor"))

        confidence, key_index, key_mode = max(scores, key=lambda item: item[0])
        key_root = key_names[key_index]
        key_label = f"{key_root}{'m' if key_mode == 'minor' else ''}"

        return {
            "bpm": bpm or default_analysis["bpm"],
            "key": key_label,
            "keyRoot": key_root,
            "keyMode": key_mode,
            "keyIndex": key_index,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        print(f"BPM/Key 분석 실패: {e}")
        return default_analysis


def build_job_payload(job_id, original_name, track_folder, files, cover_data_url, mode, analysis=None):
    relative_folder = track_folder.relative_to(OUTPUT_FOLDER).as_posix()
    return {
        "id": job_id,
        "status": "success",
        "message": "분리 완료",
        "folder": job_id,
        "relativeFolder": relative_folder,
        "originalName": original_name,
        "mode": mode,
        "modeLabel": SEPARATION_MODES.get(mode, SEPARATION_MODES["4stems"])["label"],
        "createdAt": time.time(),
        "files": files,
        "cover": cover_data_url,
        "analysis": analysis or {},
        "lyrics": None,
        "zipUrl": f"/jobs/{job_id}/zip"
    }


def get_whisper_model():
    global whisper_model
    if WhisperModel is None:
        raise RuntimeError("faster-whisper가 설치되어 있지 않습니다. pip install faster-whisper를 실행하세요.")
    if whisper_model is None:
        # Demucs는 CUDA를 쓰지만, faster-whisper는 별도 CUDA 런타임(libcublas 등)이
        # 필요할 수 있어 개발 초기에는 CPU 모드로 안정성을 우선합니다.
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return whisper_model


def find_vocal_file(job):
    files = job.get("files", [])
    candidates = sorted(files, key=lambda item: 0 if Path(item.get("name", "")).stem.lower() == "vocals" else 1)
    for file_info in candidates:
        stream_url = file_info.get("streamUrl", "")
        relative_path = stream_url.replace("/media/", "", 1)
        source_path = OUTPUT_FOLDER / relative_path
        if source_path.exists() and source_path.is_file():
            return source_path
    return None


def transcribe_lyrics(job):
    audio_path = find_vocal_file(job)
    if audio_path is None:
        raise RuntimeError("가사를 추출할 보컬 트랙을 찾을 수 없습니다.")

    print(f"가사 추출 대상 파일: {audio_path}")
    model = get_whisper_model()
    try:
        segments, info = model.transcribe(
            str(audio_path),
            language="ko",
            vad_filter=True,
            beam_size=5,
            word_timestamps=True
        )
    except Exception as e:
        global whisper_model
        print(f"Whisper CUDA 전사 실패, CPU로 재시도합니다: {e}")
        if WhisperModel is None:
            raise
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language="ko",
            vad_filter=True,
            beam_size=5,
            word_timestamps=True
        )
    segment_payloads = []
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        word_payloads = []
        for word in getattr(segment, "words", None) or []:
            word_text = getattr(word, "word", "").strip()
            if not word_text:
                continue
            word_payloads.append({
                "start": getattr(word, "start", segment.start),
                "end": getattr(word, "end", segment.end),
                "text": word_text
            })
        payload = {
            "start": segment.start,
            "end": segment.end,
            "text": text
        }
        if word_payloads:
            payload["words"] = word_payloads
        segment_payloads.append(payload)
        lines.append(text)

    return {
        "text": "\n".join(lines),
        "segments": segment_payloads,
        "language": info.language,
        "duration": info.duration
    }


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/jobs')
def list_jobs():
    cleanup_old_jobs()
    jobs = sorted(load_jobs().values(), key=lambda item: item.get("createdAt", 0), reverse=True)
    return jsonify({"jobs": jobs})


@app.route('/jobs/<job_id>')
def get_job(job_id):
    cleanup_old_jobs()
    job = load_jobs().get(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404
    return jsonify(job)


@app.route('/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    jobs = load_jobs()
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404

    relative_folder = job.get("relativeFolder")
    if relative_folder:
        shutil.rmtree(OUTPUT_FOLDER / relative_folder, ignore_errors=True)

    jobs.pop(job_id, None)
    save_jobs(jobs)
    return jsonify({"status": "success"})


@app.route('/jobs/<job_id>/zip')
def download_job_zip(job_id):
    job = load_jobs().get(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404

    files = job.get("files", [])
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_info in files:
            stream_url = file_info.get("streamUrl", "")
            relative_path = stream_url.replace("/media/", "", 1)
            source_path = OUTPUT_FOLDER / relative_path
            if source_path.exists() and source_path.is_file():
                archive.write(source_path, arcname=source_path.name)

    memory_file.seek(0)
    zip_name = f"{Path(job.get('originalName', job_id)).stem}_divimusic.zip"
    return send_file(memory_file, as_attachment=True, download_name=zip_name, mimetype="application/zip")


@app.route('/jobs/<job_id>/lyrics', methods=['POST'])
def extract_job_lyrics(job_id):
    jobs = load_jobs()
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404

    cached_lyrics = job.get("lyrics")
    has_word_timestamps = any(segment.get("words") for segment in cached_lyrics.get("segments", [])) if cached_lyrics else False
    if cached_lyrics and has_word_timestamps:
        return jsonify(cached_lyrics)

    try:
        lyrics = transcribe_lyrics(job)
        job["lyrics"] = lyrics
        jobs[job_id] = job
        save_jobs(jobs)
        return jsonify(lyrics)
    except Exception as e:
        print(f"가사 추출 실패: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    cleanup_old_jobs()

    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    if not is_allowed_audio(file.filename):
        return jsonify({"error": "지원하지 않는 오디오 형식입니다."}), 400

    mode = request.form.get("mode", "4stems")
    mode_config = SEPARATION_MODES.get(mode, SEPARATION_MODES["4stems"])

    # 1. 파일 저장
    safe_filename = build_safe_upload_name(file.filename)
    file_path = UPLOAD_FOLDER / safe_filename
    file.save(file_path)
    cover_data_url = extract_album_cover(file_path)
    audio_analysis = analyze_audio_metadata(file_path)
    print(f"파일 업로드 완료: {file_path}")

    # 2. AI 분리 작업 실행 (Demucs 예시)
    # RTX GPU가 있는 개발 환경에서는 CUDA를 명시해 GPU로 처리합니다.
    try:
        print(f"AI 분리 작업 시작 ({mode_config['label']}, CUDA GPU 활용)...")
        # -n 옵션은 모델 이름을 뜻하며, htdemucs를 주로 사용합니다.
        command = [
            "demucs", 
            "-n", mode_config["model"],
            "--device", "cuda",
            "--mp3",
            "--mp3-bitrate", "320",
            *mode_config["extra_args"],
            "-o", str(OUTPUT_FOLDER), 
            str(file_path)
        ]
        subprocess.run(command, check=True)

        job_id = Path(safe_filename).stem
        track_folder = OUTPUT_FOLDER / mode_config["output_root"] / job_id
        files = collect_output_files(track_folder)
        job = build_job_payload(job_id, file.filename, track_folder, files, cover_data_url, mode, audio_analysis)
        save_job(job)
        file_path.unlink(missing_ok=True)

        return jsonify(job)

    except FileNotFoundError:
        return jsonify({
            "status": "error",
            "message": "demucs 명령을 찾을 수 없습니다. 먼저 Demucs를 설치하거나 PATH를 확인하세요."
        }), 500
    except subprocess.CalledProcessError as e:
        print(f"Demucs 실행 실패: {e}")
        return jsonify({"status": "error", "message": "오디오 분리 작업에 실패했습니다."}), 500
    except Exception as e:
        print(f"에러 발생: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# 결과 파일을 브라우저에서 확인할 수 있게 해주는 경로
@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route('/media/<path:filename>')
def media_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    # 요청하신 8000번 포트로 서버 실행
    # 0.0.0.0으로 설정해야 외부(192.168.0.38)에서 접속이 가능합니다.
    app.run(host='0.0.0.0', port=8000, debug=True)
