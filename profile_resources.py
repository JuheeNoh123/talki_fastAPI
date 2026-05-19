"""
리소스 피크 측정 스크립트 (독립 실행 버전)
- 앱 코드 import 없이 모델을 직접 로드
- 백그라운드 스레드가 0.5초마다 RAM/VRAM 샘플링 → 피크 추적
- MediaPipe는 CPU 전용이라 VRAM 측정에 영향 없음 (별도 RAM 측정)

실행:
    python profile_resources.py <영상.mp4>
    python profile_resources.py  (영상 없이 모델 로딩만 측정)
"""

import os, sys, time, threading
import psutil, torch
from pathlib import Path

import subprocess

def _gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        return int(out.decode().strip())
    except Exception:
        return None

HAS_NVML = _gpu_util() is not None

BASE_DIR   = Path(__file__).resolve().parent
TOPIC_PATH = BASE_DIR / "Topic_model" / "topic_model_mnr"
LABEL_PATH = BASE_DIR / "Topic_model" / "label_model"
AUDIO_PATH = BASE_DIR / "temp_audio.wav"

PROCESS   = psutil.Process(os.getpid())
HAS_CUDA  = torch.cuda.is_available()
DEVICE    = "cuda" if HAS_CUDA else "cpu"


# ── 모니터 ──────────────────────────────────────────────────────────────────────

class Monitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self._running = False
        self._t = None
        self.ram       = []   # MB
        self.vram      = []   # MB
        self.gpu_util  = []   # %
        self.cpu_util  = []   # %

    def _loop(self):
        while self._running:
            self.ram.append(PROCESS.memory_info().rss / 1024 / 1024)
            self.cpu_util.append(psutil.cpu_percent(interval=None))
            if HAS_CUDA:
                self.vram.append(torch.cuda.memory_allocated() / 1024 / 1024)
            if HAS_NVML:
                v = _gpu_util()
                if v is not None:
                    self.gpu_util.append(v)
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._running = False
        self._t and self._t.join()

    def snap(self, label):
        r = PROCESS.memory_info().rss / 1024 / 1024
        v = torch.cuda.memory_allocated() / 1024 / 1024 if HAS_CUDA else 0
        vstr = f"VRAM {v:.0f} MB" if HAS_CUDA else "VRAM N/A"
        print(f"    [{label}]  RAM {r:.0f} MB  |  {vstr}")
        return r, v


# ── 측정 단계 ────────────────────────────────────────────────────────────────────

def step(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"\n{'='*55}")
    if HAS_CUDA:
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1024 / 1024
        print(f"  GPU : {props.name}  ({total_vram:.0f} MB VRAM)")
    else:
        total_vram = 0
        print(f"  GPU : 없음 — CPU 모드")
    print(f"  RAM : {psutil.virtual_memory().total/1024/1024:.0f} MB")
    print(f"  측정 간격 : 0.5초")
    print(f"{'='*55}\n")

    mon = Monitor()
    mon.start()

    # ── 0. 기준선 ──────────────────────────────────────────────────────────────
    step("0  기준선")
    r0, v0 = mon.snap("baseline")

    # ── 1. Whisper small ───────────────────────────────────────────────────────
    step("1  Whisper small 로드")
    t0 = time.time()
    import whisper
    wmodel = whisper.load_model("small", device=DEVICE)
    r1, v1 = mon.snap(f"로드 완료 {time.time()-t0:.1f}s")
    print(f"    증가 → RAM +{r1-r0:.0f} MB  VRAM +{v1-v0:.0f} MB")

    # ── 1b. Whisper 실제 추론 (오디오 파일이 있으면) ────────────────────────────
    if AUDIO_PATH.exists():
        step("1b Whisper 추론 (temp_audio.wav)")
        t0 = time.time()
        result = wmodel.transcribe(str(AUDIO_PATH), word_timestamps=False)
        r1b, v1b = mon.snap(f"추론 완료 {time.time()-t0:.1f}s")
        print(f"    추론 중 VRAM 최대 : {max(mon.vram or [v1b]):.0f} MB")
    else:
        print("\n  ※ temp_audio.wav 없음 — Whisper 추론 생략")
        r1b, v1b = r1, v1

    # ── 2. sentence-transformers (topic_model) ─────────────────────────────────
    step("2  sentence-transformers (topic_model_mnr)")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    tmodel = SentenceTransformer(str(TOPIC_PATH), device=DEVICE)
    r2, v2 = mon.snap(f"로드 완료 {time.time()-t0:.1f}s")
    print(f"    증가 → RAM +{r2-r1b:.0f} MB  VRAM +{v2-v1b:.0f} MB")

    # 추론
    tmodel.encode(["안녕하세요 발표를 시작하겠습니다"] * 8)
    r2b, v2b = mon.snap("encode 후")

    # ── 3. RoBERTa 분류기 (label_model) ───────────────────────────────────────
    step("3  RoBERTa 분류기 (label_model)")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok  = AutoTokenizer.from_pretrained(str(LABEL_PATH))
    lmodel = AutoModelForSequenceClassification.from_pretrained(str(LABEL_PATH)).to(DEVICE)
    r3, v3 = mon.snap(f"로드 완료 {time.time()-t0:.1f}s")
    print(f"    증가 → RAM +{r3-r2b:.0f} MB  VRAM +{v3-v2b:.0f} MB")

    # 추론
    inp = tok("발표 내용 테스트 문장입니다.", return_tensors="pt", truncation=True, max_length=512)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        lmodel(**inp)
    r3b, v3b = mon.snap("추론 후")

    # ── 4. MediaPipe (RAM만, CPU 전용) ─────────────────────────────────────────
    step("4  MediaPipe FaceMesh + Pose  (CPU 전용 — VRAM 영향 없음)")
    t0 = time.time()
    try:
        import mediapipe as mp
        # 신버전 Tasks API 사용
        from mediapipe.tasks.python import vision as mpv
        from mediapipe.tasks import python as mpt
        # Tasks API는 .task 모델 파일이 필요해 여기선 임포트만 측정
        r4, v4 = mon.snap(f"import 완료 {time.time()-t0:.1f}s")
        print(f"    증가 → RAM +{r4-r3b:.0f} MB  (VRAM 변화 없음)")
    except Exception as e:
        r4, v4 = r3b, v3b
        print(f"    MediaPipe import 실패: {e}")
        print(f"    → RAM 영향 측정 생략 (VRAM에는 영향 없음)")

    mon.stop()

    # ── 결과 요약 ──────────────────────────────────────────────────────────────
    peak_ram  = max(mon.ram)      if mon.ram      else r4
    peak_vram = max(mon.vram)     if mon.vram     else v3b
    peak_gpu  = max(mon.gpu_util) if mon.gpu_util else 0
    peak_cpu  = max(mon.cpu_util) if mon.cpu_util else 0
    avg_gpu   = sum(mon.gpu_util) / len(mon.gpu_util) if mon.gpu_util else 0
    avg_cpu   = sum(mon.cpu_util) / len(mon.cpu_util) if mon.cpu_util else 0

    print(f"\n{'='*55}")
    print("  최종 피크 측정 결과")
    print(f"{'='*55}")
    print(f"  RAM  피크 : {peak_ram:.0f} MB  (기준 {r0:.0f} MB → +{peak_ram-r0:.0f} MB)")
    if HAS_CUDA:
        print(f"  VRAM 피크 : {peak_vram:.0f} MB  (기준 {v0:.0f} MB → +{peak_vram-v0:.0f} MB)")
        print(f"  VRAM 여유 : {total_vram - peak_vram:.0f} MB 남음 ({(total_vram-peak_vram)/total_vram*100:.1f}%)")
    print(f"\n  GPU 사용률 피크 : {peak_gpu:.0f}%  /  평균 : {avg_gpu:.0f}%")
    print(f"  CPU 사용률 피크 : {peak_cpu:.0f}%  /  평균 : {avg_cpu:.0f}%")

    # ── AWS 추천 ───────────────────────────────────────────────────────────────
    print(f"\n  [AWS 인스턴스 추천]")
    if HAS_CUDA:
        # 동시 분석 2개 기준 2배 여유
        req_vram = peak_vram * 1.3
        print(f"  동시 분석 2개 기준 필요 VRAM : ~{req_vram:.0f} MB")
        if req_vram < 8000:
            rec = "g4dn.xlarge  (T4 16GB VRAM)  ← 충분한 여유"
        elif req_vram < 15000:
            rec = "g4dn.xlarge  (T4 16GB VRAM)  ← 빠듯 / g4dn.2xlarge 고려"
        else:
            rec = "g5.xlarge    (A10G 24GB VRAM) ← 필요"
        print(f"  GPU 추천 : {rec}")

    req_ram = peak_ram * 1.3
    print(f"  동시 분석 2개 기준 필요 RAM  : ~{req_ram:.0f} MB")
    if req_ram < 7000:
        print(f"  RAM 추천 : 8 GB 이상이면 충분")
    elif req_ram < 14000:
        print(f"  RAM 추천 : 16 GB 필요")
    else:
        print(f"  RAM 추천 : 32 GB 필요")

    # ── 타임라인 ───────────────────────────────────────────────────────────────
    bucket = max(1, int(5.0 / mon.interval))

    if mon.vram:
        print(f"\n  VRAM 타임라인 (5초 단위 최대값):")
        for i in range(0, len(mon.vram), bucket):
            chunk = mon.vram[i:i+bucket]
            t_sec = i * mon.interval
            bar = "█" * int(max(chunk) / 100)
            print(f"    {t_sec:5.0f}s  {bar}  {max(chunk):.0f} MB")

    if mon.gpu_util:
        print(f"\n  GPU 사용률 타임라인 (5초 단위 최대값):")
        for i in range(0, len(mon.gpu_util), bucket):
            chunk = mon.gpu_util[i:i+bucket]
            t_sec = i * mon.interval
            bar = "█" * int(max(chunk) / 5)
            print(f"    {t_sec:5.0f}s  {bar}  {max(chunk):.0f}%")

    if mon.cpu_util:
        print(f"\n  CPU 사용률 타임라인 (5초 단위 최대값):")
        for i in range(0, len(mon.cpu_util), bucket):
            chunk = mon.cpu_util[i:i+bucket]
            t_sec = i * mon.interval
            bar = "█" * int(max(chunk) / 5)
            print(f"    {t_sec:5.0f}s  {bar}  {max(chunk):.0f}%")

    print()


if __name__ == "__main__":
    main()
