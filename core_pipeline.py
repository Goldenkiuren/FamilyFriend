import queue
import time
import threading
import numpy as np
import sounddevice as sd
import cv2
import mss
import easyocr
import av 
from faster_whisper import WhisperModel
from fractions import Fraction
from toxic_classifier import ToxicCensor
from audio_censor import AudioCensor
from video_censor import VideoCensor

# ==========================================
# CONFIGURAÇÕES GERAIS E FILAS
# ==========================================
SAMPLE_RATE = 16000
CHANNELS = 1
VIDEO_FPS = 30

CONTEXT_DURATION = 3.0
STRIDE_DURATION = 1.0

CONTEXT_SAMPLES = int(CONTEXT_DURATION * SAMPLE_RATE)
STRIDE_SAMPLES = int(STRIDE_DURATION * SAMPLE_RATE)

class MultimodalPipeline:
    def __init__(self):
        self.ui_update = None
        self.get_threshold = None
        self.is_realtime = None

        self.is_running = False
        self.total_censored_words = 0
        self.session_metrics = []

        self.raw_audio_queue = queue.Queue(maxsize=10)
        self.chunk_queue = queue.Queue(maxsize=5)
        self.output_audio_queue = queue.Queue(maxsize=10)
        self.display_queue = queue.Queue(maxsize=2)
        self.ocr_queue = queue.Queue(maxsize=1)
        
        self.current_bboxes = []
        self.bbox_lock = threading.Lock()
        
        self.mux_lock = threading.Lock()
        self.av_container = None
        self.v_stream = None
        self.a_stream = None
        
        # NOVO: Semáforo para segurar a gravação até as IAs carregarem
        self.models_ready = threading.Event()

    def start(self):
        if self.is_running: return
        self.is_running = True
        self.total_censored_words = 0
        self.session_metrics = []
        self.current_bboxes = []
        
        self.models_ready.clear() # Sinal vermelho

        for q in [self.raw_audio_queue, self.chunk_queue, self.output_audio_queue, self.display_queue, self.ocr_queue]:
            while not q.empty(): q.get()

        # O Muxer e streams NÃO são mais iniciados aqui, foram movidos para a IA

        threading.Thread(target=self._startup_ai_and_run, daemon=True).start()
        threading.Thread(target=self._chunking_worker, daemon=True).start()
        threading.Thread(target=self._audio_playback_worker, daemon=True).start()
        threading.Thread(target=self._screen_capture_worker, daemon=True).start()

    def stop(self):
        if not self.is_running: return
        self.is_running = False
        threading.Thread(target=self._export_session, daemon=True).start()

    def get_display_frame(self):
        try:
            return self.display_queue.get_nowait()
        except queue.Empty:
            return None

    # Função auxiliar para não travar o programa se o usuário clicar em "Parar" enquanto carrega
    def _wait_for_models(self):
        while self.is_running and not self.models_ready.is_set():
            time.sleep(0.1)
        return self.is_running

    # ==========================================
    # WORKERS DE ÁUDIO E RING BUFFER
    # ==========================================
    def _audio_producer_callback(self, indata, frames, time_info, status):
        # Só começa a ouvir o microfone depois que a IA liberar
        if self.is_running and self.models_ready.is_set() and not self.raw_audio_queue.full():
            self.raw_audio_queue.put(indata.copy())

    def _chunking_worker(self):
        if not self._wait_for_models(): return
        
        ring_buffer = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        new_data_buffer = np.zeros(0, dtype=np.float32)
        
        while self.is_running:
            try:
                data = self.raw_audio_queue.get(timeout=0.1)
                new_data_buffer = np.concatenate((new_data_buffer, data.flatten()))
                
                while len(new_data_buffer) >= STRIDE_SAMPLES:
                    stride_data = new_data_buffer[:STRIDE_SAMPLES]
                    new_data_buffer = new_data_buffer[STRIDE_SAMPLES:]
                    
                    ring_buffer = np.roll(ring_buffer, -STRIDE_SAMPLES)
                    ring_buffer[-STRIDE_SAMPLES:] = stride_data
                    
                    if not self.chunk_queue.full():
                        self.chunk_queue.put(ring_buffer.copy())
            except queue.Empty:
                continue

    def _audio_playback_worker(self):
        if not self._wait_for_models(): return
        
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        stream.start()
        while self.is_running:
            try:
                audio_data = self.output_audio_queue.get(timeout=0.1)
                if self.is_realtime and self.is_realtime():
                    stream.write(audio_data)
            except queue.Empty:
                continue
        stream.stop()
        stream.close()

    # ==========================================
    # WORKERS DE VÍDEO
    # ==========================================
    def _screen_capture_worker(self):
        if not self._wait_for_models(): return
        
        self.video_processor = VideoCensor(blur_kernel_size=81)
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            frame_counter = 0
            frame_duration = 1.0 / VIDEO_FPS
            
            while self.is_running:
                start_time = time.time()
                
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                with self.bbox_lock:
                    bboxes_to_draw = list(self.current_bboxes)

                frame = self.video_processor.process_frame(frame, bboxes_to_draw)

                if frame_counter % 30 == 0:
                    if not self.ocr_queue.full():
                        self.ocr_queue.put(cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR))

                if self.av_container is not None:
                    if self.v_stream is None:
                        height, width = frame.shape[:2]
                        self.v_stream = self.av_container.add_stream("libx264", rate=VIDEO_FPS)
                        self.v_stream.width = width
                        self.v_stream.height = height
                        self.v_stream.pix_fmt = "yuv420p"
                        self.v_stream.options = {"preset": "ultrafast"}

                    av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                    with self.mux_lock:
                        for packet in self.v_stream.encode(av_frame):
                            self.av_container.mux(packet)

                frame_resized = cv2.resize(frame, (640, 360))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                if not self.display_queue.full():
                    self.display_queue.put(frame_rgb)

                frame_counter += 1
                
                elapsed = time.time() - start_time
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _ocr_inference_worker(self, toxic_censor_video):
        if not self._wait_for_models(): return
        
        if self.ui_update: self.ui_update(msg="Carregando EasyOCR na GPU...")
        reader = easyocr.Reader(['en'], gpu=True)
        if self.ui_update: self.ui_update(msg="✅ EasyOCR Pronto!")

        while self.is_running:
            try:
                frame = self.ocr_queue.get(timeout=0.5)
                results = reader.readtext(frame, detail=1)
                words_for_ia = []

                for bbox, text, conf in results:
                    if conf > 0.4 and len(text.strip()) > 1:
                        words_for_ia.append({"word": text, "bbox": bbox, "start": 0, "end": 0})

                if words_for_ia:
                    toxic_results = toxic_censor_video.detect_toxic_words(words_for_ia)
                    if toxic_results:
                        new_bboxes = [item["bbox"] for item in toxic_results]
                        with self.bbox_lock: self.current_bboxes = new_bboxes
                        censored_str = ", ".join([f"{t['word']} [TELA]" for t in toxic_results])
                        if self.ui_update: self.ui_update(msg=f"👁️ BLOQUEADO (TELA): {censored_str}")
                    else:
                        with self.bbox_lock: self.current_bboxes = []
                else:
                    with self.bbox_lock: self.current_bboxes = []
            except queue.Empty:
                continue

    # ==========================================
    # CÉREBRO PRINCIPAL (ALINHAMENTO DE TEMPO)
    # ==========================================
    def _startup_ai_and_run(self):
        if self.ui_update: self.ui_update(msg="Carregando IA Multimodal (Whisper + RoBERTa)...")
        try:
            whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            current_threshold = self.get_threshold() if self.get_threshold else 0.5
            
            toxic_censor_audio = ToxicCensor(threshold=current_threshold)
            toxic_censor_video = ToxicCensor(threshold=current_threshold)
            audio_processor = AudioCensor(sample_rate=SAMPLE_RATE) 
            
            threading.Thread(target=self._ocr_inference_worker, args=(toxic_censor_video,), daemon=True).start()

            # CORREÇÃO: Agora sim, com os pesos pesados na GPU, abrimos a gravação!
            self.av_container = av.open("live_output_censurado.mp4", mode="w")
            self.a_stream = self.av_container.add_stream("aac", rate=SAMPLE_RATE)
            self.a_stream.layout = "mono" if CHANNELS == 1 else "stereo"
            self.v_stream = None
            self.audio_fifo = av.AudioFifo()

            # Libera as threads retidas para começarem a capturar vídeo e áudio juntos!
            self.models_ready.set()
            if self.ui_update: self.ui_update(msg="✅ Modelos prontos! Gravação INICIADA.")

            new_stride_start_time = CONTEXT_DURATION - STRIDE_DURATION 
            
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=self._audio_producer_callback):
                while self.is_running:
                    try:
                        audio_window = self.chunk_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    start_time = time.time()

                    segments, _ = whisper_model.transcribe(
                        audio_window, language="en", word_timestamps=True, beam_size=5, condition_on_previous_text=False, vad_filter=True
                    )

                    words_in_stride = []
                    for segment in segments:
                        if segment.words:
                            for w in segment.words:
                                if w.end > new_stride_start_time:
                                    adj_start = max(0.0, w.start - new_stride_start_time)
                                    adj_end = w.end - new_stride_start_time
                                    words_in_stride.append({"word": w.word.strip(), "start": adj_start, "end": adj_end})
                                    
                    toxic_intervals = toxic_censor_audio.detect_toxic_words(words_in_stride)

                    stride_audio = audio_window[-STRIDE_SAMPLES:]
                    final_audio = audio_processor.process_chunk(stride_audio, toxic_intervals)

                    if not self.output_audio_queue.full():
                        self.output_audio_queue.put(final_audio.reshape(-1, 1))

                    if self.av_container is not None and self.a_stream is not None:
                        av_audio = av.AudioFrame.from_ndarray(final_audio.reshape(1, -1), format="flt", layout="mono")
                        av_audio.sample_rate = SAMPLE_RATE
                        av_audio.time_base = Fraction(1, SAMPLE_RATE)
                        
                        with self.mux_lock:
                            self.audio_fifo.write(av_audio)
                            while True:
                                a_frame = self.audio_fifo.read(1024)
                                if a_frame is None:
                                    break
                                for packet in self.a_stream.encode(a_frame):
                                    self.av_container.mux(packet)

                    processing_time = time.time() - start_time
                    rtf = processing_time / STRIDE_DURATION
                    self.session_metrics.append({"rtf": rtf, "proc_time": processing_time})

                    if self.ui_update:
                        if toxic_intervals:
                            censored_count = len(toxic_intervals)
                            self.total_censored_words += censored_count
                            censored_words_str = ", ".join([f"{t['word']} [{t.get('label', 'ÁUDIO')}]" for t in toxic_intervals])
                            self.ui_update(msg=f"🔊 BLOQUEADO (ÁUDIO): {censored_words_str}", rtf_val=rtf, total_censored=self.total_censored_words)
                        else:
                            self.ui_update(msg=f"✓ Stride limpo.", rtf_val=rtf, total_censored=self.total_censored_words)
                    
        except Exception as e:
            self.is_running = False
            if self.ui_update: self.ui_update(msg=f"❌ Erro fatal: {str(e)}")

    def _export_session(self):
        # Apenas tenta exportar se o container realmente chegou a ser aberto (usuário não cancelou no carregamento)
        if self.av_container is not None:
            if self.ui_update: self.ui_update(msg="Salvando vídeo e empacotando Muxer...")
            
            with self.mux_lock:
                if self.v_stream is not None:
                    for packet in self.v_stream.encode():
                        self.av_container.mux(packet)
                
                if self.a_stream is not None:
                    while True:
                        a_frame = self.audio_fifo.read(1024)
                        if a_frame is None:
                            for packet in self.a_stream.encode():
                                self.av_container.mux(packet)
                            break
                        for packet in self.a_stream.encode(a_frame):
                            self.av_container.mux(packet)
                        
                self.av_container.close()
                self.av_container = None
                
            print("\n" + "="*50)
            print("✅ Vídeo final MUXADO salvo como 'live_output_censurado.mp4'!")
            print("="*50 + "\n")
            if self.ui_update: self.ui_update(msg="✅ Arquivo .mp4 finalizado com sucesso!")