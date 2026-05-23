import queue
import time
import threading
import numpy as np
import sounddevice as sd
import customtkinter as ctk
from faster_whisper import WhisperModel

# Importa os módulos que criamos
from toxic_classifier import ToxicCensor
from audio_censor import AudioCensor

# ==========================================
# CONFIGURAÇÕES GERAIS E FILAS
# ==========================================
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
OVERLAP_DURATION = 0.5
CHANNELS = 1

CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
OVERLAP_SAMPLES = int(OVERLAP_DURATION * SAMPLE_RATE)
STEP_SAMPLES = CHUNK_SAMPLES - OVERLAP_SAMPLES

raw_audio_queue = queue.Queue()
chunk_queue = queue.Queue()
output_audio_queue = queue.Queue()

is_running = False
total_censored_words = 0

# ==========================================
# THREADS DE PROCESSAMENTO (BACKEND)
# ==========================================

def audio_producer_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ Status mic: {status}")
    if is_running:
        raw_audio_queue.put(indata.copy())

def chunking_worker():
    buffer = np.zeros((0, CHANNELS), dtype=np.float32)
    while is_running:
        try:
            data = raw_audio_queue.get(timeout=0.1)
            buffer = np.vstack((buffer, data))
            
            while len(buffer) >= CHUNK_SAMPLES:
                chunk = buffer[:CHUNK_SAMPLES]
                chunk_queue.put(chunk)
                buffer = buffer[STEP_SAMPLES:]
        except queue.Empty:
            continue

def audio_playback_worker():
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    stream.start()
    while is_running:
        try:
            audio_data = output_audio_queue.get(timeout=0.1)
            stream.write(audio_data)
        except queue.Empty:
            continue
    stream.stop()
    stream.close()

def main_processing_loop(ui_callback, threshold_value):
    """Loop central executando as inferências"""
    global is_running, total_censored_words
    
    ui_callback(msg="Carregando modelos na RTX 4080... Aguarde.")
    
    try:
        whisper_model = WhisperModel("small.en", device="cuda", compute_type="float16")
        toxic_censor = ToxicCensor(threshold=threshold_value)
        audio_processor = AudioCensor(sample_rate=SAMPLE_RATE, overlap_duration=OVERLAP_DURATION)
        
        ui_callback(msg="✅ Modelos prontos! Pipeline AO VIVO.")
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                            dtype='float32', callback=audio_producer_callback):
                            
            while is_running:
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                audio_data = chunk.flatten()
                
                # Transcrição Whisper
                segments, _ = whisper_model.transcribe(
                    audio_data, language="en", word_timestamps=True, beam_size=5
                )
                
                words_found = []
                for segment in segments:
                    if segment.words:
                        for w in segment.words:
                            words_found.append({
                                "word": w.word.strip(),
                                "start": w.start,
                                "end": w.end
                            })
                
                # Detecção e Censura
                toxic_intervals = toxic_censor.detect_toxic_words(words_found)
                final_audio = audio_processor.process_chunk(audio_data, toxic_intervals)
                output_audio_queue.put(final_audio.reshape(-1, 1))
                
                # Cálculo de Métricas
                rtf = (time.time() - start_time) / CHUNK_DURATION
                
                if toxic_intervals:
                    censored_count = len(toxic_intervals)
                    total_censored_words += censored_count
                    censored_words_str = ", ".join([t['word'] for t in toxic_intervals])
                    log_msg = f"🚨 BLOQUEADO: {censored_words_str}"
                    ui_callback(msg=log_msg, rtf_val=rtf, new_censored=censored_count)
                else:
                    log_msg = f"✓ Chunk limpo processado ({len(words_found)} palavras)."
                    ui_callback(msg=log_msg, rtf_val=rtf, new_censored=0)

    except Exception as e:
        is_running = False
        ui_callback(msg=f"❌ Erro fatal: {str(e)}")


# ==========================================
# INTERFACE GRÁFICA MODERNA (CustomTkinter)
# ==========================================

# Configuração do tema
ctk.set_appearance_mode("dark")  # Opções: "dark", "light", "system"
ctk.set_default_color_theme("blue")

class AudioStreamUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("FamilyFriend AI - Censor Pipeline")
        self.geometry("800x500")
        self.minsize(800, 500)
        
        # Grid Principal: 1 linha, 2 colunas (Menu lateral e Painel central)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self._build_sidebar()
        self._build_main_panel()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1) # Joga os itens pro topo
        
        # Logo / Titulo
        self.logo_label = ctk.CTkLabel(self.sidebar, text="🛡️ FamilyFriend", 
                                       font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))
        
        # Status Badge
        self.status_badge = ctk.CTkLabel(self.sidebar, text="● INATIVO", 
                                         text_color="gray", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_badge.grid(row=1, column=0, padx=20, pady=10)
        
        # Botão Iniciar
        self.btn_toggle = ctk.CTkButton(self.sidebar, text="INICIAR PIPELINE", height=40,
                                        fg_color="#10B981", hover_color="#059669", 
                                        font=ctk.CTkFont(weight="bold"), command=self.toggle_stream)
        self.btn_toggle.grid(row=2, column=0, padx=20, pady=20)
        
        # Slider de Threshold
        self.lbl_thresh = ctk.CTkLabel(self.sidebar, text="Sensibilidade do BERT: 0.50", font=ctk.CTkFont(size=12))
        self.lbl_thresh.grid(row=3, column=0, padx=20, pady=(20, 0), sticky="w")
        
        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=0.1, to=0.9, command=self._update_thresh_lbl)
        self.slider_thresh.set(0.5)
        self.slider_thresh.grid(row=4, column=0, padx=20, pady=(10, 20))

    def _build_main_panel(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Dashboard de Métricas (Topo)
        self.dash_frame = ctk.CTkFrame(self.main_frame, height=80)
        self.dash_frame.grid(row=0, column=0, pady=(0, 20), sticky="ew")
        self.dash_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Box RTF
        self.rtf_box = ctk.CTkFrame(self.dash_frame, fg_color="transparent")
        self.rtf_box.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        ctk.CTkLabel(self.rtf_box, text="REAL TIME FACTOR (RTF)", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w")
        self.rtf_label = ctk.CTkLabel(
            self.rtf_box,
            text="--",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.rtf_label.pack(anchor="w")
        
        # Box Palavrões Bloqueados
        self.blocks_box = ctk.CTkFrame(self.dash_frame, fg_color="transparent")
        self.blocks_box.grid(row=0, column=1, padx=20, pady=15, sticky="e")
        ctk.CTkLabel(self.blocks_box, text="OFENSAS BLOQUEADAS", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="e")
        self.blocks_label = ctk.CTkLabel(
            self.blocks_box,
            text="0",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#EF4444"
        )
        self.blocks_label.pack(anchor="e")
        # Console de Logs (Centro)
        self.log_console = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.log_console.grid(row=1, column=0, sticky="nsew")
        self.log_console.insert("0.0", "Sistema inicializado. Aguardando comando...\n")
        self.log_console.configure(state="disabled")

    def _update_thresh_lbl(self, value):
        self.lbl_thresh.configure(text=f"Sensibilidade do BERT: {value:.2f}")

    def safe_ui_update(self, msg=None, rtf_val=None, new_censored=0):
        """Método Thread-Safe para atualizar a UI a partir da thread de IA"""
        self.after(0, self._apply_ui_updates, msg, rtf_val, new_censored)

    def _apply_ui_updates(self, msg, rtf_val, new_censored):
        if msg:
            self.log_console.configure(state="normal")
            self.log_console.insert("end", msg + "\n")
            self.log_console.see("end")
            self.log_console.configure(state="disabled")
            
        if rtf_val is not None:
            color = "#10B981" if rtf_val <= 1.0 else "#EF4444"
            self.rtf_label.configure(text=f"{rtf_val:.3f}x", text_color=color)
            
        if new_censored > 0:
            self.blocks_label.configure(text=str(total_censored_words))

    def toggle_stream(self):
        global is_running, total_censored_words
        
        if not is_running:
            is_running = True
            total_censored_words = 0
            
            # Atualiza Visuais
            self.status_badge.configure(text="● AO VIVO", text_color="#EF4444")
            self.btn_toggle.configure(text="PARAR PIPELINE", fg_color="#EF4444", hover_color="#B91C1C")
            self.slider_thresh.configure(state="disabled")
            self.blocks_label.configure(text="0")
            self.rtf_label.configure(text="--", text_color="white")
            self.safe_ui_update(msg="\n" + "="*40 + "\n🚀 INICIANDO PIPELINE DE ÁUDIO...\n" + "="*40)
            
            # Limpa lixo de memória
            while not raw_audio_queue.empty(): raw_audio_queue.get()
            while not chunk_queue.empty(): chunk_queue.get()
            while not output_audio_queue.empty(): output_audio_queue.get()
            
            # Pega o valor do slider
            current_threshold = self.slider_thresh.get()
            
            # Dispara as threads
            threading.Thread(target=chunking_worker, daemon=True).start()
            threading.Thread(target=audio_playback_worker, daemon=True).start()
            threading.Thread(target=main_processing_loop, args=(self.safe_ui_update, current_threshold), daemon=True).start()
            
        else:
            is_running = False
            self.status_badge.configure(text="● INATIVO", text_color="gray")
            self.btn_toggle.configure(text="INICIAR PIPELINE", fg_color="#10B981", hover_color="#059669")
            self.slider_thresh.configure(state="normal")
            self.safe_ui_update(msg="\n🛑 PIPELINE INTERROMPIDA PELO USUÁRIO.")

if __name__ == "__main__":
    app = AudioStreamUI()
    # Garante encerramento limpo das threads
    def on_closing():
        global is_running
        is_running = False
        app.destroy()
        
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()