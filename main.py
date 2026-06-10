import queue
import time
import threading
import wave
import numpy as np
import sounddevice as sd
import soundfile as sf
import customtkinter as ctk
from tkinter import filedialog
from faster_whisper import WhisperModel

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

is_pure_recording = False
pure_record_buffer = []

# Buffers de sessão para exportação final
session_audio_buffer = []
session_metrics = []

# Modelos globais para não recarregar ao trocar de abas
whisper_model = None
toxic_censor = None
voice_cloner = None

# ==========================================
# THREADS DE PROCESSAMENTO (BACKEND)
# ==========================================

def load_models_if_needed(ui_callback, threshold_value):
    global whisper_model, toxic_censor
    if whisper_model is None or toxic_censor is None:
        ui_callback(msg="Carregando modelos na GPU... Aguarde.")
        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        toxic_censor = ToxicCensor(threshold=threshold_value)
        ui_callback(msg="✅ Modelos prontos!")
    else:
        # Atualiza o threshold caso o slider tenha sido mexido
        toxic_censor.threshold = threshold_value

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

def audio_playback_worker(play_realtime_ref):
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    stream.start()
    while is_running:
        try:
            audio_data = output_audio_queue.get(timeout=0.1)
            if play_realtime_ref():
                stream.write(audio_data)
        except queue.Empty:
            continue
    stream.stop()
    stream.close()

def main_processing_loop(ui_callback, threshold_value):
    global is_running, total_censored_words, session_audio_buffer, session_metrics
    
    try:
        load_models_if_needed(ui_callback, threshold_value)
        audio_processor = AudioCensor(sample_rate=SAMPLE_RATE, overlap_duration=OVERLAP_DURATION)
        
        ui_callback(msg="✅ Pipeline AO VIVO iniciado.")
        chunk_count = 0
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                            dtype='float32', callback=audio_producer_callback):
                            
            while is_running:
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                audio_data = chunk.flatten()
                
                # A: Transcrição
                segments, _ = whisper_model.transcribe(
                    audio_data, language="en", word_timestamps=True, beam_size=5
                )
                
                words_found = []
                for segment in segments:
                    if segment.words:
                        for w in segment.words:
                            words_found.append({"word": w.word.strip(), "start": w.start, "end": w.end})
                
                # B: IA de Censura
                toxic_intervals = toxic_censor.detect_toxic_words(words_found, use_synonyms=False)
                
                # C: Edição Matemática
                is_first = (chunk_count == 0)
                final_audio = audio_processor.process_chunk(audio_data, toxic_intervals, is_first_chunk=is_first)
                
                session_audio_buffer.append(final_audio)
                output_audio_queue.put(final_audio.reshape(-1, 1))
                
                # E: Métricas
                processing_time = time.time() - start_time
                rtf = processing_time / CHUNK_DURATION
                session_metrics.append({"rtf": rtf, "proc_time": processing_time})
                
                if toxic_intervals:
                    censored_count = len(toxic_intervals)
                    total_censored_words += censored_count
                    censored_words_str = ", ".join([f"{t['word']} [{t['label']}]" for t in toxic_intervals])
                    ui_callback(msg=f"🚨 BLOQUEADO: {censored_words_str}", rtf_val=rtf, new_censored=censored_count)
                else:
                    ui_callback(msg=f"✓ Chunk {chunk_count+1} limpo.", rtf_val=rtf, new_censored=0)
                    
                chunk_count += 1

    except Exception as e:
        is_running = False
        ui_callback(msg=f"❌ Erro fatal: {str(e)}")


def pure_record_callback(indata, frames, time_info, status):
    """Apenas guarda o áudio na memória, sem IA."""
    if status:
        print(f"⚠️ Status mic: {status}")
    if is_pure_recording:
        pure_record_buffer.append(indata.copy())

def process_recorded_buffer(audio_chunks, save_path, ui_callback, threshold_value, use_voice_cloning=True):
    """Pega o áudio gravado, passa na IA e salva no local escolhido."""
    global whisper_model, toxic_censor, voice_cloner
    
    try:
        if not audio_chunks:
            ui_callback(msg="⚠️ Nenhum áudio foi gravado.")
            return

        ui_callback(msg="⚙️ Processando a gravação offline... Aguarde.")
        full_audio = np.concatenate(audio_chunks).flatten()
        
        # 1. Carrega Whisper e RoBERTa
        load_models_if_needed(ui_callback, threshold_value)
        
        # 2. Lazy Loading do Voice Cloner (SÓ carrega se o usuário ativou o botão)
        if use_voice_cloning and voice_cloner is None:
            ui_callback(msg="🗣️ Carregando motor de clonagem de voz (F5-TTS)...")
            from voice_cloner import VoiceCloner
            voice_cloner = VoiceCloner()
            ui_callback(msg="✅ Motor de voz carregado!")

        start_time = time.time()
        
        segments, _ = whisper_model.transcribe(
            full_audio, language="en", word_timestamps=True, beam_size=5
        )
        
        words_found = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words_found.append({"word": w.word.strip(), "start": w.start, "end": w.end})
                    
                    
        ui_callback(msg="🕵️ Analisando toxicidade...")
        # Usa sinônimos apenas se o modo de voz estiver ativo
        toxic_intervals = toxic_censor.detect_toxic_words(words_found, use_synonyms=use_voice_cloning)
        
        ui_callback(msg=f"✂️ {len(toxic_intervals)} ofensas encontradas. Iniciando edição...")
        
        offline_censor = AudioCensor(sample_rate=SAMPLE_RATE, overlap_duration=0.0)
        
        # AQUI ESTÁ A BIFURCAÇÃO DA SUA NOVA FEATURE
        if use_voice_cloning:
            final_audio = offline_censor.process_offline_replacement(
                full_audio=full_audio, 
                toxic_intervals=toxic_intervals,
                words_list=words_found,
                voice_cloner=voice_cloner
            )
        else:
            ui_callback(msg="🔇 Modo Bip selecionado. Aplicando censura clássica...")
            # A função process_chunk serve perfeitamente para o áudio inteiro se overlap for 0
            final_audio = offline_censor.process_chunk(
                audio_chunk=full_audio, 
                toxic_intervals=toxic_intervals, 
                is_first_chunk=True
            )
        
        import soundfile as sf
        sf.write(save_path, final_audio, SAMPLE_RATE)
        
        proc_time = time.time() - start_time
        ui_callback(msg=f"✅ Áudio salvo com sucesso em:\n{save_path}")
        ui_callback(msg=f"⏱️ Tempo de Edição IA: {proc_time:.2f}s | Substituições: {len(toxic_intervals)}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        ui_callback(msg=f"❌ Erro ao processar gravação: {str(e)}")


def export_session():
    global session_audio_buffer, session_metrics
    if not session_audio_buffer:
        return
        
    full_audio = np.concatenate(session_audio_buffer)
    audio_int16 = np.int16(full_audio * 32767)
    
    filename = "output_censored.wav"
    with wave.open(filename, "w") as f:
        f.setnchannels(CHANNELS)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio_int16.tobytes())
        
    print(f"✅ Gravação Ao Vivo salva: {filename}")


# ==========================================
# INTERFACE GRÁFICA MODERNA
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AudioStreamUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FamilyFriend AI - Censor Pipeline")
        self.geometry("850x550")
        self.minsize(850, 550)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self._build_sidebar()
        self._build_main_panel()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1) 
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="FamilyFriend", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))
        
        self.status_badge = ctk.CTkLabel(self.sidebar, text="● INATIVO", text_color="gray", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_badge.grid(row=1, column=0, padx=20, pady=10)
        
        self.switch_realtime = ctk.CTkSwitch(self.sidebar, text="Som ao Vivo (Loopback)")
        self.switch_realtime.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="w")
        self.switch_realtime.deselect()
        
        self.lbl_thresh = ctk.CTkLabel(self.sidebar, text="Sensibilidade: 0.50", font=ctk.CTkFont(size=12))
        self.lbl_thresh.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=0.1, to=0.9, command=self._update_thresh_lbl)
        self.slider_thresh.set(0.5)
        self.slider_thresh.grid(row=5, column=0, padx=20, pady=(10, 20))

    def _build_main_panel(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # SISTEMA DE ABAS
        self.tabview = ctk.CTkTabview(self.main_frame, height=140)
        self.tabview.grid(row=0, column=0, pady=(0, 20), sticky="ew")
        self.tab_live = self.tabview.add("📡 Mic (Stream)")
        self.tab_file = self.tabview.add("📁 Gravação (Batch)")
        
        # --- CONTEÚDO ABA LIVE ---
        self.tab_live.grid_columnconfigure((0, 1), weight=1)
        self.rtf_box = ctk.CTkFrame(self.tab_live, fg_color="transparent")
        self.rtf_box.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        ctk.CTkLabel(self.rtf_box, text="REAL TIME FACTOR (RTF)", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.rtf_label = ctk.CTkLabel(self.rtf_box, text="--", font=ctk.CTkFont(size=28, weight="bold"))
        self.rtf_label.pack(anchor="w")
        
        self.blocks_box = ctk.CTkFrame(self.tab_live, fg_color="transparent")
        self.blocks_box.grid(row=0, column=1, padx=20, pady=10, sticky="e")
        ctk.CTkLabel(self.blocks_box, text="OFENSAS BLOQUEADAS", font=ctk.CTkFont(size=11)).pack(anchor="e")
        self.blocks_label = ctk.CTkLabel(self.blocks_box, text="0", font=ctk.CTkFont(size=28, weight="bold"), text_color="#EF4444")
        self.blocks_label.pack(anchor="e")
        
        self.btn_toggle = ctk.CTkButton(self.tab_live, text="INICIAR PIPELINE", height=40, fg_color="#10B981", hover_color="#059669", font=ctk.CTkFont(weight="bold"), command=self.toggle_stream)
        self.btn_toggle.grid(row=1, column=0, columnspan=2, pady=10)
        
        # --- CONTEÚDO ABA GRAVAÇÃO ---
        self.tab_file.grid_columnconfigure(0, weight=1)
        
        # AQUI: O novo botão para escolher o modo
        self.switch_censor_mode = ctk.CTkSwitch(self.tab_file, text="Usar IA de Clonagem de Voz", font=ctk.CTkFont(weight="bold"))
        self.switch_censor_mode.grid(row=0, column=0, pady=(40, 10))
        self.switch_censor_mode.select() # Deixar ativado por padrão
        
        self.btn_record = ctk.CTkButton(
            self.tab_file, 
            text="🔴 INICIAR GRAVAÇÃO", 
            height=60, 
            font=ctk.CTkFont(size=16, weight="bold"), 
            command=self.toggle_pure_recording
        )
        self.btn_record.grid(row=1, column=0, pady=20) # Mudei o row de 0 para 1
        
        # --- CONSOLE COMPARTILHADO ---
        self.log_console = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.log_console.grid(row=1, column=0, sticky="nsew")
        self.log_console.insert("0.0", "Sistema inicializado. Escolha uma aba acima.\n")
        self.log_console.configure(state="disabled")

    def _update_thresh_lbl(self, value):
        self.lbl_thresh.configure(text=f"Sensibilidade: {value:.2f}")
        
    def _is_realtime_enabled(self):
        return self.switch_realtime.get() == 1

    def safe_ui_update(self, msg=None, rtf_val=None, new_censored=0):
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

    def toggle_pure_recording(self):
        global is_pure_recording, pure_record_buffer
        
        if is_running:
            self.safe_ui_update(msg="⚠️ Pare o modo 'Mic (Stream)' antes de usar a gravação.")
            return

        if not is_pure_recording:
            # INICIA A GRAVAÇÃO
            is_pure_recording = True
            pure_record_buffer = []
            self.btn_record.configure(text="⏹️ PARAR E SALVAR", fg_color="#EF4444", hover_color="#B91C1C")
            self.safe_ui_update(msg="\n🎙️ GRAVANDO... Fale no microfone.")
            
            # Abre o microfone apenas para armazenar os dados
            self.rec_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, 
                dtype='float32', callback=pure_record_callback
            )
            self.rec_stream.start()
            
        else:
            # PARA A GRAVAÇÃO E PEDE O NOME DO ARQUIVO
            is_pure_recording = False
            self.rec_stream.stop()
            self.rec_stream.close()
            
            self.btn_record.configure(text="🔴 INICIAR GRAVAÇÃO", fg_color="#3B82F6", hover_color="#2563EB")
            
            # AQUI: Pede para VOCÊ escolher o nome do arquivo que será salvo
            save_path = filedialog.asksaveasfilename(
                title="Salvar gravação censurada",
                defaultextension=".wav",
                filetypes=[("Áudio WAV", "*.wav")]
            )
            
            if save_path:
                current_threshold = self.slider_thresh.get()
                use_voice = self.switch_censor_mode.get() == 1  # <--- Lê o botão (1 é ativado, 0 desativado)
                
                # Roda a IA em uma Thread separada para não travar a tela
                threading.Thread(
                    target=process_recorded_buffer, 
                    args=(pure_record_buffer.copy(), save_path, self.safe_ui_update, current_threshold, use_voice), # <--- Adicione use_voice aqui
                    daemon=True
                ).start()
            else:
                self.safe_ui_update(msg="❌ Salvamento cancelado. Áudio descartado.")

    def toggle_stream(self):
        global is_running, total_censored_words, session_audio_buffer, session_metrics
        
        if not is_running:
            is_running = True
            total_censored_words = 0
            session_audio_buffer = []
            session_metrics = []
            
            self.status_badge.configure(text="● GRAVANDO", text_color="#EF4444")
            self.btn_toggle.configure(text="PARAR E SALVAR", fg_color="#EF4444", hover_color="#B91C1C")
            
            # CORREÇÃO AQUI: Atualizado para o novo nome do botão
            self.btn_record.configure(state="disabled") 
            
            self.blocks_label.configure(text="0")
            self.rtf_label.configure(text="--", text_color="white")
            self.safe_ui_update(msg="\n" + "="*40 + "\n🚀 INICIANDO PIPELINE DE ÁUDIO...\n" + "="*40)
            
            while not raw_audio_queue.empty(): raw_audio_queue.get()
            while not chunk_queue.empty(): chunk_queue.get()
            while not output_audio_queue.empty(): output_audio_queue.get()
            
            current_threshold = self.slider_thresh.get()
            
            threading.Thread(target=chunking_worker, daemon=True).start()
            threading.Thread(target=audio_playback_worker, args=(self._is_realtime_enabled,), daemon=True).start()
            threading.Thread(target=main_processing_loop, args=(self.safe_ui_update, current_threshold), daemon=True).start()
            
        else:
            is_running = False
            self.status_badge.configure(text="● INATIVO", text_color="gray")
            self.btn_toggle.configure(text="INICIAR PIPELINE", fg_color="#10B981", hover_color="#059669")
            
            # CORREÇÃO AQUI: Atualizado para o novo nome do botão
            self.btn_record.configure(state="normal")
            
            self.safe_ui_update(msg="\n🛑 COMPILANDO ÁUDIO AO VIVO...")
            threading.Thread(target=export_session, daemon=True).start()

if __name__ == "__main__":
    app = AudioStreamUI()
    def on_closing():
        global is_running
        is_running = False
        app.destroy()
        
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()