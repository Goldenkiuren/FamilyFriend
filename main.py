import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import customtkinter as ctk
from tkinter import filedialog
import os
import torch
import torchaudio
from faster_whisper import WhisperModel

from toxic_classifier import ToxicCensor
from audio_censor import AudioCensor

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
SAMPLE_RATE = 16000
CHANNELS = 1

is_pure_recording = False
pure_record_buffer = []

# Modelos globais
whisper_model = None
toxic_censor = None
voice_cloner = None

# ==========================================
# BACKEND & PROCESSAMENTO
# ==========================================

def load_models_if_needed(ui_callback, threshold_value):
    global whisper_model, toxic_censor
    if whisper_model is None or toxic_censor is None:
        ui_callback(msg="Carregando modelos na GPU... Aguarde.")
        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        toxic_censor = ToxicCensor(threshold=threshold_value)
        ui_callback(msg="✅ Modelos base prontos!")
    else:
        toxic_censor.threshold = threshold_value

def pure_record_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ Status mic: {status}")
    if is_pure_recording:
        pure_record_buffer.append(indata.copy())

def generate_censored_text(words_list, toxic_intervals, mode):
    """Gera a string do texto censurado com base nos intervalos."""
    toxic_map = {round(t["start"], 2): t for t in toxic_intervals}
    censored_words = []
    
    for w in words_list:
        start_round = round(w["start"], 2)
        if start_round in toxic_map:
            rep = toxic_map[start_round].get("replacement")
            if mode == "beep" or rep is None:
                censored_words.append("[BIP]")
            else:
                censored_words.append(rep.upper())
        else:
            censored_words.append(w["word"])
            
    return " ".join(censored_words)

def process_audio(full_audio, save_path, ui_callback, threshold_value, selected_mode, save_orig, save_transcripts):
    """Processa o array de áudio (seja gravado do mic ou carregado de arquivo)."""
    global whisper_model, toxic_censor, voice_cloner
    
    try:
        ui_callback(msg="⚙️ Iniciando processamento... Aguarde.")
        load_models_if_needed(ui_callback, threshold_value)
        
        modes_to_run = ["beep", "clone"] if selected_mode == "all" else [selected_mode]
        
        if ("clone" in modes_to_run or "rewrite" in modes_to_run) and voice_cloner is None:
            ui_callback(msg="🗣️ Carregando motor de clonagem de voz (F5-TTS)...")
            from voice_cloner import VoiceCloner
            voice_cloner = VoiceCloner()
            ui_callback(msg="✅ Motor de voz carregado!")

        start_time = time.time()
        
        # 1. Transcrição Inicial
        ui_callback(msg="📝 Transcrevendo áudio com Whisper...")
        segments, _ = whisper_model.transcribe(
            full_audio, language="en", word_timestamps=True, beam_size=5, vad_filter=True
        )
        
        words_found = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words_found.append({"word": w.word.strip(), "start": w.start, "end": w.end})
                    
        original_text = " ".join([w["word"] for w in words_found])
        
        offline_censor = AudioCensor(sample_rate=SAMPLE_RATE, overlap_duration=0.0)
        
        base_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        
        # 2. Salvamento do Áudio Original (se solicitado)
        if save_orig:
            orig_path = os.path.join(base_dir, f"{base_name}_original.wav")
            sf.write(orig_path, full_audio, SAMPLE_RATE)
            ui_callback(msg=f"💾 Cópia original salva em: {orig_path}")

        # 3. Processamento dos Modos
        for mode in modes_to_run:
            if mode == "rewrite":
                ui_callback(msg="⚠️ Modo de Reescrita Contextual ainda não implementado. Pulando...")
                continue
                
            ui_callback(msg=f"\n🕵️ Analisando toxicidade para o modo: [{mode.upper()}]")
            toxic_intervals = toxic_censor.detect_toxic_words(words_found, mode=mode)
            ui_callback(msg=f"✂️ {len(toxic_intervals)} ofensas encontradas. Iniciando edição...")
            
            # Geração de Áudio Censurado
            if mode == "clone":
                final_audio = offline_censor.process_offline_replacement(
                    full_audio=full_audio, 
                    toxic_intervals=toxic_intervals,
                    words_list=words_found,
                    voice_cloner=voice_cloner
                )
            elif mode == "beep":
                final_audio = offline_censor.process_chunk(
                    audio_chunk=full_audio, 
                    toxic_intervals=toxic_intervals, 
                    is_first_chunk=True
                )
                
            suffix = "" if len(modes_to_run) == 1 else f"_{mode}"
            final_save_path = os.path.join(base_dir, f"{base_name}{suffix}.wav")
            sf.write(final_save_path, final_audio, SAMPLE_RATE)
            ui_callback(msg=f"✅ Áudio [{mode.upper()}] salvo em: {final_save_path}")
            
            # 4. Salvamento de Transcrições (se solicitado)
            if save_transcripts:
                censored_text = generate_censored_text(words_found, toxic_intervals, mode)
                txt_path = os.path.join(base_dir, f"{base_name}{suffix}_transcript.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"=== TRANSCRIÇÃO ORIGINAL ===\n{original_text}\n\n")
                    f.write(f"=== TRANSCRIÇÃO CENSURADA ({mode.upper()}) ===\n{censored_text}\n")
                ui_callback(msg=f"📄 Transcrição salva em: {txt_path}")
            
        proc_time = time.time() - start_time
        ui_callback(msg=f"\n🎉 Processamento concluído! Tempo total: {proc_time:.2f}s")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        ui_callback(msg=f"❌ Erro ao processar: {str(e)}")

# ==========================================
# INTERFACE GRÁFICA MODERNA
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AudioBatchUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FamilyFriend AI - Batch Censor")
        self.geometry("900x600")
        self.minsize(900, 600)
        
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
        
        self.lbl_thresh = ctk.CTkLabel(self.sidebar, text="Sensibilidade RoBERTa: 0.50", font=ctk.CTkFont(size=12))
        self.lbl_thresh.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=0.1, to=0.9, command=self._update_thresh_lbl)
        self.slider_thresh.set(0.5)
        self.slider_thresh.grid(row=5, column=0, padx=20, pady=(10, 20))

    def _build_main_panel(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(3, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # 1. SELEÇÃO DE MODO DE CENSURA
        self.mode_frame = ctk.CTkFrame(self.main_frame)
        self.mode_frame.grid(row=0, column=0, pady=(0, 10), sticky="ew", ipadx=10, ipady=10)
        
        ctk.CTkLabel(self.mode_frame, text="Selecione o Modo de Censura:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        self.censor_mode_var = ctk.StringVar(value="beep")
        ctk.CTkRadioButton(self.mode_frame, text="1. Censura com Bip (Apenas Dicionário)", variable=self.censor_mode_var, value="beep").pack(anchor="w", padx=20, pady=5)
        ctk.CTkRadioButton(self.mode_frame, text="2. Clonagem de Voz Textual (Dicionário + F5-TTS)", variable=self.censor_mode_var, value="clone").pack(anchor="w", padx=20, pady=5)
        ctk.CTkRadioButton(self.mode_frame, text="3. Reescrita Contextual (Em Breve)", variable=self.censor_mode_var, value="rewrite", state="disabled").pack(anchor="w", padx=20, pady=5)
        ctk.CTkRadioButton(self.mode_frame, text="4. Gerar Todas as Saídas Disponíveis (Modo 1 e 2)", variable=self.censor_mode_var, value="all").pack(anchor="w", padx=20, pady=5)

        # 2. OPÇÕES DE EXPORTAÇÃO (Checkboxes)
        self.options_frame = ctk.CTkFrame(self.main_frame)
        self.options_frame.grid(row=1, column=0, pady=(0, 15), sticky="ew", ipadx=10, ipady=5)
        
        ctk.CTkLabel(self.options_frame, text="Opções de Salvamento:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        self.chk_save_orig_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.options_frame, text="Salvar cópia do áudio original", variable=self.chk_save_orig_var).pack(anchor="w", padx=20, pady=5)
        
        self.chk_save_trans_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.options_frame, text="Salvar arquivo .txt com transcrições (Original e Censurada)", variable=self.chk_save_trans_var).pack(anchor="w", padx=20, pady=5)

        # 3. BOTÕES DE ENTRADA (Mic e Arquivo)
        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame.grid(row=2, column=0, pady=5)
        
        self.btn_load = ctk.CTkButton(
            self.buttons_frame, text="📁 CARREGAR ARQUIVO", height=50, width=200, 
            font=ctk.CTkFont(size=14, weight="bold"), fg_color="#10B981", hover_color="#059669",
            command=self.load_audio_file
        )
        self.btn_load.pack(side="left", padx=10)
        
        self.btn_record = ctk.CTkButton(
            self.buttons_frame, text="🔴 GRAVAR MICROFONE", height=50, width=200, 
            font=ctk.CTkFont(size=14, weight="bold"), 
            command=self.toggle_pure_recording
        )
        self.btn_record.pack(side="left", padx=10)
        
        # 4. CONSOLE
        self.log_console = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.log_console.grid(row=3, column=0, sticky="nsew", pady=(10,0))
        self.log_console.insert("0.0", "Sistema Batch inicializado.\nCarregue um arquivo ou grave o microfone.\n")
        self.log_console.configure(state="disabled")

    def _update_thresh_lbl(self, value):
        self.lbl_thresh.configure(text=f"Sensibilidade RoBERTa: {value:.2f}")

    def safe_ui_update(self, msg=None):
        self.after(0, self._apply_ui_updates, msg)

    def _apply_ui_updates(self, msg):
        if msg:
            self.log_console.configure(state="normal")
            self.log_console.insert("end", msg + "\n")
            self.log_console.see("end")
            self.log_console.configure(state="disabled")

    def load_audio_file(self):
        file_path = filedialog.askopenfilename(
            title="Selecione um arquivo de áudio",
            filetypes=[("Arquivos de Áudio", "*.wav *.mp3 *.ogg *.flac *.m4a")]
        )
        
        if not file_path:
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Onde salvar o resultado processado?",
            defaultextension=".wav",
            initialfile=f"censurado_{os.path.basename(file_path)}",
            filetypes=[("Áudio WAV", "*.wav")]
        )
        
        if not save_path:
            self.safe_ui_update(msg="❌ Processamento cancelado. Local de salvamento não escolhido.")
            return

        self.safe_ui_update(msg=f"\n📂 Carregando arquivo: {os.path.basename(file_path)}")
        
        try:
            # Usa o torchaudio para ler qualquer formato e converter para as especificações exatas
            waveform, sr = torchaudio.load(file_path)
            
            # Força Reamostragem se não for 16kHz
            if sr != SAMPLE_RATE:
                self.safe_ui_update(msg=f"🔄 Reamostrando de {sr}Hz para {SAMPLE_RATE}Hz...")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # Força Mono se for Estéreo (fazendo a média dos canais)
            if waveform.shape[0] > 1:
                self.safe_ui_update(msg="🔄 Convertendo áudio estéreo para mono...")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            audio_array = waveform.squeeze().numpy()
            
            self._start_processing_thread(audio_array, save_path)
            
        except Exception as e:
            self.safe_ui_update(msg=f"❌ Falha ao ler arquivo de áudio: {str(e)}")

    def toggle_pure_recording(self):
        global is_pure_recording, pure_record_buffer
        
        if not is_pure_recording:
            is_pure_recording = True
            pure_record_buffer = []
            self.btn_record.configure(text="⏹️ PARAR E PROCESSAR", fg_color="#EF4444", hover_color="#B91C1C")
            self.btn_load.configure(state="disabled")
            self.safe_ui_update(msg="\n🎙️ GRAVANDO... Fale no microfone.")
            
            self.rec_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, 
                dtype='float32', callback=pure_record_callback
            )
            self.rec_stream.start()
            
        else:
            is_pure_recording = False
            self.rec_stream.stop()
            self.rec_stream.close()
            
            self.btn_record.configure(text="🔴 GRAVAR MICROFONE", fg_color="#3B82F6", hover_color="#2563EB")
            self.btn_load.configure(state="normal")
            
            save_path = filedialog.asksaveasfilename(
                title="Salvar gravação censurada",
                defaultextension=".wav",
                filetypes=[("Áudio WAV", "*.wav")]
            )
            
            if save_path:
                full_audio = np.concatenate(pure_record_buffer).flatten()
                self._start_processing_thread(full_audio, save_path)
            else:
                self.safe_ui_update(msg="❌ Salvamento cancelado. Áudio descartado.")

    def _start_processing_thread(self, audio_array, save_path):
        current_threshold = self.slider_thresh.get()
        selected_mode = self.censor_mode_var.get()
        save_orig = self.chk_save_orig_var.get()
        save_trans = self.chk_save_trans_var.get()
        
        threading.Thread(
            target=process_audio, 
            args=(audio_array, save_path, self.safe_ui_update, current_threshold, selected_mode, save_orig, save_trans),
            daemon=True
        ).start()

if __name__ == "__main__":
    app = AudioBatchUI()
    app.mainloop()