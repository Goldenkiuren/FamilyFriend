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
WHISPER_RATE = 16000      # O Whisper exige 16kHz mono
MIC_RATE = 44100          # Taxa de captura do microfone (qualidade alta)
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
        ui_callback(msg="Carregando modelos na GPU... aguarde.")
        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        toxic_censor = ToxicCensor(threshold=threshold_value)
        ui_callback(msg="Modelos base prontos.")
    else:
        toxic_censor.threshold = threshold_value


def pure_record_callback(indata, frames, time_info, status):
    if status:
        print(f"[mic] status: {status}")
    if is_pure_recording:
        pure_record_buffer.append(indata.copy())


def _to_whisper_audio(proc_audio, proc_rate):
    """Cria uma cópia 16kHz mono apenas para o Whisper, sem mexer no áudio de trabalho."""
    if proc_rate == WHISPER_RATE:
        return np.ascontiguousarray(proc_audio, dtype=np.float32)
    tensor = torch.as_tensor(proc_audio, dtype=torch.float32).unsqueeze(0)
    out = torchaudio.functional.resample(tensor, proc_rate, WHISPER_RATE)
    return np.ascontiguousarray(out.squeeze(0).numpy(), dtype=np.float32)


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


def process_audio(proc_audio, proc_rate, subtype, is_file, save_path, ui_callback,
                  threshold_value, selected_mode, save_orig, save_transcripts):
    """Processa o áudio na sua taxa de amostragem NATIVA (preserva qualidade).
    O Whisper recebe uma cópia 16kHz separada; a edição e a saída ficam na taxa original."""
    global whisper_model, toxic_censor, voice_cloner

    try:
        ui_callback(msg="Iniciando processamento...")
        load_models_if_needed(ui_callback, threshold_value)

        modes_to_run = ["beep", "clone"] if selected_mode == "all" else [selected_mode]

        if ("clone" in modes_to_run or "rewrite" in modes_to_run) and voice_cloner is None:
            ui_callback(msg="Carregando motor de clonagem de voz (F5-TTS)...")
            from voice_cloner import VoiceCloner
            voice_cloner = VoiceCloner()
            ui_callback(msg="Motor de voz carregado.")

        start_time = time.time()

        # 1. Transcrição (em cópia 16kHz mono; o áudio de trabalho continua na taxa nativa)
        ui_callback(msg="Transcrevendo áudio com Whisper...")
        whisper_audio = _to_whisper_audio(proc_audio, proc_rate)
        segments, _ = whisper_model.transcribe(
            whisper_audio, language="en", word_timestamps=True, beam_size=5, vad_filter=True
        )

        words_found = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words_found.append({"word": w.word.strip(), "start": w.start, "end": w.end})

        original_text = " ".join([w["word"] for w in words_found])

        # AudioCensor opera na taxa NATIVA do áudio (tempos do Whisper são em segundos)
        offline_censor = AudioCensor(sample_rate=proc_rate, overlap_duration=0.0)

        base_dir = os.path.dirname(save_path)
        # Extrai o nome base e a extensão que o usuário escolheu
        base_name, ext = os.path.splitext(os.path.basename(save_path))
        if not ext:
            ext = ".wav"  # Fallback de segurança
            
        output_folder = os.path.join(base_dir, base_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # 2. Cópia do original (microfone)
        if save_orig and not is_file:
            orig_path = os.path.join(output_folder, f"{base_name}_original{ext}")
            # Evita usar 'subtype' em formatos comprimidos para não causar erro no soundfile
            if ext.lower() in [".flac", ".ogg"]:
                sf.write(orig_path, proc_audio, proc_rate)
            else:
                sf.write(orig_path, proc_audio, proc_rate, subtype=subtype)
            ui_callback(msg=f"Cópia original salva: {orig_path}")
        # 3. Processamento dos modos
        for mode in modes_to_run:
            if mode == "rewrite":
                ui_callback(msg="Modo de Reescrita Contextual ainda não implementado. Pulando.")
                continue

            ui_callback(msg=f"Analisando toxicidade — modo [{mode.upper()}]...")
            toxic_intervals = toxic_censor.detect_toxic_words(words_found, mode=mode)
            ui_callback(msg=f"{len(toxic_intervals)} ocorrência(s) encontrada(s). Editando...")

            if mode == "clone":
                final_audio = offline_censor.process_offline_replacement(
                    full_audio=proc_audio,
                    toxic_intervals=toxic_intervals,
                    words_list=words_found,
                    voice_cloner=voice_cloner,
                )
            elif mode == "beep":
                final_audio = offline_censor.process_chunk(
                    audio_chunk=proc_audio,
                    toxic_intervals=toxic_intervals,
                    is_first_chunk=True,
                )

            suffix = "" if len(modes_to_run) == 1 else f"_{mode}"
            
            # Atualiza para usar a extensão dinâmica {ext} em vez de .wav
            final_save_path = os.path.join(output_folder, f"{base_name}{suffix}{ext}")
            
            # Salva de forma segura dependendo do formato
            if ext.lower() in [".flac", ".ogg"]:
                sf.write(final_save_path, final_audio, proc_rate)
            else:
                sf.write(final_save_path, final_audio, proc_rate, subtype=subtype)
                
            ui_callback(msg=f"Áudio [{mode.upper()}] salvo: {final_save_path}")

            # 4. Transcrições (se solicitado)
            if save_transcripts:
                censored_text = generate_censored_text(words_found, toxic_intervals, mode)
                txt_path = os.path.join(output_folder, f"{base_name}{suffix}_transcript.txt")   
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"=== TRANSCRICAO ORIGINAL ===\n{original_text}\n\n")
                    f.write(f"=== TRANSCRICAO CENSURADA ({mode.upper()}) ===\n{censored_text}\n")
                ui_callback(msg=f"Transcrição salva: {txt_path}")

        proc_time = time.time() - start_time
        ui_callback(msg=f"Concluído em {proc_time:.1f}s  ({proc_rate} Hz, mono).")

    except Exception as e:
        import traceback
        traceback.print_exc()
        ui_callback(msg=f"Erro ao processar: {str(e)}")


# ==========================================
# INTERFACE GRÁFICA
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AudioBatchUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FamilyFriend AI — Censor de Áudio")
        self.geometry("920x620")
        self.minsize(920, 620)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_sidebar()
        self._build_main_panel()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar, text="FamilyFriend", font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(32, 4))

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar, text="Censor de Áudio por IA",
            font=ctk.CTkFont(size=12), text_color=("gray70", "gray60")
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 24))

        self.lbl_thresh = ctk.CTkLabel(
            self.sidebar, text="Sensibilidade (modo Reescrita): 0.50",
            font=ctk.CTkFont(size=12)
        )
        self.lbl_thresh.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")

        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=0.1, to=0.9, command=self._update_thresh_lbl)
        self.slider_thresh.set(0.5)
        self.slider_thresh.grid(row=5, column=0, padx=20, pady=(8, 20))

        self.version_label = ctk.CTkLabel(
            self.sidebar, text="v0.1", font=ctk.CTkFont(size=11), text_color=("gray60", "gray50")
        )
        self.version_label.grid(row=7, column=0, padx=20, pady=(0, 16), sticky="s")

    def _build_main_panel(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=24, pady=24, sticky="nsew")
        self.main_frame.grid_rowconfigure(3, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # 1. SELEÇÃO DE MODO
        self.mode_frame = ctk.CTkFrame(self.main_frame)
        self.mode_frame.grid(row=0, column=0, pady=(0, 12), sticky="ew", ipadx=10, ipady=10)

        ctk.CTkLabel(
            self.mode_frame, text="Modo de censura", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=14, pady=(8, 6))

        self.censor_mode_var = ctk.StringVar(value="beep")
        ctk.CTkRadioButton(self.mode_frame, text="1.  Bip sobre o palavrão (dicionário)",
                           variable=self.censor_mode_var, value="beep").pack(anchor="w", padx=22, pady=4)
        ctk.CTkRadioButton(self.mode_frame, text="2.  Substituir por sinônimo na voz original (F5-TTS)",
                           variable=self.censor_mode_var, value="clone").pack(anchor="w", padx=22, pady=4)
        ctk.CTkRadioButton(self.mode_frame, text="3.  Reescrita contextual (em breve)",
                           variable=self.censor_mode_var, value="rewrite", state="disabled").pack(anchor="w", padx=22, pady=4)
        ctk.CTkRadioButton(self.mode_frame, text="4.  Gerar todas as saídas (modos 1 e 2)",
                           variable=self.censor_mode_var, value="all").pack(anchor="w", padx=22, pady=4)

        # 2. OPÇÕES DE SALVAMENTO
        self.options_frame = ctk.CTkFrame(self.main_frame)
        self.options_frame.grid(row=1, column=0, pady=(0, 16), sticky="ew", ipadx=10, ipady=5)

        ctk.CTkLabel(
            self.options_frame, text="Opções de salvamento", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=14, pady=(8, 6))

        self.chk_save_orig_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.options_frame,
                        text="Salvar cópia do áudio original (apenas gravações do microfone)",
                        variable=self.chk_save_orig_var).pack(anchor="w", padx=22, pady=4)

        self.chk_save_trans_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.options_frame,
                        text="Salvar .txt com transcrições (original e censurada)",
                        variable=self.chk_save_trans_var).pack(anchor="w", padx=22, pady=4)

        # 3. BOTÕES DE ENTRADA
        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame.grid(row=2, column=0, pady=6)

        self.btn_load = ctk.CTkButton(
            self.buttons_frame, text="Carregar arquivo", height=48, width=210,
            font=ctk.CTkFont(size=14, weight="bold"), fg_color="#10B981", hover_color="#059669",
            command=self.load_audio_file
        )
        self.btn_load.pack(side="left", padx=10)

        self.btn_record = ctk.CTkButton(
            self.buttons_frame, text="Gravar microfone", height=48, width=210,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.toggle_pure_recording
        )
        self.btn_record.pack(side="left", padx=10)

        # 4. CONSOLE
        self.log_console = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.log_console.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        self.log_console.insert("0.0", "Sistema inicializado.\nCarregue um arquivo ou grave pelo microfone para começar.\n")
        self.log_console.configure(state="disabled")

    def _update_thresh_lbl(self, value):
        self.lbl_thresh.configure(text=f"Sensibilidade (modo Reescrita): {value:.2f}")

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
            initialfile=f"censurado_{os.path.splitext(os.path.basename(file_path))[0]}",
            filetypes=[
                ("Áudio WAV (Sem compressão)", "*.wav"),
                ("Áudio FLAC (Alta qualidade)", "*.flac"),
                ("Áudio OGG (Leve)", "*.ogg")
            ]
        )
        if not save_path:
            self.safe_ui_update(msg="Processamento cancelado (local de salvamento não escolhido).")
            return

        self.safe_ui_update(msg=f"\nCarregando arquivo: {os.path.basename(file_path)}")

        try:
            # Lê na taxa NATIVA (sem reamostrar) para preservar a qualidade
            waveform, sr = torchaudio.load(file_path)

            if waveform.shape[0] > 1:
                self.safe_ui_update(msg="Convertendo estéreo para mono...")
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            audio_array = np.ascontiguousarray(waveform.squeeze(0).numpy(), dtype=np.float32)

            # Tenta preservar a profundidade de bits (subtype) do arquivo de origem
            subtype = None
            try:
                info = sf.info(file_path)
                if info.subtype and info.subtype.upper().startswith(("PCM_", "FLOAT", "DOUBLE")):
                    subtype = info.subtype
            except Exception:
                subtype = None

            self.safe_ui_update(msg=f"Taxa de amostragem: {sr} Hz (preservada na saída).")
            self._start_processing_thread(audio_array, save_path, proc_rate=sr,
                                          subtype=subtype, is_file=True)

        except Exception as e:
            self.safe_ui_update(msg=f"Falha ao ler o arquivo de áudio: {str(e)}")

    def toggle_pure_recording(self):
        global is_pure_recording, pure_record_buffer

        if not is_pure_recording:
            is_pure_recording = True
            pure_record_buffer = []
            self.btn_record.configure(text="Parar e processar", fg_color="#EF4444", hover_color="#B91C1C")
            self.btn_load.configure(state="disabled")
            self.safe_ui_update(msg="\nGravando... fale no microfone.")

            self.rec_stream = sd.InputStream(
                samplerate=MIC_RATE, channels=CHANNELS,
                dtype='float32', callback=pure_record_callback
            )
            self.rec_stream.start()

        else:
            is_pure_recording = False
            self.rec_stream.stop()
            self.rec_stream.close()

            self.btn_record.configure(text="Gravar microfone", fg_color="#3B82F6", hover_color="#2563EB")
            self.btn_load.configure(state="normal")

            save_path = filedialog.asksaveasfilename(
                title="Salvar gravação censurada",
                defaultextension=".wav",
                filetypes=[
                    ("Áudio WAV (Sem compressão)", "*.wav"),
                    ("Áudio FLAC (Alta qualidade)", "*.flac"),
                    ("Áudio OGG (Leve)", "*.ogg")
                ]
            )

            if save_path:
                full_audio = np.concatenate(pure_record_buffer).flatten().astype(np.float32)
                self._start_processing_thread(full_audio, save_path, proc_rate=MIC_RATE,
                                              subtype="PCM_16", is_file=False)
            else:
                self.safe_ui_update(msg="Salvamento cancelado. Áudio descartado.")

    def _start_processing_thread(self, audio_array, save_path, proc_rate, subtype, is_file):
        current_threshold = self.slider_thresh.get()
        selected_mode = self.censor_mode_var.get()
        save_orig = self.chk_save_orig_var.get()
        save_trans = self.chk_save_trans_var.get()

        threading.Thread(
            target=process_audio,
            args=(audio_array, proc_rate, subtype, is_file, save_path, self.safe_ui_update,
                  current_threshold, selected_mode, save_orig, save_trans),
            daemon=True
        ).start()


if __name__ == "__main__":
    app = AudioBatchUI()
    app.mainloop()
