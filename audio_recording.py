import queue
import time
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Configurações do Áudio
SAMPLE_RATE = 16000       
CHUNK_DURATION = 3.0      
OVERLAP_DURATION = 0.5    
CHANNELS = 1              

CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
OVERLAP_SAMPLES = int(OVERLAP_DURATION * SAMPLE_RATE)
STEP_SAMPLES = CHUNK_SAMPLES - OVERLAP_SAMPLES

raw_audio_queue = queue.Queue()
chunk_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ Status do microfone: {status}", flush=True)
    raw_audio_queue.put(indata.copy())

def chunking_worker():
    buffer = np.zeros((0, CHANNELS), dtype=np.float32)
    while True:
        data = raw_audio_queue.get()
        if data is None:
            break
        buffer = np.vstack((buffer, data))
        while len(buffer) >= CHUNK_SAMPLES:
            chunk = buffer[:CHUNK_SAMPLES]
            chunk_queue.put(chunk)
            buffer = buffer[STEP_SAMPLES:]

def main():
    # ---------------------------------------------------------
    # MUDANÇA AQUI: Usando modelo em Inglês, CUDA e FP16
    # ---------------------------------------------------------
    print("🤖 Carregando o modelo Faster-Whisper (Medium.en na GPU)...")
    # Tente "small.en" primeiro. Se quiser ainda mais precisão e o RTF continuar < 1, suba para "large.en"
    model = WhisperModel("large-v3", device="cuda", compute_type="float16", vad_filter=True)
    print("✅ Modelo carregado com sucesso!\n")

    chunker_thread = threading.Thread(target=chunking_worker, daemon=True)
    chunker_thread.start()

    print(f"🎤 Gravando... Fale no microfone em Inglês. (Janela: {CHUNK_DURATION}s | Overlap: {OVERLAP_DURATION}s)")
    print("Pressione Ctrl+C para encerrar.\n")
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                            dtype='float32', callback=audio_callback):
            while True:
                chunk = chunk_queue.get()
                audio_data = chunk.flatten()
                
                start_time = time.time()
                
                segments, info = model.transcribe(
                    audio_data, 
                    language="en",        # <--- Forçando o idioma para Inglês
                    word_timestamps=True, 
                    beam_size=5,         # Na 4080 podemos usar 5 (padrão) para melhor precisão sem lag
                    condition_on_previous_text=False
                )
                
                words_found = []
                for segment in segments:
                    if segment.words:
                        for w in segment.words:
                            words_found.append({
                                "word": w.word.strip(),
                                "start": round(w.start, 2),
                                "end": round(w.end, 2)
                            })
                
                end_time = time.time()
                
                processing_time = end_time - start_time
                rtf = processing_time / CHUNK_DURATION
                
                print("-" * 60)
                status_rtf = "🟢 SAUDÁVEL" if rtf <= 1.0 else "🔴 LENTO (Overhead!)"
                print(f"⏱️ Tempo de Processamento: {processing_time:.3f}s | RTF: {rtf:.2f} ({status_rtf})")
                
                if words_found:
                    print("🗣️ Palavras detectadas no bloco:")
                    for w in words_found:
                        print(f"  └─ [{w['start']}s -> {w['end']}s]: \"{w['word']}\"")
                else:
                    print("... (Silêncio ou sem fala compreensível) ...")
                print("-" * 60 + "\n")
                
    except KeyboardInterrupt:
        print("\n🛑 Encerrando sistema...")
        raw_audio_queue.put(None)
        chunker_thread.join()

if __name__ == "__main__":
    main()