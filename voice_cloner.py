import os
import tempfile
import numpy as np
import soundfile as sf
import torch
import torchaudio
from f5_tts.api import F5TTS

class VoiceCloner:
    def __init__(self, device="cuda"):
        print("[VoiceCloner] Inicializando F5-TTS (Flow Matching) na GPU...")

        # A API oficial do F5-TTS baixa os pesos automaticamente na primeira execução
        self.tts = F5TTS(device=device)
        self.device = device
        self.f5_rate = 24000  # O F5-TTS gera nativamente em 24kHz
        print("[VoiceCloner] F5-TTS pronto para sintese Zero-Shot.")

    def generate_replacement(self, reference_audio_array, ref_text, text_to_say,
                             nfe_step=48, cfg_strength=2.0, remove_silence=False, target_rms=0.1,
                             speed=1.0, output_rate=24000):
        """
        Gera a frase-portadora (carrier) usando a voz original do falante.
        - reference_audio_array: Áudio de referência LIMPO do falante (NumPy 16kHz).
          Deve ser um trecho longo (idealmente 3-8s) e sem palavrões para clonar bem o timbre.
        - ref_text: Transcrição exata desse áudio de referência (obrigatório no F5).
        - text_to_say: A frase completa com o sinônimo no lugar do palavrão (dá contexto/coarticulação).
        - nfe_step: passos do flow-matching (mais alto = mais qualidade, mais lento). 48 aproveita a 4080.
        - remove_silence: corta silêncios de borda gerados pelo F5.
        """
        # 1. Cria um arquivo WAV temporário na memória para o modelo ler
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name

        try:
            # Salva a referência na MESMA taxa de saída do pipeline (preserva qualidade).
            # O F5 reamostra internamente para 24kHz para processar.
            sf.write(temp_path, reference_audio_array, int(output_rate))

            # 2. Inferência Zero-Shot
            # O modelo lê o áudio, cruza com o ref_text para mapear as frequências da voz,
            # e sintetiza o texto novo com o mesmo timbre e acústica do ambiente.
            wav_output = self.tts.infer(
                ref_file=temp_path,
                ref_text=ref_text,
                gen_text=text_to_say,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                remove_silence=remove_silence,
                target_rms=target_rms,
                speed=speed,
                show_info=lambda *a, **k: None,
            )

            # Isola o array de áudio (a API pode retornar uma tupla com métricas extras)
            wav_array = wav_output[0] if isinstance(wav_output, tuple) else wav_output
            wav_array = np.asarray(wav_array, dtype=np.float32)

            # 3. Reamostra do nativo do F5 (24kHz) para a taxa do pipeline (na GPU)
            if int(output_rate) == self.f5_rate:
                return wav_array
            tensor_wav = torch.as_tensor(wav_array, dtype=torch.float32, device=self.device).unsqueeze(0)
            resampled = torchaudio.functional.resample(tensor_wav, self.f5_rate, int(output_rate))
            return resampled.squeeze(0).cpu().numpy()

        finally:
            # Limpeza de rastro de memória
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ==========================================
# Teste isolado do módulo
# ==========================================
if __name__ == "__main__":
    cloner = VoiceCloner()
    
    # Simulação do áudio do microfone e da resposta do Whisper
    mock_audio = np.random.normal(0, 0.1, 16000 * 3).astype(np.float32)
    mock_transcription = "I really hate my boss"
    palavra_limpa = "jerk"
    
    print(f"\nExtraindo acústica da frase: '{mock_transcription}'")
    print(f"Gerando áudio da palavra: '{palavra_limpa}'...")
    
    novo_audio = cloner.generate_replacement(
        reference_audio_array=mock_audio, 
        ref_text=mock_transcription,
        text_to_say=palavra_limpa
    )
    
    print("-" * 40)
    print(f"Tipo do dado: {type(novo_audio)}")
    print(f"Amostras geradas: {len(novo_audio)}")
    print(f"Tempo da palavra (em 16kHz): {len(novo_audio) / 16000:.2f} segundos")