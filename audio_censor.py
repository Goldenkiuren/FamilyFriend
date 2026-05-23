import numpy as np

class AudioCensor:
    def __init__(self, sample_rate=16000, beep_freq=1000.0, overlap_duration=0.5):
        """
        Inicializa o processador de censura de áudio.
        beep_freq: 1000Hz é a frequência padrão do bip de censura da TV.
        overlap_duration: Tempo que precisará ser cortado do início do chunk final.
        """
        self.sample_rate = sample_rate
        self.beep_freq = beep_freq
        self.overlap_samples = int(overlap_duration * sample_rate)

    def _generate_beep(self, duration):
        """
        Gera a onda senoidal do bip.
        Inclui um micro fade-in e fade-out (10ms) nas pontas do bip para evitar
        o efeito de 'clipping' (estalos secos) quando o áudio da voz é cortado abruptamente.
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # Amplitude de 0.5 para o bip não estourar os alto-falantes do usuário
        beep = 0.5 * np.sin(self.beep_freq * t * 2 * np.pi)
        
        # Suavização de bordas (Envelope)
        fade_duration = int(self.sample_rate * 0.01) # 10 milissegundos
        if len(beep) > 2 * fade_duration:
            fade_in = np.linspace(0, 1, fade_duration)
            fade_out = np.linspace(1, 0, fade_duration)
            beep[:fade_duration] *= fade_in
            beep[-fade_duration:] *= fade_out
            
        return beep.astype(np.float32)

    def process_chunk(self, audio_chunk, toxic_intervals):
        """
        Recebe o array do chunk inteiro (3.0s) e os tempos ofensivos.
        Substitui as ofensas por bips e DEPOIS corta o overlap, conforme sua especificação.
        """
        # Trabalhamos em uma cópia para não alterar o buffer original acidentalmente
        censored_chunk = audio_chunk.copy()
        chunk_duration = len(audio_chunk) / self.sample_rate
        
        # 1. Aplica a censura nos tempos exatos
        for interval in toxic_intervals:
            # Garante que os tempos não ultrapassem os limites físicos do chunk
            start_time = max(0.0, interval["start"])
            end_time = min(chunk_duration, interval["end"])
            
            if start_time >= end_time:
                continue
                
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            duration = end_time - start_time
            
            beep = self._generate_beep(duration)
            
            # Ajuste de tamanho para evitar erro de indexação por arredondamento
            max_len = min(len(beep), end_idx - start_idx)
            censored_chunk[start_idx:start_idx+max_len] = beep[:max_len]
            
        # 2. Corta o Overlap
        # Como especificado: "antes de usar o chunk tem que cortar a parte de overlap"
        # Isso garante que a parte final do bip do chunk anterior não seja tocada duas vezes
        final_output = censored_chunk[self.overlap_samples:]
        
        return final_output

# ==========================================
# Teste isolado do módulo
# ==========================================
if __name__ == "__main__":
    # Cria 3 segundos de silêncio para simular o chunk (16000 amostras * 3s = 48000)
    print("Módulo AudioCensor: Teste de Manipulação de Array")
    mock_audio = np.zeros(48000, dtype=np.float32)
    
    # Simula a resposta do modelo ToxicRoBERTa que fizemos no passo anterior
    mock_intervals = [
        {"word": "fucking", "start": 0.5, "end": 0.9},
        {"word": "idiot", "start": 0.9, "end": 1.3}
    ]
    
    censor = AudioCensor(sample_rate=16000, overlap_duration=0.5)
    
    # Processa o áudio
    final_audio = censor.process_chunk(mock_audio, mock_intervals)
    
    # Validações matemáticas baseadas na sua especificação
    expected_samples = 48000 - (16000 * 0.5) # Chunk (3s) - Overlap (0.5s) = 2.5s finais
    
    print("-" * 40)
    print(f"Tamanho do Chunk Original: {len(mock_audio)} amostras")
    print(f"Tamanho do Chunk Final:    {len(final_audio)} amostras")
    print(f"Bip aplicado com sucesso?  {'Sim' if np.any(final_audio != 0) else 'Não'}")
    
    if len(final_audio) == expected_samples:
        print("✅ Overlap cortado corretamente (Restaram 2.5s para o Consumidor).")
    else:
        print("❌ Erro no cálculo de corte do overlap.")