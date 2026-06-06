import numpy as np

class AudioCensor:
    def __init__(self, sample_rate=16000, beep_freq=1000.0, beep_margin_start=0.1, beep_margin_end=0.25):
        """
        beep_margin_start: 100ms antes da palavra.
        beep_margin_end: 250ms depois da palavra, para cobrir o "rastro" da pronúncia.
        """
        self.sample_rate = sample_rate
        self.beep_freq = beep_freq
        self.beep_margin_start = beep_margin_start
        self.beep_margin_end = beep_margin_end

    def _generate_beep(self, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        beep = 0.5 * np.sin(self.beep_freq * t * 2 * np.pi)
        
        # Fade in/out suave para não estourar o alto-falante (clipping)
        fade_duration = int(self.sample_rate * 0.01)
        if len(beep) > 2 * fade_duration:
            fade_in = np.linspace(0, 1, fade_duration)
            fade_out = np.linspace(1, 0, fade_duration)
            beep[:fade_duration] *= fade_in
            beep[-fade_duration:] *= fade_out
            
        return beep.astype(np.float32)

    def process_chunk(self, audio_chunk, toxic_intervals):
        censored_chunk = audio_chunk.copy()
        chunk_duration = len(audio_chunk) / self.sample_rate
        
        for interval in toxic_intervals:
            start_time = max(0.0, interval["start"] - self.beep_margin_start)
            end_time = min(chunk_duration, interval["end"] + self.beep_margin_end)
            
            if start_time >= end_time:
                continue
                
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            duration = end_time - start_time
            
            beep = self._generate_beep(duration)
            
            max_len = min(len(beep), end_idx - start_idx)
            censored_chunk[start_idx:start_idx+max_len] = beep[:max_len]
            
        return censored_chunk