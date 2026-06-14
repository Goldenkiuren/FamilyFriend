import numpy as np
import string

class AudioCensor:
    def __init__(self, sample_rate=16000, beep_freq=1000.0, overlap_duration=0.5, beep_margin_start=0.1, beep_margin_end=0.25):
        self.sample_rate = sample_rate
        self.beep_freq = beep_freq
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.beep_margin_start = beep_margin_start
        self.beep_margin_end = beep_margin_end

    def _generate_beep(self, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        beep = 0.5 * np.sin(self.beep_freq * t * 2 * np.pi)
        
        fade_duration = int(self.sample_rate * 0.01)
        if len(beep) > 2 * fade_duration:
            fade_in = np.linspace(0, 1, fade_duration)
            fade_out = np.linspace(1, 0, fade_duration)
            beep[:fade_duration] *= fade_in
            beep[-fade_duration:] *= fade_out
            
        return beep.astype(np.float32)

    def _trim_and_normalize(self, generated_audio, reference_audio):
        max_amp = np.max(np.abs(generated_audio))
        if max_amp < 1e-4:
            return generated_audio
            
        threshold = 0.02 * max_amp
        non_silent = np.where(np.abs(generated_audio) > threshold)[0]
        
        if len(non_silent) > 0:
            trimmed_audio = generated_audio[non_silent[0]:non_silent[-1]+1]
        else:
            trimmed_audio = generated_audio

        ref_max = np.max(np.abs(reference_audio))
        ref_max = max(ref_max, 0.3) 
        
        gen_max = np.max(np.abs(trimmed_audio))
        if gen_max > 0:
            trimmed_audio = (trimmed_audio / gen_max) * ref_max
            
        return trimmed_audio

    def _insert_with_crossfade(self, original_audio, new_audio, start_idx, end_idx, fade_samples=1024):
        """ Crossfade de Potência Constante (Equal Power) para evitar 'pulos' de volume """
        before = original_audio[:start_idx].copy()
        after = original_audio[end_idx:].copy()
        
        linear = np.linspace(0, 1, fade_samples, dtype=np.float32)
        
        if len(new_audio) > fade_samples * 2:
            fade_in = np.sin((np.pi / 2) * linear)
            fade_out = np.cos((np.pi / 2) * linear)
            new_audio[:fade_samples] *= fade_in
            new_audio[-fade_samples:] *= fade_out
            
        if len(before) > fade_samples:
            before[-fade_samples:] *= np.cos((np.pi / 2) * linear)
        if len(after) > fade_samples:
            after[:fade_samples] *= np.sin((np.pi / 2) * linear)

        result = np.concatenate([before, new_audio, after])
        size_diff = len(new_audio) - (end_idx - start_idx)
        return result, size_diff

    def _get_ultra_precise_boundaries(self, audio_array, whisper_start_idx, whisper_end_idx, search_margin_ms=150, window_ms=10):
        """ Usado no Modo Bip: RMS + ZCR para envolver cirurgicamente apenas a palavra """
        margin_samples = int((search_margin_ms / 1000.0) * self.sample_rate)
        start_search = max(0, whisper_start_idx - margin_samples)
        end_search = min(len(audio_array), whisper_end_idx + margin_samples)

        segment = audio_array[start_search:end_search]
        window_samples = int((window_ms / 1000.0) * self.sample_rate)

        if len(segment) < window_samples:
            return whisper_start_idx, whisper_end_idx

        rms_values, zcr_values = [], []

        for i in range(0, len(segment) - window_samples, window_samples):
            window = segment[i : i + window_samples]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(window))))
            zcr_values.append(zero_crossings)

        rms_norm = np.array(rms_values) / (np.max(rms_values) + 1e-6)
        zcr_norm = np.array(zcr_values) / (np.max(zcr_values) + 1e-6)

        activity = rms_norm + (zcr_norm * 0.3)
        threshold = np.max(activity) * 0.04
        valid_indices = [i for i, act in enumerate(activity) if act > threshold]

        if not valid_indices:
            return whisper_start_idx, whisper_end_idx 

        tight_start = start_search + (valid_indices[0] * window_samples)
        tight_end = start_search + (valid_indices[-1] * window_samples) + window_samples

        return tight_start, tight_end

    def _find_safe_valley(self, audio_array, whisper_sec, limit_sec, is_start_cut=True, window_ms=5):
        """ 
        Usado no Modo Clonagem: Acha o vale acústico estritamente limitado pelas 
        paredes das palavras vizinhas para impedir amputações e efeito gagueira.
        """
        base_idx = int(whisper_sec * self.sample_rate)
        limit_idx = int(limit_sec * self.sample_rate)

        margin_samples = int(0.15 * self.sample_rate)
        
        if is_start_cut:
            search_start = max(limit_idx, base_idx - margin_samples)
            search_end = min(len(audio_array), base_idx + int(0.05 * self.sample_rate))
        else:
            search_start = max(0, base_idx - int(0.05 * self.sample_rate))
            search_end = min(limit_idx, base_idx + margin_samples)
            
        if search_start >= search_end:
            return base_idx
            
        search_area = audio_array[search_start:search_end]
        window_samples = int((window_ms / 1000.0) * self.sample_rate)
        
        if len(search_area) < window_samples:
            return base_idx
            
        rms_values = []
        step = max(1, window_samples // 2)
        for i in range(0, len(search_area) - window_samples, step):
            window = search_area[i : i + window_samples]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append((rms, i))
            
        if not rms_values:
            return base_idx
            
        min_rms, min_i = min(rms_values, key=lambda x: x[0])
        return search_start + min_i + (window_samples // 2)
    
    def process_chunk(self, audio_chunk, toxic_intervals, is_first_chunk=False):
        """ Motor de Censura Pontual (Bip) """
        censored_chunk = audio_chunk.copy()
        chunk_duration = len(audio_chunk) / self.sample_rate
        
        for interval in toxic_intervals:
            raw_start_idx = int(max(0.0, interval["start"]) * self.sample_rate)
            raw_end_idx = int(min(chunk_duration, interval["end"]) * self.sample_rate)
            
            tight_start_idx, tight_end_idx = self._get_ultra_precise_boundaries(
                audio_chunk, raw_start_idx, raw_end_idx
            )
            
            micro_margin = int(0.01 * self.sample_rate)
            final_start_idx = max(0, tight_start_idx - micro_margin)
            final_end_idx = min(len(censored_chunk), tight_end_idx + micro_margin)
            
            duration = (final_end_idx - final_start_idx) / self.sample_rate
            
            if duration <= 0:
                continue
                
            beep = self._generate_beep(duration)
            max_len = min(len(beep), final_end_idx - final_start_idx)
            censored_chunk[final_start_idx:final_start_idx+max_len] = beep[:max_len]
            
        if not is_first_chunk:
            return censored_chunk[self.overlap_samples:]
        return censored_chunk

    def _build_replacement_phrases(self, words_list, toxic_intervals, pause_threshold=0.5, max_duration=9.0):
        """ Motor de Agrupamento Lógico para a IA (Extrai as Paredes Vizinhas) """
        phrases_to_replace = []
        if not words_list or not toxic_intervals:
            return phrases_to_replace

        current_phrase_words = []
        current_toxic_replacements = {} 
        toxic_map = {round(t["start"], 2): t for t in toxic_intervals}

        for i, w in enumerate(words_list):
            start_time_rounded = round(w["start"], 2)
            is_toxic = start_time_rounded in toxic_map

            if is_toxic:
                current_toxic_replacements[i] = toxic_map[start_time_rounded].get("replacement", "bleep")

            current_phrase_words.append((i, w))
            current_duration = w["end"] - current_phrase_words[0][1]["start"]

            is_last_word = (i == len(words_list) - 1)
            break_phrase = is_last_word

            if not is_last_word:
                next_w = words_list[i+1]
                gap = next_w["start"] - w["end"]
                clean_word = w["word"].strip()
                punctuation_break = clean_word.endswith(('.', '!', '?'))
                if gap > pause_threshold or punctuation_break or current_duration >= max_duration:
                    break_phrase = True

            if break_phrase:
                if current_toxic_replacements:
                    ref_words = []
                    gen_words = []
                    
                    start_time = current_phrase_words[0][1]["start"]
                    end_time = current_phrase_words[-1][1]["end"]

                    first_idx = current_phrase_words[0][0]
                    last_idx = current_phrase_words[-1][0]

                    prev_end = words_list[first_idx - 1]["end"] if first_idx > 0 else 0.0
                    next_start = words_list[last_idx + 1]["start"] if last_idx < len(words_list) - 1 else end_time + 1.0

                    for idx, word_info in current_phrase_words:
                        clean_w = word_info["word"].strip()
                        ref_words.append(clean_w)
                        if idx in current_toxic_replacements:
                            gen_words.append(current_toxic_replacements[idx])
                        else:
                            gen_words.append(clean_w)

                    phrases_to_replace.append({
                        "start": start_time,
                        "end": end_time,
                        "prev_end": prev_end,         
                        "next_start": next_start,     
                        "ref_text": " ".join(ref_words).strip(),
                        "gen_text": " ".join(gen_words).strip()
                    })

                current_phrase_words = []
                current_toxic_replacements = {}

        return phrases_to_replace

    def process_offline_replacement(self, full_audio, toxic_intervals, words_list, voice_cloner):
        """ Motor Principal do Modo Clone (Processamento Reverso e Edição) """
        print("🔧 [DSP] Iniciando reconstrução com busca de Vale Acústico Seguro...")
        final_audio = full_audio.copy()

        phrases_to_replace = self._build_replacement_phrases(
            words_list=words_list, 
            toxic_intervals=toxic_intervals
        )

        for phrase in reversed(phrases_to_replace):
            start_idx = self._find_safe_valley(
                full_audio, whisper_sec=phrase["start"], limit_sec=phrase["prev_end"], is_start_cut=True
            )
            end_idx = self._find_safe_valley(
                full_audio, whisper_sec=phrase["end"], limit_sec=phrase["next_start"], is_start_cut=False
            )
            
            start_idx = max(0, min(start_idx, len(final_audio)))
            end_idx = max(0, min(end_idx, len(final_audio)))

            reference_audio = full_audio[start_idx:end_idx]
            ref_text = phrase["ref_text"]
            gen_text = phrase["gen_text"]

            print(f"\n🎙️ Clonando: Original: '{ref_text}' -> Gen: '{gen_text}'")
            
            try:
                generated_phrase_array = voice_cloner.generate_replacement(
                    reference_audio_array=reference_audio,
                    ref_text=ref_text,
                    text_to_say=gen_text
                )
                
                processed_phrase = self._trim_and_normalize(generated_phrase_array, reference_audio)
                
                final_audio, _ = self._insert_with_crossfade(
                    original_audio=final_audio, 
                    new_audio=processed_phrase, 
                    start_idx=start_idx, 
                    end_idx=end_idx,
                    fade_samples=1024
                )
                
            except Exception as e:
                print(f"⚠️ Erro ao clonar: {e}")
                duration = (end_idx - start_idx) / self.sample_rate
                beep = self._generate_beep(duration)
                final_audio, _ = self._insert_with_crossfade(
                    original_audio=final_audio, 
                    new_audio=beep, 
                    start_idx=start_idx, 
                    end_idx=end_idx,
                    fade_samples=1024
                )

        return final_audio