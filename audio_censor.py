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
        before = original_audio[:start_idx].copy()
        after = original_audio[end_idx:].copy()
        
        if len(new_audio) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
            new_audio[:fade_samples] *= fade_in
            new_audio[-fade_samples:] *= fade_out
            
        if len(before) > fade_samples:
            before[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)
        if len(after) > fade_samples:
            after[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)

        result = np.concatenate([before, new_audio, after])
        size_diff = len(new_audio) - (end_idx - start_idx)
        return result, size_diff

    def _find_dynamic_silence(self, audio_array, start_idx, direction="forward", max_search_sec=1.0, window_ms=20, threshold=0.015):
        # Proteção contra busca nula (quando palavras estão coladas)
        if max_search_sec <= 0.01:
            return start_idx

        max_search_samples = int(max_search_sec * self.sample_rate)
        window_samples = int((window_ms / 1000.0) * self.sample_rate)
        
        if direction == "forward":
            search_area = audio_array[start_idx : start_idx + max_search_samples]
        else: 
            search_area = audio_array[max(0, start_idx - max_search_samples) : start_idx]
            search_area = search_area[::-1] 
            
        if len(search_area) < window_samples:
            return start_idx
            
        for i in range(0, len(search_area) - window_samples, window_samples):
            window = search_area[i : i + window_samples]
            rms = np.sqrt(np.mean(window**2))
            
            if rms < threshold:
                offset = i + (window_samples // 2)
                return start_idx + offset if direction == "forward" else start_idx - offset
                
        return start_idx + max_search_samples if direction == "forward" else max(0, start_idx - max_search_samples)

    def _get_ultra_precise_boundaries(self, audio_array, whisper_start_idx, whisper_end_idx, search_margin_ms=150, window_ms=10):
        """
        Expande a busca para fora dos limites do Whisper e usa uma combinação 
        de Volume (RMS) e Frequência (ZCR) para achar as bordas exatas da palavra.
        """
        # 1. Expandimos a "caixa" do Whisper em 150ms para trás e para frente
        margin_samples = int((search_margin_ms / 1000.0) * self.sample_rate)
        start_search = max(0, whisper_start_idx - margin_samples)
        end_search = min(len(audio_array), whisper_end_idx + margin_samples)

        segment = audio_array[start_search:end_search]
        window_samples = int((window_ms / 1000.0) * self.sample_rate)

        if len(segment) < window_samples:
            return whisper_start_idx, whisper_end_idx

        rms_values = []
        zcr_values = []

        # 2. Varredura Acústica
        for i in range(0, len(segment) - window_samples, window_samples):
            window = segment[i : i + window_samples]
            
            # Volume (RMS) - Capta o núcleo da palavra (vogais)
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
            
            # Frequência (ZCR) - Capta as consoantes secas (s, sh, f, k)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(window))))
            zcr_values.append(zero_crossings)

        # 3. Normalização (coloca Volume e Frequência na mesma escala de 0 a 1)
        rms_norm = np.array(rms_values) / (np.max(rms_values) + 1e-6)
        zcr_norm = np.array(zcr_values) / (np.max(zcr_values) + 1e-6)

        # 4. Atividade Combinada (O ZCR tem um peso de 30% para não ser eng Enganado por chiado de fundo)
        activity = rms_norm + (zcr_norm * 0.3)

        # 5. Threshold dinâmico: o som começa quando a atividade atinge 4% do pico máximo daquela região
        threshold = np.max(activity) * 0.04
        valid_indices = [i for i, act in enumerate(activity) if act > threshold]

        if not valid_indices:
            return whisper_start_idx, whisper_end_idx 

        # 6. Mapeamento de volta para os índices originais do áudio completo
        tight_start = start_search + (valid_indices[0] * window_samples)
        tight_end = start_search + (valid_indices[-1] * window_samples) + window_samples

        return tight_start, tight_end
    
    def process_chunk(self, audio_chunk, toxic_intervals, is_first_chunk=False):
        censored_chunk = audio_chunk.copy()
        chunk_duration = len(audio_chunk) / self.sample_rate
        
        for interval in toxic_intervals:
            raw_start_idx = int(max(0.0, interval["start"]) * self.sample_rate)
            raw_end_idx = int(min(chunk_duration, interval["end"]) * self.sample_rate)
            
            # Chama a análise combinada de RMS e ZCR
            tight_start_idx, tight_end_idx = self._get_ultra_precise_boundaries(
                audio_chunk, raw_start_idx, raw_end_idx
            )
            
            # A micro margem agora pode ser ainda menor, apenas 10ms (Crossfade suave)
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
        """
        Agrupa palavras em 'blocos de respiração' (frases).
        FORÇA a quebra se a frase atingir o max_duration (9 segundos) para evitar o erro 'clipping short' do F5-TTS.
        """
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
            
            # Calcula o tamanho da frase atual em segundos
            current_duration = w["end"] - current_phrase_words[0][1]["start"]

            is_last_word = (i == len(words_list) - 1)
            break_phrase = is_last_word

            if not is_last_word:
                next_w = words_list[i+1]
                gap = next_w["start"] - w["end"]
                
                clean_word = w["word"].strip()
                punctuation_break = clean_word.endswith(('.', '!', '?'))
                
                # A NOVA TRAVA: Se passar de 9 segundos, forçamos o corte imediatamente!
                if gap > pause_threshold or punctuation_break or current_duration >= max_duration:
                    break_phrase = True

            if break_phrase:
                if current_toxic_replacements:
                    ref_words = []
                    gen_words = []
                    
                    start_time = current_phrase_words[0][1]["start"]
                    end_time = current_phrase_words[-1][1]["end"]

                    # Caixa de contenção segura (Safe Bounding Box)
                    first_word_idx = current_phrase_words[0][0]
                    last_word_idx = current_phrase_words[-1][0]

                    safe_start = max(0.0, start_time - 0.15)
                    safe_end = end_time + 0.25
                    
                    if first_word_idx > 0:
                        prev_word_end = words_list[first_word_idx - 1]["end"]
                        safe_start = max(prev_word_end, safe_start)

                    if last_word_idx < len(words_list) - 1:
                        next_word_start = words_list[last_word_idx + 1]["start"]
                        safe_end = min(next_word_start, safe_end)

                    for idx, word_info in current_phrase_words:
                        clean_w = word_info["word"].strip()
                        ref_words.append(clean_w)
                        
                        if idx in current_toxic_replacements:
                            gen_words.append(current_toxic_replacements[idx])
                        else:
                            gen_words.append(clean_w)

                    ref_text = " ".join(ref_words).strip()
                    gen_text = " ".join(gen_words).strip()

                    phrases_to_replace.append({
                        "start": start_time,
                        "end": end_time,
                        "safe_start": safe_start,
                        "safe_end": safe_end,
                        "ref_text": ref_text,
                        "gen_text": gen_text
                    })

                current_phrase_words = []
                current_toxic_replacements = {}

        return phrases_to_replace

    def process_offline_replacement(self, full_audio, toxic_intervals, words_list, voice_cloner):
        print("🔧 [DSP] Iniciando reconstrução de áudio baseada em Nível de Frase...")
        full_original_text = " ".join([w["word"].strip() for w in words_list])
        print("="*60)
        print(f"📜 [TRANSCRIÇÃO ORIGINAL COMPLETA]:\n{full_original_text}")
        print("="*60)
        final_audio = full_audio.copy()
        accumulated_offset = 0

        phrases_to_replace = self._build_replacement_phrases(
            words_list=words_list, 
            toxic_intervals=toxic_intervals,
            pause_threshold=0.5
        )

        for phrase in phrases_to_replace:
            whisper_start_idx = int(phrase["start"] * self.sample_rate)
            whisper_end_idx = int(phrase["end"] * self.sample_rate)
            
            # Forçamos a busca do silêncio a obedecer à caixa de segurança
            max_back_sec = max(0.0, phrase["start"] - phrase["safe_start"])
            max_fwd_sec = max(0.0, phrase["safe_end"] - phrase["end"])
            
            orig_start_idx = self._find_dynamic_silence(
                full_audio, whisper_start_idx, direction="backward", max_search_sec=max_back_sec
            )
            orig_end_idx = self._find_dynamic_silence(
                full_audio, whisper_end_idx, direction="forward", max_search_sec=max_fwd_sec
            )
            
            start_idx = orig_start_idx + accumulated_offset
            end_idx = orig_end_idx + accumulated_offset
            
            start_idx = max(0, min(start_idx, len(final_audio)))
            end_idx = max(0, min(end_idx, len(final_audio)))

            reference_audio = full_audio[orig_start_idx:orig_end_idx]
            ref_text = phrase["ref_text"]
            gen_text = phrase["gen_text"]

            print(f"\n🎙️ Clonando frase inteira:")
            print(f"   Original: '{ref_text}'")
            print(f"   Censurado: '{gen_text}'")
            
            try:
                generated_phrase_array = voice_cloner.generate_replacement(
                    reference_audio_array=reference_audio,
                    ref_text=ref_text,
                    text_to_say=gen_text
                )
                
                processed_phrase = self._trim_and_normalize(generated_phrase_array, reference_audio)
                
                final_audio, size_diff = self._insert_with_crossfade(
                    original_audio=final_audio, 
                    new_audio=processed_phrase, 
                    start_idx=start_idx, 
                    end_idx=end_idx,
                    fade_samples=1024
                )
                
                accumulated_offset += size_diff
                print(f"✅ Frase substituída. (Deslocamento: {size_diff} amostras).")
                
            except Exception as e:
                print(f"⚠️ Erro ao clonar frase '{ref_text}'. Aplicando Bip de fallback. Erro: {e}")
                
                duration = (orig_end_idx - orig_start_idx) / self.sample_rate
                beep = self._generate_beep(duration)
                final_audio, size_diff = self._insert_with_crossfade(final_audio, beep, start_idx, end_idx)
                accumulated_offset += size_diff

        return final_audio