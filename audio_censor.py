import numpy as np
import string

class AudioCensor:
    def __init__(self, sample_rate=16000, beep_freq=1000.0, overlap_duration=0.5, beep_margin_start=0.1, beep_margin_end=0.25):
        self.sample_rate = sample_rate
        self.beep_freq = beep_freq
        self.overlap_samples = int(overlap_duration * sample_rate)
        # Margens para o Bip Clássico (Ao Vivo)
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
        """
        Crossfade mais suave e longo (1024 amostras = ~64ms em 16kHz) 
        ideal para cortes feitos em momentos de pausa/silêncio.
        """
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

    def process_chunk(self, audio_chunk, toxic_intervals, is_first_chunk=False):
        """Aplicado no áudio Ao Vivo (Streaming) - Usa Bip clássico."""
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
            
        if not is_first_chunk:
            return censored_chunk[self.overlap_samples:]
        return censored_chunk

    def _build_replacement_phrases(self, words_list, toxic_intervals, pause_threshold=0.5):
        """
        Agrupa palavras em 'blocos de respiração' (frases).
        Se a frase contiver palavras tóxicas, prepara o bloco para ser recriado.
        """
        phrases_to_replace = []
        if not words_list or not toxic_intervals:
            return phrases_to_replace

        current_phrase_words = []
        current_toxic_replacements = {} 

        # Mapeia palavras tóxicas por tempo de início para busca rápida
        toxic_map = {round(t["start"], 2): t for t in toxic_intervals}

        for i, w in enumerate(words_list):
            start_time_rounded = round(w["start"], 2)
            is_toxic = start_time_rounded in toxic_map

            if is_toxic:
                current_toxic_replacements[i] = toxic_map[start_time_rounded].get("replacement", "bleep")

            current_phrase_words.append((i, w))

            is_last_word = (i == len(words_list) - 1)
            break_phrase = is_last_word

            # Lógica de quebra de frase: Silêncio ou Pontuação forte
            if not is_last_word:
                next_w = words_list[i+1]
                gap = next_w["start"] - w["end"]
                
                clean_word = w["word"].strip()
                punctuation_break = clean_word.endswith(('.', '!', '?'))
                
                if gap > pause_threshold or punctuation_break:
                    break_phrase = True

            # Se a frase acabou, validamos se precisa de censura
            if break_phrase:
                if current_toxic_replacements:
                    ref_words = []
                    gen_words = []
                    
                    start_time = current_phrase_words[0][1]["start"]
                    end_time = current_phrase_words[-1][1]["end"]

                    for idx, word_info in current_phrase_words:
                        clean_w = word_info["word"].strip()
                        ref_words.append(clean_w)
                        
                        if idx in current_toxic_replacements:
                            gen_words.append(current_toxic_replacements[idx])
                        else:
                            gen_words.append(clean_w)

                    # Remove pontuações que o Whisper traz coladas nas palavras para não quebrar o prompt
                    ref_text = " ".join(ref_words).strip()
                    gen_text = " ".join(gen_words).strip()

                    phrases_to_replace.append({
                        "start": start_time,
                        "end": end_time,
                        "ref_text": ref_text,
                        "gen_text": gen_text
                    })

                # Reseta os buffers para a próxima frase
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

        # Passo 1: Converter palavras isoladas em blocos de contexto (Frases)
        phrases_to_replace = self._build_replacement_phrases(
            words_list=words_list, 
            toxic_intervals=toxic_intervals,
            pause_threshold=0.5 # Corta se o silêncio for maior que 0.5s
        )

        for phrase in phrases_to_replace:
            # Pegamos um pequeno respiro ao redor da frase (0.1s)
            orig_start_time = max(0.0, phrase["start"] - 0.1)
            orig_end_time = phrase["end"] + 0.1
            
            orig_start_idx = int(orig_start_time * self.sample_rate)
            orig_end_idx = int(orig_end_time * self.sample_rate)
            
            start_idx = orig_start_idx + accumulated_offset
            end_idx = orig_end_idx + accumulated_offset
            
            # Travas de segurança
            start_idx = max(0, min(start_idx, len(final_audio)))
            end_idx = max(0, min(end_idx, len(final_audio)))

            reference_audio = full_audio[orig_start_idx:orig_end_idx]
            ref_text = phrase["ref_text"]
            gen_text = phrase["gen_text"]

            print(f"\n🎙️ Clonando frase inteira:")
            print(f"   Original: '{ref_text}'")
            print(f"   Censurado: '{gen_text}'")
            
            try:
                # O F5-TTS recria a frase com a prosódia completa!
                generated_phrase_array = voice_cloner.generate_replacement(
                    reference_audio_array=reference_audio,
                    ref_text=ref_text,
                    text_to_say=gen_text
                )
                
                processed_phrase = self._trim_and_normalize(generated_phrase_array, reference_audio)
                
                # Crossfade ocorre no silêncio/respiração da frase, não no meio de uma vogal
                final_audio, size_diff = self._insert_with_crossfade(
                    original_audio=final_audio, 
                    new_audio=processed_phrase, 
                    start_idx=start_idx, 
                    end_idx=end_idx,
                    fade_samples=1024  # ~64ms fade, bem macio
                )
                
                accumulated_offset += size_diff
                print(f"✅ Frase substituída. (Deslocamento: {size_diff} amostras).")
                
            except Exception as e:
                print(f"⚠️ Erro ao clonar frase '{ref_text}'. Aplicando Bip de fallback. Erro: {e}")
                duration = orig_end_time - orig_start_time
                beep = self._generate_beep(duration)
                final_audio, size_diff = self._insert_with_crossfade(final_audio, beep, start_idx, end_idx)
                accumulated_offset += size_diff

        return final_audio