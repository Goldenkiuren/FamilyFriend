import numpy as np
import string


class AudioCensor:
    def __init__(self, sample_rate=16000, beep_freq=1000.0, overlap_duration=0.5, beep_margin_start=0.1, beep_margin_end=0.25):
        self.sample_rate = sample_rate
        self.beep_freq = beep_freq
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.beep_margin_start = beep_margin_start
        self.beep_margin_end = beep_margin_end

    # ==========================================================
    # DSP helpers
    # ==========================================================
    @staticmethod
    def _rms(audio):
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))

    def _match_rms(self, segment, target_rms, max_gain=4.0, min_gain=0.25):
        """Casa o nível (loudness) do trecho gerado com o do trecho original substituído."""
        cur = self._rms(segment)
        if cur < 1e-6 or target_rms <= 0:
            return segment
        gain = float(min(max(target_rms / cur, min_gain), max_gain))
        return (segment * gain).astype(np.float32)

    def _trim_edges(self, audio, rel_threshold=0.02):
        """Remove silêncio residual nas bordas do áudio gerado pelo F5."""
        if len(audio) == 0:
            return audio
        peak = np.max(np.abs(audio))
        if peak < 1e-4:
            return audio
        non_silent = np.where(np.abs(audio) > rel_threshold * peak)[0]
        if len(non_silent) == 0:
            return audio
        return audio[non_silent[0]:non_silent[-1] + 1]

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

    def _crossfade(self, a, b, fade_samples):
        """Overlap-add de potência constante entre dois trechos (sem buracos/cliques)."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if len(a) == 0:
            return b.copy()
        if len(b) == 0:
            return a.copy()
        f = int(min(fade_samples, len(a), len(b)))
        if f <= 0:
            return np.concatenate([a, b])
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        fin = np.sin(0.5 * np.pi * ramp)
        fout = np.cos(0.5 * np.pi * ramp)
        mid = a[-f:] * fout + b[:f] * fin
        return np.concatenate([a[:-f], mid, b[f:]]).astype(np.float32)

    def _splice_replace(self, audio, insert, start_idx, end_idx, fade_samples=512):
        """Substitui audio[start:end] por `insert` (comprimento natural do insert),
        com crossfade de potência constante nas duas bordas."""
        before = audio[:start_idx]
        after = audio[end_idx:]
        out = self._crossfade(before, insert, fade_samples)
        out = self._crossfade(out, after, fade_samples)
        return out

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
            window = segment[i: i + window_samples]
            rms = np.sqrt(np.mean(window ** 2))
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
        """Acha o vale acústico (menor energia) limitado pelas paredes das palavras
        vizinhas, para cortar a frase numa pausa real e não no meio de uma palavra."""
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
            window = search_area[i: i + window_samples]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append((rms, i))

        if not rms_values:
            return base_idx

        min_rms, min_i = min(rms_values, key=lambda x: x[0])
        return search_start + min_i + (window_samples // 2)

    # ==========================================================
    # MODO 1: BIP
    # ==========================================================
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
            censored_chunk[final_start_idx:final_start_idx + max_len] = beep[:max_len]

        if not is_first_chunk:
            return censored_chunk[self.overlap_samples:]
        return censored_chunk

    # ==========================================================
    # MODO 2: CLONAGEM (substituição por frase inteira)
    # ==========================================================
    def _build_replacement_phrases(self, words_list, toxic_intervals, pause_threshold=0.5, max_duration=9.0):
        """Agrupa as palavras em frases (cortando em pausas/pontuação). Cada frase que
        contém palavrão vira um trabalho de geração inteiro (mantém intel. e ambiência)."""
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
                next_w = words_list[i + 1]
                gap = next_w["start"] - w["end"]
                clean_word = w["word"].strip()
                punctuation_break = clean_word.endswith(('.', '!', '?'))
                if gap > pause_threshold or punctuation_break or current_duration >= max_duration:
                    break_phrase = True

            if break_phrase:
                if current_toxic_replacements:
                    gen_words = []

                    start_time = current_phrase_words[0][1]["start"]
                    end_time = current_phrase_words[-1][1]["end"]

                    first_idx = current_phrase_words[0][0]
                    last_idx = current_phrase_words[-1][0]

                    prev_end = words_list[first_idx - 1]["end"] if first_idx > 0 else 0.0
                    next_start = words_list[last_idx + 1]["start"] if last_idx < len(words_list) - 1 else end_time + 1.0

                    for idx, word_info in current_phrase_words:
                        clean_w = word_info["word"].strip()
                        if idx in current_toxic_replacements:
                            gen_words.append(current_toxic_replacements[idx])
                        else:
                            gen_words.append(clean_w)

                    phrases_to_replace.append({
                        "first_idx": first_idx,
                        "last_idx": last_idx,
                        "start": start_time,
                        "end": end_time,
                        "prev_end": prev_end,
                        "next_start": next_start,
                        "gen_text": " ".join(gen_words).strip(),
                    })

                current_phrase_words = []
                current_toxic_replacements = {}

        return phrases_to_replace

    def _build_long_reference(self, full_audio, words_list, first_idx, last_idx,
                              min_sec=3.0, max_sec=10.0, turn_gap=0.8):
        """Monta uma referência de voz LONGA (>= min_sec) para o F5 clonar bem.
        Parte da própria frase e expande para os vizinhos do MESMO turno de fala
        (pausas <= turn_gap), o que evita pegar o outro locutor sem diarização."""
        n = len(words_list)
        lo, hi = first_idx, last_idx

        def span_sec():
            return words_list[hi]["end"] - words_list[lo]["start"]

        # Expande alternadamente para os lados enquanto for curto e mesmo turno
        while span_sec() < min_sec and span_sec() < max_sec:
            grew = False
            if lo > 0 and (words_list[lo]["start"] - words_list[lo - 1]["end"]) <= turn_gap:
                lo -= 1
                grew = True
            if span_sec() < min_sec and hi < n - 1 and (words_list[hi + 1]["start"] - words_list[hi]["end"]) <= turn_gap:
                hi += 1
                grew = True
            if not grew:
                break

        s = int(words_list[lo]["start"] * self.sample_rate)
        e = int(words_list[hi]["end"] * self.sample_rate)
        ref_audio = full_audio[max(0, s):min(len(full_audio), e)]
        ref_text = " ".join(words_list[i]["word"].strip() for i in range(lo, hi + 1)).strip()
        return ref_audio, ref_text

    def process_offline_replacement(self, full_audio, toxic_intervals, words_list, voice_cloner,
                                    whisper_model=None):
        """Substitui cada FRASE com palavrão por uma versão regerada pelo F5 (frase inteira,
        contínua => intel. e ambiência consistentes), usando uma referência de voz longa do
        mesmo locutor. Boundaries em vales de pausa; nível casado; crossfade equal-power."""
        print("🔧 [DSP] Modo Clone: regeração por frase com referência longa do mesmo locutor...")
        final_audio = full_audio.copy()
        sr = self.sample_rate

        phrases = self._build_replacement_phrases(words_list, toxic_intervals)

        for phrase in reversed(phrases):
            start_idx = self._find_safe_valley(
                full_audio, whisper_sec=phrase["start"], limit_sec=phrase["prev_end"], is_start_cut=True
            )
            end_idx = self._find_safe_valley(
                full_audio, whisper_sec=phrase["end"], limit_sec=phrase["next_start"], is_start_cut=False
            )
            start_idx = max(0, min(start_idx, len(final_audio)))
            end_idx = max(start_idx, min(end_idx, len(final_audio)))
            if end_idx - start_idx <= 0:
                continue

            ref_audio, ref_text = self._build_long_reference(
                full_audio, words_list, phrase["first_idx"], phrase["last_idx"]
            )
            gen_text = phrase["gen_text"]
            ref_sec = len(ref_audio) / sr

            print(f"\n🎙️ Clonando frase -> '{gen_text}'  (ref {ref_sec:.1f}s: '{ref_text[:60]}...')")
            if ref_sec < 1.0:
                print("   ⚠️ Referência curta (<1s): o turno é curto; a clonagem pode sair fraca.")

            try:
                generated = voice_cloner.generate_replacement(
                    reference_audio_array=ref_audio,
                    ref_text=ref_text,
                    text_to_say=gen_text,
                )
                generated = self._trim_edges(generated)
                if len(generated) < int(0.05 * sr):
                    raise ValueError("saída do F5 vazia/curta demais")

                # Nível: casa com o loudness do trecho original que está sendo substituído
                target_rms = self._rms(full_audio[start_idx:end_idx])
                generated = self._match_rms(generated, target_rms)

                final_audio = self._splice_replace(
                    final_audio, generated, start_idx, end_idx, fade_samples=512
                )

            except Exception as e:
                print(f"⚠️ Erro ao clonar (caindo para bip): {e}")
                beep = self._generate_beep((end_idx - start_idx) / sr)
                final_audio = self._splice_replace(
                    final_audio, beep, start_idx, end_idx, fade_samples=512
                )

        # Limitador de segurança
        peak = float(np.max(np.abs(final_audio))) if len(final_audio) else 0.0
        if peak > 0.99:
            final_audio = (final_audio * (0.99 / peak)).astype(np.float32)

        return final_audio
