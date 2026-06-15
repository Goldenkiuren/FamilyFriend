import numpy as np


class AudioCensor:
    def __init__(self, sample_rate=16000, beep_freq=1000.0, overlap_duration=0.5, beep_margin_start=0.1, beep_margin_end=0.25):
        self.sample_rate = sample_rate
        self.beep_freq = beep_freq
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.beep_margin_start = beep_margin_start
        self.beep_margin_end = beep_margin_end


    @staticmethod
    def group_into_phrases(words_list, pause_threshold=0.5, max_duration=9.0):
        """Agrupa a lista de palavras do Whisper em blocos coerentes (frases puras)."""
        phrases = []
        if not words_list:
            return phrases

        current_phrase = []
        for i, w in enumerate(words_list):
            current_phrase.append(w)
            current_duration = w["end"] - current_phrase[0]["start"]
            
            is_last = (i == len(words_list) - 1)
            break_phrase = is_last
            
            if not is_last:
                next_w = words_list[i + 1]
                gap = next_w["start"] - w["end"]
                clean_word = w["word"].strip()
                punctuation_break = clean_word.endswith(('.', '!', '?'))
                if gap > pause_threshold or punctuation_break or current_duration >= max_duration:
                    break_phrase = True

            if break_phrase:
                phrases.append(current_phrase)
                current_phrase = []
        return phrases

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

    def _trim_edges(self, audio, lead_thr=0.02, tail_thr=0.006, margin_ms=35):
        """Remove silêncio nas bordas do áudio gerado, mas com limiar mais BAIXO no fim
        (+ margem) para NÃO comer sílabas finais fracas (ex.: '-er' de 'Hatcher')."""
        if len(audio) == 0:
            return audio
        a = np.abs(audio)
        peak = float(np.max(a))
        if peak < 1e-4:
            return audio
        lead = np.where(a > lead_thr * peak)[0]
        tail = np.where(a > tail_thr * peak)[0]
        if len(lead) == 0 or len(tail) == 0:
            return audio
        margin = int((margin_ms / 1000.0) * self.sample_rate)
        start = max(0, lead[0] - margin)
        end = min(len(audio), tail[-1] + 1 + margin)
        return audio[start:end]

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

    def _find_phrase_end_cut(self, audio, word_end_sec, next_start_sec,
                             search_ahead=0.40, sustain_ms=40, rel_thr=0.08):
        """Coloca o corte FINAL no início do silêncio DEPOIS da última palavra (não no
        meio dela), evitando deixar a cauda da palavra original tocando após a frase gerada.
        Procura, a partir do pico de energia, o ponto onde a energia cai e PERMANECE baixa."""
        sr = self.sample_rate
        base = int(word_end_sec * sr)
        lo = max(0, base - int(0.05 * sr))
        hi = min(len(audio), int(next_start_sec * sr), base + int(search_ahead * sr))
        win = max(1, int(0.01 * sr))
        step = max(1, win // 2)
        if hi - lo < 2 * win:
            return min(base, len(audio))

        pos, rms = [], []
        i = lo
        while i < hi - win:
            pos.append(i + win // 2)
            rms.append(self._rms(audio[i:i + win]))
            i += step
        rms = np.array(rms)
        peak = float(rms.max())
        if peak < 1e-6:
            return pos[0]

        pk = int(np.argmax(rms))
        thr = rel_thr * peak
        sustain = max(1, int((sustain_ms / 1000.0) * sr / step))
        for j in range(pk, len(rms) - sustain):
            if np.all(rms[j:j + sustain] < thr):
                return pos[j]
        return pos[int(np.argmin(rms))]

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
                        "first_idx": first_idx,
                        "last_idx": last_idx,
                        "start": start_time,
                        "end": end_time,
                        "prev_end": prev_end,
                        "next_start": next_start,
                        "ref_text": " ".join(ref_words).strip(),
                        "gen_text": " ".join(gen_words).strip(),
                    })

                current_phrase_words = []
                current_toxic_replacements = {}

        return phrases_to_replace

    def _capped_region_reference(self, full_audio, words_list, region_start, region_end, target_sec=6.0):
        """Reescrita: referência = um trecho CURTO do próprio áudio da região
        (mesma ideia do modo Clone, que usa o áudio da própria frase), limitado a
        ~target_sec para NÃO acionar o eco/repetição do F5 com referências longas.
        O ref_text casa exatamente com o áudio recortado. Retorna (ref_audio, ref_text)."""
        sr = self.sample_rate

        # Acumula palavras da região até atingir ~target_sec (mantém ref_text
        # alinhado às fronteiras reais das palavras no áudio).
        run = []
        for w in words_list:
            if w["end"] <= region_start - 1e-3 or w["start"] >= region_end + 1e-3:
                continue  # fora da região
            run.append(w)
            if run[-1]["end"] - run[0]["start"] >= target_sec:
                break

        if not run:
            # Fallback bruto: recorte temporal da região, sem ref_text.
            s = max(0, int(region_start * sr))
            e = min(len(full_audio), int(min(region_end, region_start + target_sec) * sr))
            return full_audio[s:e], ""

        s = max(0, int(run[0]["start"] * sr))
        e = min(len(full_audio), int(run[-1]["end"] * sr))
        ref_text = " ".join(w["word"] for w in run).strip()
        return full_audio[s:e], ref_text

    def process_offline_replacement(self, full_audio, toxic_intervals, words_list, voice_cloner,
                                    whisper_model=None):
        """Substitui cada FRASE com palavrão por uma versão regerada pelo F5 (frase inteira,
        contínua => intel. e ambiência consistentes). A REFERÊNCIA é o áudio da própria frase:
        isso casa a taxa de fala do F5 com a original (sem acelerar nem deslocar palavras).
        Boundaries em vales de pausa; nível casado; crossfade equal-power."""
        print("🔧 [DSP] Modo Clone: regeração por frase (referência = áudio da própria frase)...")
        final_audio = full_audio.copy()
        sr = self.sample_rate

        if any(t.get("label") == "contextual_rewrite" for t in toxic_intervals):
            print("📝 Identificado Modo de Reescrita. Aplicando saídas diretas do LLM.")
            phrases = []
            for t in toxic_intervals:
                # Limpa ALL CAPS forçando a primeira letra maiúscula e o resto minúscula
                clean_replacement = t["replacement"].capitalize() 
                
                phrases.append({
                    "start": t["start"],
                    "end": t["end"],
                    "prev_end": max(0, t["start"] - 0.5), # Margem segura
                    "next_start": t["end"] + 0.5,
                    "ref_text": t["word"],
                    "gen_text": clean_replacement,
                    "is_rewrite": True
                })
        else:
            # Comportamento original para o Modo 2 (Clone Dicionário)
            phrases = self._build_replacement_phrases(words_list, toxic_intervals)

        for phrase in reversed(phrases):
            start_idx = self._find_safe_valley(
                full_audio, whisper_sec=phrase["start"], limit_sec=phrase["prev_end"], is_start_cut=True
            )
            end_idx = self._find_phrase_end_cut(
                full_audio, word_end_sec=phrase["end"], next_start_sec=phrase["next_start"]
            )
            start_idx = max(0, min(start_idx, len(final_audio)))
            end_idx = max(start_idx, min(end_idx, len(final_audio)))
            if end_idx - start_idx <= 0:
                continue

            gen_text = phrase["gen_text"]

            if phrase.get("is_rewrite"):
                # Reescrita: referência = trecho CURTO do próprio áudio da região
                # (como o Clone), capado p/ evitar o eco/repetição do F5 com
                # referências longas. Pacing natural; o splice usa o comprimento
                # natural do áudio gerado.
                ref_audio, ref_text = self._capped_region_reference(
                    full_audio, words_list, phrase["start"], phrase["end"]
                )
                speed = 1.0
            else:
                # Clone (dicionário): referência = a própria frase curta, com
                # correção de aceleração pela contagem de caracteres.
                # Sinônimo mais curto que o palavrão => menos tempo => acelera;
                # reduzimos o 'speed' nessa proporção (só desacelera, com piso).
                ref_audio = full_audio[start_idx:end_idx]
                ref_text = phrase["ref_text"]
                ref_chars = max(1, len(ref_text.encode("utf-8")))
                gen_chars = max(1, len(gen_text.encode("utf-8")))
                speed = float(min(1.0, max(0.82, gen_chars / ref_chars)))

            ref_sec = len(ref_audio) / sr
            print(f"\n🎙️ Clonando frase: '{ref_text}' -> '{gen_text}'  (ref {ref_sec:.1f}s, speed {speed:.2f})")

            try:
                generated = voice_cloner.generate_replacement(
                    reference_audio_array=ref_audio,
                    ref_text=ref_text,
                    text_to_say=gen_text,
                    speed=speed,
                    output_rate=self.sample_rate,
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
