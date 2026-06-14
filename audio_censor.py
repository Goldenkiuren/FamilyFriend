import numpy as np
import string
import difflib

def _norm(token):
    """Normaliza um token para casamento de texto (minúsculo, sem pontuação)."""
    return token.lower().strip(string.punctuation + " \t\n\r")

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

    @staticmethod
    def _rms(audio):
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))

    def _match_rms(self, segment, target_rms, max_gain=8.0):
        """Casa o nível (loudness) do segmento gerado com o contexto original."""
        cur = self._rms(segment)
        if cur < 1e-6 or target_rms <= 0:
            return segment
        gain = min(target_rms / cur, max_gain)
        return (segment * gain).astype(np.float32)

    def _trim_edges(self, audio, rel_threshold=0.02):
        """Remove silêncio residual nas bordas do áudio gerado."""
        if len(audio) == 0:
            return audio
        peak = np.max(np.abs(audio))
        if peak < 1e-4:
            return audio
        non_silent = np.where(np.abs(audio) > rel_threshold * peak)[0]
        if len(non_silent) == 0:
            return audio
        return audio[non_silent[0]:non_silent[-1] + 1]

    def _snap_to_zero_crossing(self, audio, idx, search_ms=5):
        """Move o ponto de corte para o zero-crossing mais próximo (evita cliques)."""
        idx = int(max(0, min(idx, len(audio) - 1)))
        radius = int((search_ms / 1000.0) * self.sample_rate)
        lo = max(1, idx - radius)
        hi = min(len(audio) - 1, idx + radius)
        if hi <= lo:
            return idx
        best_idx, best_dist = idx, radius + 1
        for i in range(lo, hi):
            if audio[i - 1] <= 0 <= audio[i] or audio[i - 1] >= 0 >= audio[i]:
                dist = abs(i - idx)
                if dist < best_dist:
                    best_idx, best_dist = i, dist
        return best_idx

    def _valley_index(self, audio, lo_sec, hi_sec, default_sec, window_ms=5):
        """Acha o sample de menor energia (vale acústico) dentro de [lo_sec, hi_sec]."""
        lo = int(max(0, lo_sec) * self.sample_rate)
        hi = int(min(len(audio) / self.sample_rate, hi_sec) * self.sample_rate)
        default_idx = int(default_sec * self.sample_rate)
        if hi - lo < 2:
            return default_idx
        win = max(1, int((window_ms / 1000.0) * self.sample_rate))
        step = max(1, win // 2)
        best_rms, best_i = None, default_idx
        for i in range(lo, hi - win, step):
            rms = self._rms(audio[i:i + win])
            if best_rms is None or rms < best_rms:
                best_rms, best_i = rms, i + win // 2
        return best_i

    def _find_word_cut(self, audio, w_start_sec, w_end_sec, prev_end_sec, next_start_sec):
        """Define o recorte cirúrgico de UMA palavra: vale acústico na folga até os
        vizinhos + snap em zero-crossing. Busca preferencialmente no gap entre palavras."""
        start_idx = self._valley_index(
            audio,
            lo_sec=max(prev_end_sec, w_start_sec - 0.12),
            hi_sec=w_start_sec + 0.03,
            default_sec=w_start_sec,
        )
        end_idx = self._valley_index(
            audio,
            lo_sec=w_end_sec - 0.03,
            hi_sec=min(next_start_sec, w_end_sec + 0.12),
            default_sec=w_end_sec,
        )
        start_idx = self._snap_to_zero_crossing(audio, start_idx)
        end_idx = self._snap_to_zero_crossing(audio, end_idx)
        if end_idx <= start_idx:
            end_idx = min(len(audio), int(w_end_sec * self.sample_rate))
        return start_idx, end_idx

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

    def _splice_replace(self, audio, insert, start_idx, end_idx, fade_samples=256):
        """Substitui audio[start:end] por `insert` no comprimento NATURAL do insert
        (a linha do tempo estica/encolhe conforme necessário) com crossfade curto nas bordas."""
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
        """Agrupa palavras em frases-portadoras (carrier). Cada frase com palavrão vira
        um trabalho de geração; dentro dela, cada palavrão é rastreado pela sua POSIÇÃO
        de token no gen_text, para depois extrairmos só ela do áudio gerado."""
        phrases = []
        if not words_list or not toxic_intervals:
            return phrases

        toxic_map = {round(t["start"], 2): t for t in toxic_intervals}
        n = len(words_list)
        current = []  # lista de (global_idx, word_info)

        def flush(group):
            if not group:
                return
            has_toxic = any(round(wi["start"], 2) in toxic_map for _, wi in group)
            if not has_toxic:
                return

            gen_tokens = []        # tokens normalizados do gen_text (para alinhamento)
            gen_display = []        # texto legível passado ao F5
            replacements = []

            for pos, (gidx, wi) in enumerate(group):
                key = round(wi["start"], 2)
                if key in toxic_map:
                    syn = toxic_map[key].get("replacement") or "bleep"
                    syn_tokens = [_norm(t) for t in syn.split() if _norm(t)]
                    if not syn_tokens:
                        syn_tokens = ["bleep"]
                    tok_start = len(gen_tokens)
                    gen_tokens.extend(syn_tokens)
                    gen_display.append(syn)
                    prev_end = words_list[gidx - 1]["end"] if gidx > 0 else 0.0
                    next_start = words_list[gidx + 1]["start"] if gidx < n - 1 else wi["end"] + 1.0
                    replacements.append({
                        "syn_text": syn,
                        "tok_start": tok_start,
                        "tok_len": len(syn_tokens),
                        "orig_start": wi["start"],
                        "orig_end": wi["end"],
                        "prev_end": prev_end,
                        "next_start": next_start,
                    })
                else:
                    clean = wi["word"].strip()
                    gen_tokens.append(_norm(clean))
                    gen_display.append(clean)

            phrases.append({
                "first_idx": group[0][0],
                "last_idx": group[-1][0],
                "phrase_start": group[0][1]["start"],
                "gen_text": " ".join(gen_display).strip(),
                "gen_tokens": gen_tokens,
                "replacements": replacements,
            })

        for i, w in enumerate(words_list):
            current.append((i, w))
            is_last = (i == n - 1)
            do_break = is_last
            if not is_last:
                gap = words_list[i + 1]["start"] - w["end"]
                punct_break = w["word"].strip().endswith(('.', '!', '?'))
                dur = w["end"] - current[0][1]["start"]
                if gap > pause_threshold or punct_break or dur >= max_duration:
                    do_break = True
            if do_break:
                flush(current)
                current = []

        return phrases

    def _build_reference(self, full_audio, words_list, first_idx, last_idx, toxic_idx_set,
                         turn_gap=0.8, max_ref=8.0, min_ref=0.4):
        """Seleciona uma referência de voz LIMPA dentro do mesmo turno de fala.
        Multi-locutor: como os locutores se alternam (sem sobreposição), expandimos só
        até pausas <= turn_gap, o que mantém a referência no mesmo falante sem diarização."""
        n = len(words_list)

        # 1. Expande para os limites do turno (pausas curtas = mesmo falante)
        t0 = first_idx
        while t0 - 1 >= 0 and (words_list[t0]["start"] - words_list[t0 - 1]["end"]) <= turn_gap:
            t0 -= 1
        t1 = last_idx
        while t1 + 1 < n and (words_list[t1 + 1]["start"] - words_list[t1]["end"]) <= turn_gap:
            t1 += 1

        # 2. Coleta corridas contíguas de palavras limpas (sem palavrão) no turno
        runs, cur = [], []
        for i in range(t0, t1 + 1):
            if i in toxic_idx_set:
                if cur:
                    runs.append(cur); cur = []
                continue
            if cur and (words_list[i]["start"] - words_list[cur[-1]]["end"]) > turn_gap:
                runs.append(cur); cur = []
            cur.append(i)
        if cur:
            runs.append(cur)
        if not runs:
            return None, None

        def dur(run):
            return words_list[run[-1]]["end"] - words_list[run[0]]["start"]

        best = max(runs, key=dur)
        if dur(best) < min_ref:
            return None, None

        # 3. Limita a max_ref mantendo as palavras mais próximas da frase
        ref_anchor = first_idx
        best = best[:]
        while dur(best) > max_ref and len(best) > 1:
            if abs(best[0] - ref_anchor) >= abs(best[-1] - ref_anchor):
                best.pop(0)
            else:
                best.pop()

        s = int(words_list[best[0]]["start"] * self.sample_rate)
        e = int(words_list[best[-1]]["end"] * self.sample_rate)
        ref_audio = full_audio[max(0, s):min(len(full_audio), e)]
        ref_text = " ".join(words_list[i]["word"].strip() for i in best).strip()
        if len(ref_audio) < int(min_ref * self.sample_rate) or not ref_text:
            return None, None
        return ref_audio, ref_text

    def _locate_in_carrier(self, gen_tokens, carrier_words, tok_start, tok_len, carrier_dur):
        """Localiza, no áudio gerado, o intervalo (s,e) em segundos correspondente aos
        tokens do sinônimo, casando a sequência de tokens com o que o Whisper ouviu."""
        C = [_norm(cw["word"]) for cw in carrier_words]
        a2b = {}
        sm = difflib.SequenceMatcher(a=gen_tokens, b=C, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for off in range(i2 - i1):
                    a2b[i1 + off] = j1 + off

        mapped = [a2b[t] for t in range(tok_start, tok_start + tok_len) if t in a2b]
        if mapped:
            s = carrier_words[min(mapped)]["start"]
            e = carrier_words[max(mapped)]["end"]
            return float(s), float(e)

        # Fallback proporcional caso o Whisper não tenha casado o sinônimo
        total = max(1, len(gen_tokens))
        return (tok_start / total) * carrier_dur, ((tok_start + tok_len) / total) * carrier_dur

    def _beep_into(self, final_audio, start_idx, end_idx):
        duration = (end_idx - start_idx) / self.sample_rate
        if duration <= 0:
            return final_audio
        beep = self._generate_beep(duration)
        return self._splice_replace(final_audio, beep, start_idx, end_idx, fade_samples=256)

    def process_offline_replacement(self, full_audio, toxic_intervals, words_list, voice_cloner,
                                    whisper_model=None):
        """Motor do Modo Clone: gera a palavra DENTRO do contexto da frase (coarticulação
        natural), realinha o áudio gerado com o Whisper, extrai SÓ a palavra trocada e a
        encaixa cirurgicamente no slot original. Mantém ~todo o áudio original intacto."""
        print("🔧 [DSP] Reconstrução por palavra-em-contexto (gerar -> realinhar -> extrair -> encaixar)...")
        final_audio = full_audio.copy()
        sr = self.sample_rate

        toxic_idx_set = set()
        toxic_map = {round(t["start"], 2): t for t in toxic_intervals}
        for gidx, w in enumerate(words_list):
            if round(w["start"], 2) in toxic_map:
                toxic_idx_set.add(gidx)

        phrases = self._build_replacement_phrases(words_list, toxic_intervals)

        for phrase in reversed(phrases):
            replacements = phrase["replacements"]
            if not replacements:
                continue

            ref_audio, ref_text = self._build_reference(
                full_audio, words_list, phrase["first_idx"], phrase["last_idx"], toxic_idx_set
            )

            carrier_words = None
            carrier_audio = None

            if ref_audio is not None and whisper_model is not None:
                try:
                    print(f"🎙️ Gerando carrier: '{phrase['gen_text']}'  (ref: '{ref_text[:50]}...')")
                    carrier_audio = voice_cloner.generate_replacement(
                        reference_audio_array=ref_audio,
                        ref_text=ref_text,
                        text_to_say=phrase["gen_text"],
                    )
                    carrier_audio = self._trim_edges(carrier_audio)
                    segs, _ = whisper_model.transcribe(
                        carrier_audio, language="en", word_timestamps=True,
                        beam_size=5, vad_filter=False, condition_on_previous_text=False,
                    )
                    carrier_words = []
                    for seg in segs:
                        if seg.words:
                            for ww in seg.words:
                                carrier_words.append({"word": ww.word, "start": ww.start, "end": ww.end})
                except Exception as e:
                    print(f"⚠️ Falha ao gerar/alinhar carrier: {e}")
                    carrier_audio, carrier_words = None, None

            carrier_dur = (len(carrier_audio) / sr) if carrier_audio is not None else 0.0

            # Encaixa cada palavra trocada, da direita para a esquerda (índices estáveis)
            for rep in sorted(replacements, key=lambda r: r["orig_start"], reverse=True):
                start_idx, end_idx = self._find_word_cut(
                    full_audio, rep["orig_start"], rep["orig_end"], rep["prev_end"], rep["next_start"]
                )
                start_idx = max(0, min(start_idx, len(final_audio)))
                end_idx = max(start_idx, min(end_idx, len(final_audio)))
                slot_len = end_idx - start_idx

                # Sem carrier (ref insuficiente, multi-locutor sem trecho limpo, etc.) -> bip
                if carrier_audio is None or not carrier_words or slot_len <= 0:
                    final_audio = self._beep_into(final_audio, start_idx, end_idx)
                    continue

                s_sec, e_sec = self._locate_in_carrier(
                    phrase["gen_tokens"], carrier_words, rep["tok_start"], rep["tok_len"], carrier_dur
                )
                # Padding nas bordas para não cortar consoantes de ataque/final (ex.: /f/, /s/)
                pad = int(0.045 * sr)
                g_s = max(0, int(s_sec * sr) - pad)
                g_e = min(len(carrier_audio), int(e_sec * sr) + pad)
                gen_word = carrier_audio[g_s:g_e].astype(np.float32).copy()

                if len(gen_word) < int(0.02 * sr):
                    final_audio = self._beep_into(final_audio, start_idx, end_idx)
                    continue

                # Nível: casa o loudness com o da palavra ORIGINAL que está sendo trocada
                # (a região da palavra é fala; usar os silêncios vizinhos deixaria baixo demais).
                ow_s = int(rep["orig_start"] * sr)
                ow_e = int(rep["orig_end"] * sr)
                target_rms = self._rms(full_audio[ow_s:ow_e])
                if target_rms < 1e-5:
                    target_rms = self._rms(full_audio)
                gen_word = self._match_rms(gen_word, target_rms, max_gain=4.0)

                # Sem time-stretch: mantém a palavra no comprimento natural (a linha do tempo
                # acomoda a diferença). Apenas limita picos para não estourar.
                peak = float(np.max(np.abs(gen_word))) if len(gen_word) else 0.0
                if peak > 0.99:
                    gen_word *= (0.99 / peak)

                print(f"   ↳ '{rep['syn_text']}' encaixado "
                      f"({slot_len/sr:.2f}s slot / {len(gen_word)/sr:.2f}s gerado, natural)")

                final_audio = self._splice_replace(
                    final_audio, gen_word, start_idx, end_idx, fade_samples=256
                )

        # Limitador de segurança: garante que nada estoure após os crossfades
        peak = float(np.max(np.abs(final_audio))) if len(final_audio) else 0.0
        if peak > 0.99:
            final_audio = (final_audio * (0.99 / peak)).astype(np.float32)

        return final_audio