import os
import string
import transformers
import gc
from toxic_synonyms import TOXIC_SYNONYMS
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()
os.environ["TORCH_LOAD_IS_SAFE"] = "True"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class ToxicCensor:
    # Rótulos que disparam uma reescrita (e que também verificam a saída do LLM).
    REWRITE_TARGET_LABELS = [
        'toxicity',
        'obscene',
        'identity_attack',
        'severe_toxicity',
        'insult',
        'threat',
        'sexual_explicit',
    ]
    # Falado pelo F5-TTS quando a reescrita falha na verificação. Neutro e sem ódio.
    SAFE_FALLBACK = "Let's just say it was a frustrating experience."

    def __init__(self, thresholds_dict=None, device=0):
        # Dicionário de limiares individuais (1.0 = Efetivamente desativado)
        self.thresholds_dict = thresholds_dict or {
            'toxicity': 1.0,
            'identity_attack': 0.5,
            'severe_toxicity': 0.5,
            'obscene': 0.5,
            'insult': 0.5,
            'threat': 0.5,
            'sexual_explicit': 0.5
        }
        self.device = device
        
        # Modelos carregados sob demanda (Lazy Loading) para poupar VRAM
        self.roberta_classifier = None
        self.llm_pipeline = None 
        
        print("🟢 [ToxicCensor] Inicializado com Dicionário Rígido e Filtros Avançados.")

    def _rewrite_with_llm(self, original_text, deterministic=True):
        """Usa o LLM para reescrever a frase, invertendo discursos de ódio.

        deterministic=True usa decodificação gulosa (reproduzível) na primeira
        tentativa; nas retentativas usamos amostragem para gerar uma saída
        DIFERENTE da que falhou na verificação."""
        prompt = [
            {"role": "system", "content": "You are a strict anti-toxicity AI. Your job is to rewrite the user's sentence. If the sentence is racist, hateful, or abusive, you MUST completely invert the hateful meaning into a positive, friendly, or complimentary observation. Do not just remove slurs; destroy the hateful intent. Output ONLY the final rewritten sentence in standard capitalization. No explanations."},
            {"role": "user", "content": f"Rewrite this: '{original_text}'"}
        ]

        gen_kwargs = dict(max_new_tokens=80, return_full_text=False)
        if deterministic:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=0.8, top_p=0.95)

        output = self.llm_pipeline(prompt, **gen_kwargs)
        return output[0]['generated_text'].strip()

    def _score_text(self, text):
        """Roda o RoBERTa e retorna {label: score}."""
        sentence_scores = self.roberta_classifier(text)[0]
        return {score['label']: score['score'] for score in sentence_scores}

    def _is_toxic(self, text):
        """True se algum rótulo-alvo ultrapassa seu limiar. Retorna (bool, scores)."""
        scores = self._score_text(text)
        hit = any(
            scores.get(label, 0) > self.thresholds_dict.get(label, 0.5)
            for label in self.REWRITE_TARGET_LABELS
        )
        return hit, scores

    def _is_toxic_windowed(self, words_list, window=10, stride=5):
        """Detecção por janelas deslizantes: pontua a frase inteira E fatias
        sobrepostas dela, sinalizando se QUALQUER uma cruzar o limiar. Impede que
        um trecho de ódio curto seja diluído numa frase longa.
        Retorna (bool, scores) — os scores da janela que disparou, ou os piores."""
        texts = []
        full_text = " ".join(w["word"] for w in words_list).strip()
        if full_text:
            texts.append(full_text)

        # Janelas de palavras sobrepostas (só se a frase for maior que a janela).
        if len(words_list) > window:
            for start in range(0, len(words_list) - 1, stride):
                chunk = words_list[start:start + window]
                if len(chunk) >= 3:  # janelas minúsculas não carregam contexto
                    texts.append(" ".join(w["word"] for w in chunk).strip())
                if start + window >= len(words_list):
                    break  # esta janela já alcançou o fim da frase

        worst_scores = {}
        worst_peak = -1.0
        for t in texts:
            hit, scores = self._is_toxic(t)
            if hit:
                return True, scores
            peak = max(scores.values()) if scores else 0.0
            if peak > worst_peak:
                worst_peak, worst_scores = peak, scores
        return False, worst_scores

    def _apply_dictionary_swap(self, text):
        """Post-pass: troca palavrões residuais do dicionário na saída do LLM."""
        out = []
        for tok in text.split():
            clean = tok.strip(string.punctuation + " \t\n\r").lower()
            out.append(TOXIC_SYNONYMS.get(clean, tok))
        return " ".join(out)

    def _rewrite_and_verify(self, original_text, max_attempts=3):
        """Reescreve, aplica o dicionário na saída e re-verifica com o RoBERTa.
        Primeira tentativa gulosa; retentativa amostrada; senão, fallback seguro."""
        for attempt in range(max_attempts):
            rewritten = self._rewrite_with_llm(original_text, deterministic=(attempt == 0))
            rewritten = self._apply_dictionary_swap(rewritten)

            still_toxic, scores = self._is_toxic(rewritten)
            if not still_toxic:
                return rewritten
            print(f"↻ Reescrita ainda tóxica (tentativa {attempt + 1}/{max_attempts}). Score: {scores}")

        print("⛔ Reescrita falhou na verificação. Usando fallback seguro.")
        return self.SAFE_FALLBACK

    def _load_ai_models_if_needed(self):
        """Carrega ou move o RoBERTa e o LLM de volta para a GPU."""
        if self.roberta_classifier is None:
            print("🤬 [ToxicCensor] Carregando ToxicRoBERTa na GPU...")
            self.roberta_classifier = pipeline(
                "text-classification", 
                model="unitary/unbiased-toxic-roberta",
                revision="refs/pr/4", 
                top_k=None, 
                device=self.device,
                use_safetensors=True
            )
        elif self.roberta_classifier.model.device.type == 'cpu':
            self.roberta_classifier.model.to(self.device)
            
        if self.llm_pipeline is None:
            print("🧠 [ToxicCensor] Carregando LLM de Reescrita (Qwen-3B) na GPU...")
            model_id = "Qwen/Qwen2.5-3B-Instruct"
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✅ [ToxicCensor] Modelos de IA prontos na VRAM!")
        elif getattr(self.llm_pipeline.model, 'device', None) and self.llm_pipeline.model.device.type == 'cpu':
             self.llm_pipeline.model.to(self.device)

    def offload_models(self):
        """Move os modelos NLP para a RAM do sistema para liberar espaço ao F5-TTS."""
        if self.roberta_classifier is not None:
            self.roberta_classifier.model.to("cpu")
        if self.llm_pipeline is not None:
            # Qwen com device_map auto pode precisar de uma iteração manual se estiver fragmentado,
            # mas na 4080 o 3B cabe inteiro no device 0.
            self.llm_pipeline.model.to("cpu")
            
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 [ToxicCensor] VRAM liberada. Modelos NLP estacionados na RAM.")

    def detect_toxic_words(self, words_list, mode="beep"):
        if not words_list:
            return []
            
        intervals_to_censor = []

        if mode == "rewrite":
            self._load_ai_models_if_needed()
            
            # Aqui assumimos que `words_list` já representa uma frase coerente
            # (passada pelo loop de frases no áudio censor)
            original_full_text = " ".join([w["word"] for w in words_list]).strip()
            if not original_full_text: return []

            # Detecção: pontua o texto BRUTO (sem pré-troca de dicionário, que
            # mascararia o sinal de ódio) em janelas deslizantes, para que um
            # trecho de ódio curto não se dilua numa frase longa.
            is_toxic, scores_dict = self._is_toxic_windowed(words_list)

            if is_toxic:
                print(f"⚠️ Toxicidade Detectada! Passando para o LLM. Score: {scores_dict}")

                # Reescreve com contexto total, aplica o dicionário na saída e
                # re-verifica; se ainda for tóxico, retenta amostrando; senão, fallback.
                rewritten_text = self._rewrite_and_verify(original_full_text)

                # Embala o resultado no formato esperado pelo F5-TTS
                # Passamos o intervalo de tempo de toda a frase
                intervals_to_censor.append({
                    "start": words_list[0]["start"],
                    "end": words_list[-1]["end"],
                    "word": original_full_text,
                    "replacement": rewritten_text,
                    "label": "contextual_rewrite",
                    "score": max(scores_dict.values())
                })
                
            return intervals_to_censor


        # ================================================================
        # MODOS 1 e 2: Bip e Clonagem (Apenas Dicionário Rígido)
        # ================================================================
        for w_info in words_list:
            # Limpa as pontuações e espaçamentos grudados na palavra
            clean_word = w_info["word"].strip(string.punctuation + " \t\n\r")
            lower_word = clean_word.lower()
            
            if len(clean_word) < 2:
                continue
                
            if lower_word in TOXIC_SYNONYMS:
                # Se for bip, passamos None para o AudioCensor saber que deve bipar.
                # Se for clone, passamos o sinônimo da lista.
                replacement = TOXIC_SYNONYMS[lower_word] if mode == "clone" else None
                
                intervals_to_censor.append({
                    "start": w_info["start"],
                    "end": w_info["end"],
                    "word": clean_word,
                    "replacement": replacement,
                    "label": "hard_dict_match",
                    "score": 1.0
                })
                
        return intervals_to_censor