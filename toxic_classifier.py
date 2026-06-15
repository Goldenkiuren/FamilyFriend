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

    def _rewrite_with_llm(self, original_text):
        """Usa o LLM para reescrever a frase, invertendo discursos de ódio."""
        prompt = [
            {"role": "system", "content": "You are a strict anti-toxicity AI. Your job is to rewrite the user's sentence. If the sentence is racist, hateful, or abusive, you MUST completely invert the hateful meaning into a positive, friendly, or complimentary observation. Do not just remove slurs; destroy the hateful intent. Output ONLY the final rewritten sentence in standard capitalization. No explanations."},
            {"role": "user", "content": f"Rewrite this: '{original_text}'"}
        ]
        
        output = self.llm_pipeline(
            prompt, 
            max_new_tokens=60, 
            temperature=0.6,
            do_sample=True,
            return_full_text=False
        )
        return output[0]['generated_text'].strip()

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
            
            # 1. DOUBLE PASS: Substituição por dicionário para não "assustar" o RoBERTa
            # com palavrões casuais (ex: "Fuck, she is pretty" -> "Screw, she is pretty")
            pre_processed_words = []
            for w in words_list:
                clean = w["word"].strip(string.punctuation + " \t\n\r").lower()
                if clean in TOXIC_SYNONYMS:
                    pre_processed_words.append(TOXIC_SYNONYMS[clean])
                else:
                    pre_processed_words.append(w["word"])
            
            double_pass_text = " ".join(pre_processed_words)
            
            # 2. Avaliação do RoBERTa no texto pré-processado
            sentence_scores = self.roberta_classifier(double_pass_text)[0]
            scores_dict = {score['label']: score['score'] for score in sentence_scores}
            target_labels = [
                'toxicity',
                'obscene', 
                'identity_attack', 
                'severe_toxicity', 
                'insult', 
                'threat', 
                'sexual_explicit'
            ]
            
            # Se a toxicidade profunda for detectada
            if any(scores_dict.get(label, 0) > self.thresholds_dict.get(label, 0.5) for label in target_labels):
                print(f"⚠️ Toxicidade Detectada! Passando para o LLM. Score: {scores_dict}")
                
                # O LLM reescreve a frase original para ter contexto total
                rewritten_text = self._rewrite_with_llm(original_full_text)
                
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