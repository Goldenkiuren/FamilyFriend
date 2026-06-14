import os
import string
import transformers

transformers.logging.set_verbosity_error()
os.environ["TORCH_LOAD_IS_SAFE"] = "True"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import pipeline

class ToxicCensor:
    def __init__(self, threshold=0.5, device=0):
        # Mantemos o carregamento do RoBERTa pronto para o futuro Modo 3
        print("🤬 [ToxicCensor] Carregando ToxicRoBERTa (Unbiased) via PR na GPU...")
        self.classifier = pipeline(
            "text-classification", 
            model="unitary/unbiased-toxic-roberta",
            revision="refs/pr/4", 
            top_k=None, 
            device=device,
            use_safetensors=True
        )
        self.threshold = threshold
        print("✅ [ToxicCensor] Modelo carregado e pronto!")

    def detect_toxic_words(self, words_list, mode="beep"):
        """
        mode: "beep", "clone", ou "rewrite"
        """
        if not words_list:
            return []
            
        intervals_to_censor = []
        from toxic_synonyms import TOXIC_SYNONYMS

        # ================================================================
        # MODO 3 (ESQUELETO): Reescrita Contextual (IA de Frase Inteira)
        # ================================================================
        if mode == "rewrite":
            full_text = " ".join([w["word"] for w in words_list]).strip()
            if not full_text: return []
            
            sentence_scores = self.classifier(full_text)[0]
            scores_dict = {score['label']: score['score'] for score in sentence_scores}
            target_labels = ['obscene', 'identity_hate', 'severe_toxic', 'insult']
            
            sentence_is_toxic = any(scores_dict.get(label, 0) > self.threshold for label in target_labels)
            
            if sentence_is_toxic:
                # TODO: Implementar lógica de passagem para o LLM aqui no futuro
                # Por enquanto, retorna vazio para não quebrar
                pass 
                
            return intervals_to_censor

        # ================================================================
        # MODOS 1 e 2: Bip e Clonagem (Apenas Dicionário Rígido)
        # ================================================================
        for w_info in words_list:
            clean_word = w_info["word"].strip(string.punctuation + " \t\n\r")
            lower_word = clean_word.lower()
            
            if len(clean_word) < 2:
                continue
                
            if lower_word in TOXIC_SYNONYMS:
                # Se for bip, passamos None para o AudioCensor saber que deve bipar.
                # Se for clone, passamos o sinônimo.
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