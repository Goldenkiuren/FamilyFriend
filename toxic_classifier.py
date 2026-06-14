import os
import string
import transformers
from toxic_synonyms import TOXIC_SYNONYMS

transformers.logging.set_verbosity_error()
os.environ["TORCH_LOAD_IS_SAFE"] = "True"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import pipeline

class ToxicCensor:
    def __init__(self, threshold=0.5, device=0):
        self.threshold = threshold
        self.device = device
        self.classifier = None # Começamos vazio para poupar VRAM!
        
        print("🟢 [ToxicCensor] Inicializado com Dicionário Rígido.")
        print("   (Modelo RoBERTa em espera. Só será carregado no Modo 3)")

    def _load_roberta_if_needed(self):
        """Gatilho de Lazy Loading: Só roda quando estritamente necessário."""
        if self.classifier is None:
            print("🤬 [ToxicCensor] Carregando ToxicRoBERTa (Unbiased) na GPU...")
            self.classifier = pipeline(
                "text-classification", 
                model="unitary/unbiased-toxic-roberta",
                revision="refs/pr/4", 
                top_k=None, 
                device=self.device,
                use_safetensors=True
            )
            print("✅ [ToxicCensor] RoBERTa carregado e pronto!")

    def detect_toxic_words(self, words_list, mode="beep"):
        """
        mode: "beep", "clone", ou "rewrite"
        """
        if not words_list:
            return []
            
        intervals_to_censor = []

        # ================================================================
        # MODO 3 (ESQUELETO): Reescrita Contextual (IA de Frase Inteira)
        # ================================================================
        if mode == "rewrite":
            # O gatilho é acionado aqui!
            self._load_roberta_if_needed()
            
            full_text = " ".join([w["word"] for w in words_list]).strip()
            if not full_text: return []
            
            sentence_scores = self.classifier(full_text)[0]
            scores_dict = {score['label']: score['score'] for score in sentence_scores}
            target_labels = ['obscene', 'identity_hate', 'severe_toxic', 'insult']
            
            sentence_is_toxic = any(scores_dict.get(label, 0) > self.threshold for label in target_labels)
            
            if sentence_is_toxic:
                # TODO: Implementar lógica de passagem para o LLM reescrever a frase.
                pass 
                
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