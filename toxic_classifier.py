import os
import string
import transformers

transformers.logging.set_verbosity_error()
# Desativa as proteções de segurança recentes para permitir o carregamento do modelo antigo
os.environ["TORCH_LOAD_IS_SAFE"] = "True"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import pipeline

class ToxicCensor:
    def __init__(self, threshold=0.5, device=0):
        """
        Inicializa o modelo ToxicRoBERTa Unbiased puxando os SafeTensors 
        direto do Pull Request do Hugging Face.
        """
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

    def detect_toxic_words(self, words_list, use_synonyms=False):
        if not words_list:
            return []

        full_text = " ".join([w["word"] for w in words_list]).strip()
        if not full_text:
            return []
            
        sentence_scores = self.classifier(full_text)[0]
        scores_dict = {score['label']: score['score'] for score in sentence_scores}
        
        target_labels = ['obscene', 'identity_hate', 'severe_toxic', 'insult']
        
        # Guardamos se a frase é tóxica, mas NÃO damos um 'return []' antecipado.
        sentence_is_toxic = any(scores_dict.get(label, 0) > self.threshold for label in target_labels)
            
        intervals_to_censor = []
        
        # Importa o dicionário apenas se o modo híbrido (offline) estiver ativo
        if use_synonyms:
            from toxic_synonyms import TOXIC_SYNONYMS
        
        for w_info in words_list:
            clean_word = w_info["word"].strip(string.punctuation + " \t\n\r")
            lower_word = clean_word.lower()
            
            if len(clean_word) < 2:
                continue
                
            # ================================================================
            # PASSO 1 (HÍBRIDO): Verificação Rígida de Dicionário
            # ================================================================
            # Só ocorre no modo gravação (onde use_synonyms é True)
            if use_synonyms and lower_word in TOXIC_SYNONYMS:
                intervals_to_censor.append({
                    "start": w_info["start"],
                    "end": w_info["end"],
                    "word": clean_word,
                    "replacement": TOXIC_SYNONYMS[lower_word],
                    "label": "hard_dict_match",
                    "score": 1.0  # Confiança máxima pois é um match direto no dicionário
                })
                continue # Pula a IA para essa palavra, já resolvemos
                
            # ================================================================
            # PASSO 2 (IA Clássica): RoBERTa
            # ================================================================
            # Se a palavra não estava no dicionário (ou se estamos ao vivo e o dicionário está desligado),
            # verificamos se a IA achou a frase suspeita para investigar a palavra.
            if sentence_is_toxic:
                word_score = self.classifier(clean_word)[0]
                w_scores_dict = {s['label']: s['score'] for s in word_score}
                
                triggered_labels = {
                    label: w_scores_dict.get(label, 0) 
                    for label in target_labels 
                    if w_scores_dict.get(label, 0) > self.threshold
                }
                
                if triggered_labels:
                    top_label = max(triggered_labels, key=triggered_labels.get)
                    top_score = triggered_labels[top_label]
                    
                    replacement = None
                    if use_synonyms:
                        continue
                    
                    intervals_to_censor.append({
                        "start": w_info["start"],
                        "end": w_info["end"],
                        "word": clean_word,
                        "replacement": replacement,
                        "label": top_label,
                        "score": top_score
                    })
                    
        return intervals_to_censor

# ==========================================
# Teste isolado do módulo
# ==========================================
if __name__ == "__main__":
    censor = ToxicCensor(threshold=0.5)
    
    mock_whisper_output = [
        {"word": "Hey", "start": 0.0, "end": 0.3},
        {"word": "you", "start": 0.3, "end": 0.5},
        {"word": "fucking", "start": 0.5, "end": 0.9},
        {"word": "idiot", "start": 0.9, "end": 1.3},
        {"word": "stop", "start": 1.3, "end": 1.7},
        {"word": "talking", "start": 1.7, "end": 2.1}
    ]
    
    print("\nSimulando entrada do Whisper:")
    print("Frase gerada: 'Hey you fucking idiot stop talking'")
    
    print("\n--- Teste 1: Modo Ao Vivo (use_synonyms=False) ---")
    live_intervals = censor.detect_toxic_words(mock_whisper_output, use_synonyms=False)
    for item in live_intervals:
        print(f"🚨 Detectado pela IA: '{item['word']}' [{item['label']}]")
        
    print("\n--- Teste 2: Modo Gravação Híbrido (use_synonyms=True) ---")
    offline_intervals = censor.detect_toxic_words(mock_whisper_output, use_synonyms=True)
    for item in offline_intervals:
        print(f"🚨 Detectado (IA ou Dict): '{item['word']}' [{item['label']}] -> Substituir por: '{item['replacement']}'")