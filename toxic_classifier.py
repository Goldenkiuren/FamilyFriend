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
        
        # Categorias que você quer censurar
        target_labels = ['obscene', 'identity_hate', 'severe_toxic', 'insult']
        
        if not any(scores_dict.get(label, 0) > self.threshold for label in target_labels):
            return []
            
        intervals_to_censor = []
        
        for w_info in words_list:
            clean_word = w_info["word"].strip(string.punctuation + " \t\n\r")
            
            if len(clean_word) < 2:
                continue
                
            word_score = self.classifier(clean_word)[0]
            w_scores_dict = {s['label']: s['score'] for s in word_score}
            
            # Filtra apenas as labels que passaram do limiar de censura
            triggered_labels = {
                label: w_scores_dict.get(label, 0) 
                for label in target_labels 
                if w_scores_dict.get(label, 0) > self.threshold
            }
            
            if triggered_labels:
                # Encontra qual foi a categoria com a maior pontuação para essa palavra
                top_label = max(triggered_labels, key=triggered_labels.get)
                top_score = triggered_labels[top_label]
                
                # AQUI: Lógica unificada para definir o sinônimo apenas se solicitado
                replacement = None
                if use_synonyms:
                    from toxic_synonyms import TOXIC_SYNONYMS
                    replacement = TOXIC_SYNONYMS.get(clean_word.lower(), "bleep")
                
                # Adiciona na lista uma única vez, com todas as chaves corretas
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
    
    # Testando com a flag ativada
    toxic_intervals = censor.detect_toxic_words(mock_whisper_output, use_synonyms=True)
    
    print("\nResultados do Censor:")
    if toxic_intervals:
        for item in toxic_intervals:
            print(f"🚨 Detectado: '{item['word']}' [{item['label']}] -> Substituir por: '{item['replacement']}'")
    else:
        print("Tudo limpo, nenhum bip necessário.")