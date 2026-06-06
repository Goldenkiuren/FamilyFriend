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

    def detect_toxic_words(self, words_list):
        if not words_list:
            return []

        target_labels = ['obscene', 'identity_hate', 'severe_toxic', 'insult']
        intervals_to_censor = []
        
        # Parâmetro de segurança para chunking (200 palavras garantem que não passaremos de 512 tokens na GPU)
        CHUNK_SIZE = 200

        # Divide a lista de palavras recebida em lotes menores
        for i in range(0, len(words_list), CHUNK_SIZE):
            chunk_words = words_list[i:i + CHUNK_SIZE]
            full_text = " ".join([w["word"] for w in chunk_words]).strip()

            if not full_text:
                continue

            # Avalia a frase inteira do lote atual. 
            # Mantemos o truncation=True apenas como última linha de defesa à prova de falhas.
            sentence_scores = self.classifier(full_text, truncation=True, max_length=512)[0]
            scores_dict = {score['label']: score['score'] for score in sentence_scores}
            
            # Se o lote inteiro for limpo, pula a verificação palavra por palavra (Otimização de GPU)
            if not any(scores_dict.get(label, 0) > self.threshold for label in target_labels):
                continue
                
            # Se a IA detectou toxicidade no lote, investiga palavra por palavra dentro desse pedaço
            for w_info in chunk_words:
                clean_word = w_info["word"].strip(string.punctuation + " \t\n\r")
                
                if len(clean_word) < 2:
                    continue
                    
                word_score = self.classifier(clean_word, truncation=True, max_length=512)[0]
                w_scores_dict = {s['label']: s['score'] for s in word_score}
                
                triggered_labels = {
                    label: w_scores_dict.get(label, 0) 
                    for label in target_labels 
                    if w_scores_dict.get(label, 0) > self.threshold
                }
                
                if triggered_labels:
                    top_label = max(triggered_labels, key=triggered_labels.get)
                    top_score = triggered_labels[top_label]
                    
                    result_item = w_info.copy()
                    result_item.update({
                        "word": clean_word,
                        "label": top_label,
                        "score": top_score
                    })
                    
                    intervals_to_censor.append(result_item)
                    
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
    
    toxic_intervals = censor.detect_toxic_words(mock_whisper_output)
    
    print("\nResultados do Censor:")
    if toxic_intervals:
        for item in toxic_intervals:
            print(f"🚨 Bipar: '{item['word']}' [{item['label']} - {(item['score']*100):.1f}%] de {item['start']}s até {item['end']}s")
    else:
        print("Tudo limpo, nenhum bip necessário.")


"""
A resposta curta é: Sim, você definitivamente deveria fazer o fine-tuning posteriormente.

A sua heurística atual (passar a frase inteira e depois palavra por palavra) atende ao seu requisito de censurar palavrões independentemente do contexto. No entanto, se o seu objetivo a longo prazo envolve pesquisa acadêmica na área (como o seu interesse em um mestrado e doutorado focados em IA), você precisará demonstrar rigor metodológico e eficiência arquitetural — e o fine-tuning resolve três problemas centrais que a heurística zero-shot não consegue cobrir:
1. Eficiência de Inferência (O "Overhead" Computacional)

Atualmente, se o Whisper gerar uma frase ofensiva com 15 palavras, a sua RTX 4080 fará 16 inferências na rede neural:

    1 vez para a frase completa.

    15 vezes (uma para cada palavra isolada).

Isso não é um problema grave no seu setup atual porque a 4080 lida com esses modelos menores sem gargalos. Mas, se você escalar isso para um servidor na nuvem processando múltiplos canais de áudio, essa ineficiência custará caro. Um modelo fine-tunado para Token Classification faz o trabalho em uma única inferência, rotulando cada palavra (token) de uma só vez, reduzindo o tempo de processamento em mais de 90%.
2. A Precisão do Tokenizer (Palavras Cortadas)

O modelo RoBERTa não lê palavras como nós lemos; ele lê subwords usando um Byte-Pair Encoding (BPE). Quando você passa uma palavra isolada como "motherfucker", o tokenizer pode dividi-la em "mother" e "fucker".
Na sua heurística, você passa a string "motherfucker" limpa. O modelo a classifica, mas ele perde a capacidade de alinhar a toxicidade com o token exato que gerou a ofensa, o que pode causar inconsistências na detecção de xingamentos compostos ou palavras não vistas no treinamento.
3. Alinhamento Fino com Seus Critérios

O unbiased-toxic-roberta foi treinado com os critérios do Jigsaw Toxic Comment Classification Challenge de anos atrás. O que a internet considerava identity_hate ou obscene na época não abrange a totalidade de novas gírias ou slurs modernos (e as variações criativas que o Whisper eventualmente tentar transcrever). Fazer o fine-tuning usando um dataset como o SemEval-2021 Task 5 (Toxic Spans Detection) permite que você:

    Treine o modelo para ignorar explicitamente false positives que você não quer censurar.

    Adicione as suas próprias palavras à base de dados.

    Modifique a última camada (o classification head) para ser uma saída binária simples (Censurar: Sim/Não), em vez de lidar com as 6 categorias originais do Jigsaw
"""