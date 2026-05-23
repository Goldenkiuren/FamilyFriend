# FamilyFriend
 Audio and image censoring for livestreaming
 - Whisper para audio transcription
 - Bert para classificação do texto

 
# TODO:

## Entrada: (Augusto)
 - captura de audio
 - chunking com overlap
 - transição (whisper) com timestamps
 - saida: buffer 

## Bert: (Giovanni, Luis e Mateus)
 - dataset de xingamento em inglês (derivado de xingamentos verbais, e não textuais com abreviações e etc)
 - modelo específico
 - fine-tuning
 - métricas
 - saida: palavras classificadas

 ## Concatenatação: (Em aberto , depois)
  - recebe timestamps e palavras classificadas
  - mapeia para audio um "bip" no timestamp
  - remove overlaps e envia audio pra buffer de saida reconstruido


  ## Paper (Mateus)
  
