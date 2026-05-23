# FamilyFriend
 Audio and image censoring for livestreaming
TODO:

Entrada:
 captura de audio
 chunking com overlap
 transição (whisper) com timestamps
 saida: buffer 

Bert:
 dataset de xingamento em inglês (derivado de xingamentos verbais, e não textuais com abreviações e etc)
 modelo específico
 fine-tuning
 métricas
 saida: palavras classificadas

 Concatenatação:
  recebe timestamps e palavras classificadas
  mapeia para audio um "bip" no timestamp
  remove overlaps e envia audio pra buffer de saida reconstruido
