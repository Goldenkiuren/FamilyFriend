# FamilyFriend
 Audio and image censoring for livestreaming

## 🛠️ Pré-requisitos

Se você estiver usando Linux, será necessário instalar a biblioteca PortAudio no sistema antes de rodar o projeto:

```bash
sudo apt update
sudo apt install libportaudio2 portaudio19-dev
 

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
