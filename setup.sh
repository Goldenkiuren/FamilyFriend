#!/bin/bash

echo "Instalando dependências do sistema de áudio (Linux) e ferramentas básicas..."
sudo apt-get update && sudo apt-get install -y curl libportaudio2 portaudio19-dev

echo "Verificando o instalador de pacotes uv..."
if ! command -v uv &> /dev/null; then
    echo "uv não encontrado. Instalando..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # O script de instalação do uv geralmente o coloca em ~/.local/bin ou ~/.cargo/bin
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    echo "uv instalado com sucesso!"
else
    echo "uv já está instalado!"
fi

echo "Limpando qualquer ambiente corrompido..."
rm -rf .venv

echo "Fixando a versão do Python estritamente no 3.12..."
# Isso cria um arquivo .python-version que proíbe o uv de usar o 3.13
uv python pin 3.12

echo "Adicionando dependências do projeto e montando a venv..."
# Substituímos o TTS legado pelo F5-TTS
uv add sounddevice numpy faster-whisper transformers soundfile sentence-transformers f5-tts

echo "Injetando PyTorch otimizado para GPU (CUDA 12.1)..."
# Injetamos as versões exatas para garantir que o torchaudio não quebre a compatibilidade
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo "✅ Setup concluído com sucesso!"
echo "👉 Ative o ambiente rodando: source .venv/bin/activate"