#!/bin/bash
echo "Instalando dependências do sistema de áudio (Linux)..."
sudo apt-get update && sudo apt-get install -y libportaudio2 portaudio19-dev

echo "Limpando qualquer ambiente corrompido..."
rm -rf .venv

echo "Fixando a versão do Python estritamente no 3.12..."
# Isso cria um arquivo .python-version que proíbe o uv de usar o 3.13
uv python pin 3.12

echo "Adicionando dependências do projeto e montando a venv..."
uv add sounddevice numpy faster-whisper transformers

echo "Injetando PyTorch otimizado para GPU (CUDA 12.1)..."
# Injetamos no final para garantir que o uv não destrua a venv depois
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "✅ Setup concluído com sucesso!"
echo "👉 Ative o ambiente rodando: source .venv/bin/activate"