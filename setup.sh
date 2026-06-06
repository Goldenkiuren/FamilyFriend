#!/bin/bash
echo "Instalando dependências do sistema de áudio e vídeo (Linux)..."
# Adicionado libgl1 para o OpenCV funcionar no Linux sem erros de libGL.so
sudo apt-get update && sudo apt-get install -y libportaudio2 portaudio19-dev libgl1

echo "Limpando qualquer ambiente corrompido..."
rm -rf .venv

echo "Fixando a versão do Python estritamente no 3.12..."
uv python pin 3.12

echo "Adicionando dependências do projeto e montando a venv..."
# Adicionamos as ferramentas de áudio, IA, visão computacional e captura de tela
uv add sounddevice numpy faster-whisper transformers opencv-python easyocr mss pillow

echo "Injetando PyTorch otimizado para GPU (CUDA 12.1)..."
# Injetamos no final para garantir que o uv não destrua a compatibilidade CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "✅ Setup concluído com sucesso!"
echo "👉 Ative o ambiente rodando: source .venv/bin/activate"