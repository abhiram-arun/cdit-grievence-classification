#!/bin/bash
set -e

echo "[*] Cloning IndicTrans2..."
git clone https://github.com/AI4Bharat/IndicTrans2.git

echo "[*] Cloning and installing IndicTransToolkit..."
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
pip install --editable ./
cd ..

echo "[*] Installing Python requirements..."
pip install -r requirements.txt

echo "[*] Downloading NLTK tokenizer..."
python3 -c "import nltk; nltk.download('punkt')"

echo "[✓] All set up!"
