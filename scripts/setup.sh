#!/usr/bin/env bash
# Reproduce the pilot from scratch: python env, models, scorer, audio.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

# Models (ONNX exports published by the sherpa-onnx project)
mkdir -p models && cd models
REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
curl -sfLO $REL/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
for f in wespeaker_en_voxceleb_resnet34_LM.onnx nemo_en_titanet_small.onnx \
         3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx; do
  curl -sfLO $REL/speaker-recongition-models/$f
done
cd ..

# Independent scorer (DIHARD's dscore)
git clone -q --depth 1 https://github.com/nryant/dscore.git tools/dscore

# Audio: public parquet mirrors of AMI (headset mix + far-field) and VoxConverse test
mkdir -p data/raw && cd data/raw
HF=https://huggingface.co/datasets/diarizers-community
curl -sfL -o ami_ihm_test0.parquet $HF/ami/resolve/main/ihm/test-00000-of-00003.parquet
curl -sfL -o ami_sdm_test0.parquet $HF/ami/resolve/main/sdm/test-00000-of-00003.parquet
curl -sfL -o vox_test0.parquet     $HF/voxconverse/resolve/main/data/test-00004-of-00011.parquet
cd ../..

.venv/bin/python eval/prepare.py testset
.venv/bin/python eval/prepare.py devset
echo "setup done"
