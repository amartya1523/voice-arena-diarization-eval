"""Diarization systems used in the pilot run.

All three share the same open segmentation model (pyannote segmentation-3.0,
ONNX export) and differ only in the speaker-embedding model, which is the part
that decides "is this the same voice?". They run fully offline on CPU through
sherpa-onnx, so the pilot is reproducible on a laptop with no API keys.
"""
import pathlib
import sherpa_onnx

ROOT = pathlib.Path(__file__).resolve().parents[1]
M = ROOT / "models"
SEG = M / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"

SYSTEMS = {
    # id: (display name, embedding model file)
    "wespeaker": ("pyannote-seg-3.0 + WeSpeaker ResNet34-LM", "wespeaker_en_voxceleb_resnet34_LM.onnx"),
    "titanet": ("pyannote-seg-3.0 + NeMo TitaNet-small", "nemo_en_titanet_small.onnx"),
    "eres2net": ("pyannote-seg-3.0 + 3D-Speaker ERes2Net-base", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"),
}


def build(system_id, threshold=0.5, num_speakers=-1, threads=int(__import__("os").environ.get("THREADS", 4))):
    _, emb = SYSTEMS[system_id]
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(M / emb), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    assert cfg.validate(), "invalid diarization config"
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)
