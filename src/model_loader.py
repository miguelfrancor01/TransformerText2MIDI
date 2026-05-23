"""
model_loader.py - Carga del modelo text2midi desde HuggingFace Hub.

Este módulo gestiona la descarga de pesos preentrenados y la inicialización
del modelo Transformer encoder-decoder descrito en el paper:
    "Text2midi: Generating Symbolic Music from Captions" (arXiv: 2412.16526)

El encoder es FLAN-T5 base (pesos congelados) y el decoder es un Transformer
autoregresivo de 18 capas que genera tokens REMI+ condicionados al texto.
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Tuple

import torch
from huggingface_hub import hf_hub_download
from transformers import T5Tokenizer

# Añadir el directorio raíz al path para importar el modelo original
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.transformer_model import Transformer  # noqa: E402

# ── Constantes ────────────────────────────────────────────────────────────────

REPO_ID = "amaai-lab/text2midi"
MODEL_FILENAME = "pytorch_model.bin"
VOCAB_FILENAME = "vocab_remi.pkl"
FLAN_T5_MODEL = "google/flan-t5-base"

# Hiperparámetros del modelo según el paper (Sección: Model Configuration)
MODEL_CONFIG = {
    "d_model": 768,           # Dimensión de embeddings
    "nhead": 8,               # Cabezas de atención
    "max_len": 2048,          # Longitud máxima de secuencia
    "num_decoder_layers": 18, # Capas del decoder
    "dim_feedforward": 1024,  # Dimensión de la FFN
    "use_moe": False,         # Sin Mixture of Experts
    "num_experts": 8,         # Expertos (inactivo si use_moe=False)
}


# ── Funciones públicas ────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Detecta y retorna el dispositivo de cómputo disponible.

    Prioridad: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU.

    Returns:
        torch.device: Dispositivo seleccionado automáticamente.

    Examples:
        >>> device = get_device()
        >>> print(device)
        cuda
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_weights() -> Tuple[str, str]:
    """
    Descarga pesos del modelo y vocabulario desde HuggingFace Hub.

    Los archivos se almacenan en caché local automáticamente. La descarga
    solo ocurre la primera vez (~1.2 GB en total).

    Returns:
        Tuple[str, str]: Rutas locales de (pytorch_model.bin, vocab_remi.pkl).

    Raises:
        requests.exceptions.ConnectionError: Sin conexión a internet.
        huggingface_hub.utils.EntryNotFoundError: Archivo no encontrado.

    Examples:
        >>> model_path, vocab_path = download_weights()
        >>> print(model_path)
        /home/user/.cache/huggingface/hub/.../pytorch_model.bin
    """
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename=VOCAB_FILENAME)
    return model_path, vocab_path


def load_remi_tokenizer(vocab_path: str):
    """
    Carga el tokenizador REMI+ desde un archivo pickle.

    REMI+ extiende REMI (Revamped MIDI-derived events) añadiendo tokens
    de programa (instrumento) y métrica, permitiendo representar
    composiciones multi-pista y multi-instrumento.

    Tokens soportados: Bar, Position, Tempo, TimeSignature,
    Program, Pitch, Velocity, Duration.

    Args:
        vocab_path (str): Ruta al archivo vocab_remi.pkl.

    Returns:
        MidiTok tokenizer: Tokenizador REMI+ con vocabulario de 524 tokens.

    Examples:
        >>> r_tok = load_remi_tokenizer("artifacts/vocab_remi.pkl")
        >>> print(len(r_tok))
        524
    """
    with open(vocab_path, "rb") as file:
        return pickle.load(file)


def build_model(vocab_size: int, device: torch.device) -> Transformer:
    """
    Instancia el modelo Transformer con la configuración del paper.

    Arquitectura (encoder-decoder):
        - Encoder: T5EncoderModel (FLAN-T5 base, pesos congelados)
        - Decoder: 18 capas TransformerDecoderLayer con Flash Attention
        - Proyección: Linear(d_model=768, vocab_size)

    Args:
        vocab_size (int): Tamaño del vocabulario REMI+ (524).
        device (torch.device): Dispositivo donde se alojará el modelo.

    Returns:
        Transformer: Instancia del modelo sin pesos cargados.

    Examples:
        >>> model = build_model(524, torch.device("cuda"))
        >>> print(sum(p.numel() for p in model.parameters()))
        272000000
    """
    return Transformer(
        n_vocab=vocab_size,
        d_model=MODEL_CONFIG["d_model"],
        nhead=MODEL_CONFIG["nhead"],
        max_len=MODEL_CONFIG["max_len"],
        num_decoder_layers=MODEL_CONFIG["num_decoder_layers"],
        dim_feedforward=MODEL_CONFIG["dim_feedforward"],
        use_moe=MODEL_CONFIG["use_moe"],
        num_experts=MODEL_CONFIG["num_experts"],
        device=device,
    )


def load_model() -> Tuple:
    """
    Carga el modelo text2midi completo listo para inferencia.

    Pipeline completo de carga:
        1. Detecta el dispositivo disponible (GPU/CPU).
        2. Descarga pesos y vocabulario desde HuggingFace Hub.
        3. Carga el tokenizador REMI+.
        4. Instancia y carga el modelo Transformer.
        5. Inicializa el tokenizador de texto FLAN-T5.

    Returns:
        Tuple[Transformer, T5Tokenizer, REMITokenizer, torch.device]:
            - model          : Transformer en modo eval().
            - text_tokenizer : T5Tokenizer para procesar captions.
            - remi_tokenizer : Tokenizador REMI+ para decodificar MIDI.
            - device         : Dispositivo donde vive el modelo.

    Examples:
        >>> model, tok, r_tok, device = load_model()
        >>> print(f"Modelo en {device} con {len(r_tok)} tokens")
        Modelo en cuda con 524 tokens
    """
    device = get_device()

    model_path, vocab_path = download_weights()
    r_tokenizer = load_remi_tokenizer(vocab_path)

    vocab_size = len(r_tokenizer)
    model = build_model(vocab_size, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    text_tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_MODEL)

    return model, text_tokenizer, r_tokenizer, device
