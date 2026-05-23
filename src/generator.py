"""
generator.py - Pipeline de inferencia texto → MIDI.

Implementa el flujo completo de generación del modelo text2midi:

    1. Tokenización del caption con FLAN-T5 (encoder).
    2. Generación autoregresiva de tokens REMI+ (decoder).
    3. Decodificación de tokens a objeto MIDI (MidiTok).
    4. Serialización del MIDI a bytes o archivo.

El mecanismo central es la generación autoregresiva: el decoder predice
un token a la vez, condicionado en los hidden states del encoder
(cross-attention) y en los tokens ya generados (masked self-attention).

Fórmula de atención: Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
    - Q (Query)  : del token MIDI actual en el decoder.
    - K, V (Keys/Values): de los hidden states HT del encoder.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ── Tokenización ──────────────────────────────────────────────────────────────

def tokenize_caption(
    caption: str,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokeniza un caption de texto para el encoder FLAN-T5.

    Convierte la descripción musical a tensores de IDs de tokens y
    máscaras de atención, aplicando padding y truncación automáticos.

    Args:
        caption (str): Descripción textual de la música.
            Puede incluir tempo, tonalidad, instrumentos, estado de ánimo.
            Ej: "A happy pop song in A major at 120 BPM with piano."
        tokenizer: T5Tokenizer de HuggingFace.
        device (torch.device): Dispositivo donde se moverán los tensores.

    Returns:
        Tuple[Tensor, Tensor]:
            - input_ids     : Shape (1, seq_len). IDs de tokens del texto.
            - attention_mask: Shape (1, seq_len). 1 = token real, 0 = padding.

    Examples:
        >>> ids, mask = tokenize_caption("A jazz song", tokenizer, device)
        >>> print(ids.shape)
        torch.Size([1, 7])
    """
    inputs = tokenizer(
        caption,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = nn.utils.rnn.pad_sequence(
        inputs.input_ids,
        batch_first=True,
        padding_value=0,
    ).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(
        inputs.attention_mask,
        batch_first=True,
        padding_value=0,
    ).to(device)
    return input_ids, attention_mask


# ── Generación autoregresiva ──────────────────────────────────────────────────

def generate_midi_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_len: int = 2000,
    temperature: float = 1.0,
) -> list:
    """
    Genera una secuencia de tokens REMI+ de forma autoregresiva.

    En cada paso t, el decoder predice el token t+1 dado:
        - Los hidden states HT del encoder (vía cross-attention).
        - Todos los tokens generados hasta t (vía masked self-attention).

    La distribución de salida es softmax(logits / temperature):
        - temperature < 1.0 → distribución más puntiaguda (conservador).
        - temperature = 1.0 → distribución original del modelo.
        - temperature > 1.0 → distribución más plana (más variado).

    Aproximadamente: 500 tokens ≈ 1 min de música.

    Args:
        model: Transformer text2midi en modo eval.
        input_ids (Tensor): IDs de tokens del caption. Shape (1, seq_len).
        attention_mask (Tensor): Máscara de atención del caption.
        max_len (int): Número máximo de tokens REMI+ a generar.
        temperature (float): Factor de escala del softmax.

    Returns:
        list[int]: Secuencia de índices de tokens REMI+ generados.

    Examples:
        >>> tokens = generate_midi_tokens(model, ids, mask, max_len=1000)
        >>> print(f"{len(tokens)} tokens generados")
        1000 tokens generados
    """
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask,
            max_len=max_len,
            temperature=temperature,
        )
    return output[0].tolist()


# ── Decodificación MIDI ───────────────────────────────────────────────────────

def decode_to_midi(tokens: list, r_tokenizer):
    """
    Decodifica una secuencia de tokens REMI+ a un objeto MIDI.

    Usa el tokenizador REMI+ de MidiTok para convertir índices enteros
    a eventos musicales: notas, tiempos, instrumentos y dinámicas.

    Args:
        tokens (list[int]): Secuencia de tokens REMI+ generados.
        r_tokenizer: Tokenizador REMI+ de MidiTok.

    Returns:
        symusic.Score: Objeto MIDI con la música decodificada.

    Examples:
        >>> midi = decode_to_midi(tokens, r_tokenizer)
        >>> midi.dump_midi("output.mid")
    """
    return r_tokenizer.decode(tokens)


def midi_to_bytes(midi_obj) -> bytes:
    """
    Serializa un objeto MIDI a bytes.

    Escribe el MIDI a un archivo temporal, lee los bytes resultantes
    y limpia el archivo. Necesario porque dump_midi() solo acepta rutas.

    Args:
        midi_obj (symusic.Score): Objeto MIDI de symusic/MidiTok.

    Returns:
        bytes: Contenido binario del archivo MIDI.

    Examples:
        >>> data = midi_to_bytes(midi_obj)
        >>> print(f"{len(data)} bytes generados")
        4096 bytes generados
    """
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        midi_obj.dump_midi(tmp_path)
        with open(tmp_path, "rb") as file:
            return file.read()
    finally:
        os.unlink(tmp_path)


def save_midi(midi_obj, output_path: str) -> None:
    """
    Guarda un objeto MIDI en disco.

    Crea los directorios intermedios si no existen.

    Args:
        midi_obj (symusic.Score): Objeto MIDI de symusic/MidiTok.
        output_path (str): Ruta de destino del archivo MIDI.

    Returns:
        None

    Examples:
        >>> save_midi(midi_obj, "outputs/mi_cancion.mid")
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    midi_obj.dump_midi(output_path)


# ── Pipeline completo ─────────────────────────────────────────────────────────

def generate(
    caption: str,
    model,
    text_tokenizer,
    r_tokenizer,
    device: torch.device,
    max_len: int = 2000,
    temperature: float = 1.0,
    output_path: Optional[str] = None,
) -> Tuple[bytes, int]:
    """
    Pipeline completo de generación: texto → MIDI.

    Encadena todos los pasos de inferencia en una sola llamada:
        1. tokenize_caption()      → input_ids, attention_mask
        2. generate_midi_tokens()  → tokens REMI+
        3. decode_to_midi()        → objeto MIDI
        4. save_midi()             → (opcional) archivo en disco
        5. midi_to_bytes()         → bytes para descarga/reproducción

    Args:
        caption (str): Descripción textual de la música a generar.
        model: Modelo text2midi en modo eval.
        text_tokenizer: T5Tokenizer para tokenizar el caption.
        r_tokenizer: Tokenizador REMI+ para decodificar tokens.
        device (torch.device): Dispositivo de cómputo del modelo.
        max_len (int): Máximo de tokens REMI+ a generar (default: 2000).
        temperature (float): Temperatura de muestreo (default: 1.0).
        output_path (str, optional): Ruta para guardar el MIDI en disco.

    Returns:
        Tuple[bytes, int]:
            - midi_bytes : Contenido binario del MIDI generado.
            - n_tokens   : Número de tokens REMI+ generados.

    Examples:
        >>> midi_bytes, n_tokens = generate(
        ...     caption="A happy jazz song with piano at 90 BPM",
        ...     model=model,
        ...     text_tokenizer=tokenizer,
        ...     r_tokenizer=r_tokenizer,
        ...     device=device,
        ...     max_len=1000,
        ...     temperature=0.9,
        ...     output_path="outputs/jazz.mid",
        ... )
        >>> print(f"Generados {n_tokens} tokens — {len(midi_bytes)} bytes")
    """
    # Paso 1: Tokenizar el caption de texto
    input_ids, attention_mask = tokenize_caption(caption, text_tokenizer, device)

    # Paso 2: Generar tokens REMI+ autoregresivamente
    tokens = generate_midi_tokens(
        model, input_ids, attention_mask, max_len, temperature
    )

    # Paso 3: Decodificar tokens a objeto MIDI
    midi_obj = decode_to_midi(tokens, r_tokenizer)

    # Paso 4 (opcional): Guardar en disco
    if output_path:
        save_midi(midi_obj, output_path)

    # Paso 5: Serializar a bytes
    midi_bytes = midi_to_bytes(midi_obj)

    return midi_bytes, len(tokens)
