"""
inference.py - Script de inferencia CLI para text2midi.

Genera un archivo MIDI a partir de una descripción textual usando
el modelo text2midi preentrenado. Los pesos se descargan automáticamente
desde HuggingFace Hub en la primera ejecución (~1.2 GB).

Uso:
    python inference.py --caption "A happy pop song in A major"
    python inference.py --caption "A jazz song" --output salida.mid
    python inference.py --caption "Dark cinematic" --max_len 1000 --temperature 0.9

Argumentos:
    --caption     Descripción textual de la música (requerido).
    --output      Ruta del MIDI de salida (default: output.mid).
    --max_len     Tokens MIDI a generar, ~500 tokens ≈ 1 min (default: 2000).
    --temperature Temperatura del softmax, >1 más creativo (default: 1.0).
"""

import argparse
import time
from pathlib import Path

from src.generator import generate
from src.model_loader import load_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Objeto con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Genera un archivo MIDI desde una descripción textual.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--caption",
        type=str,
        required=True,
        help="Descripción textual de la música a generar.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mid",
        help="Ruta del archivo MIDI de salida (default: output.mid).",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2000,
        help="Número máximo de tokens MIDI a generar (default: 2000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperatura de muestreo del softmax (default: 1.0).",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Punto de entrada principal del script de inferencia.

    Carga el modelo, ejecuta el pipeline de generación y guarda el MIDI.
    """
    args = parse_args()

    print("\n── Text2MIDI Inference ─────────────────────────────────")
    print(f"  Caption     : {args.caption}")
    print(f"  Output      : {args.output}")
    print(f"  Max tokens  : {args.max_len}  (~{args.max_len // 500} min aprox.)")
    print(f"  Temperature : {args.temperature}")
    print("────────────────────────────────────────────────────────\n")

    print("Cargando modelo...")
    model, text_tokenizer, r_tokenizer, device = load_model()
    print(f"Modelo cargado en {device}\n")

    print("Generando MIDI...")
    t0 = time.time()

    _, n_tokens = generate(
        caption=args.caption,
        model=model,
        text_tokenizer=text_tokenizer,
        r_tokenizer=r_tokenizer,
        device=device,
        max_len=args.max_len,
        temperature=args.temperature,
        output_path=args.output,
    )

    elapsed = time.time() - t0
    output_abs = Path(args.output).resolve()

    print(f"\n✓  {n_tokens} tokens generados en {elapsed:.1f}s")
    print(f"✓  Guardado en: {output_abs}\n")


if __name__ == "__main__":
    main()
