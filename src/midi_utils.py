"""
midi_utils.py - Utilidades de serialización y reproducción MIDI.

Funciones auxiliares para codificar, formatear y visualizar archivos MIDI
generados por el modelo text2midi en la interfaz Streamlit.
Incluye conversión MIDI → MP3 vía FluidSynth y pydub.
"""

import base64
import os
import tempfile


def midi_to_base64(midi_bytes: bytes) -> str:
    """
    Codifica bytes MIDI en string base64 para uso en HTML.

    Args:
        midi_bytes (bytes): Contenido binario del archivo MIDI.

    Returns:
        str: Representación base64 del MIDI.

    Examples:
        >>> b64 = midi_to_base64(midi_bytes)
        >>> src = f"data:audio/midi;base64,{b64}"
    """
    return base64.b64encode(midi_bytes).decode("utf-8")


def midi_to_mp3(midi_bytes: bytes) -> bytes:
    """
    Convierte bytes MIDI a bytes MP3 usando FluidSynth y pydub.

    Pipeline de conversión:
        1. Escribe el MIDI en un archivo temporal.
        2. FluidSynth renderiza MIDI → WAV usando el soundfont GM.
        3. pydub convierte WAV → MP3.
        4. Retorna los bytes del MP3 y limpia archivos temporales.

    Requiere instalación previa:
        sudo apt install fluidsynth fluid-soundfont-gm ffmpeg -y
        pip install midi2audio pydub

    Args:
        midi_bytes (bytes): Contenido binario del archivo MIDI.

    Returns:
        bytes: Contenido binario del archivo MP3 resultante.

    Raises:
        FileNotFoundError: Si el soundfont no está instalado.
        RuntimeError: Si FluidSynth falla durante la conversión.

    Examples:
        >>> mp3_bytes = midi_to_mp3(midi_bytes)
        >>> with open("output.mp3", "wb") as f:
        ...     f.write(mp3_bytes)
    """
    from midi2audio import FluidSynth
    from pydub import AudioSegment

    soundfont = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

    # Archivos temporales
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp.write(midi_bytes)
        midi_path = tmp.name

    wav_path = midi_path.replace(".mid", ".wav")
    mp3_path = midi_path.replace(".mid", ".mp3")

    try:
        # Paso 1: MIDI → WAV con FluidSynth
        fs = FluidSynth(sound_font=soundfont)
        fs.midi_to_audio(midi_path, wav_path)

        # Paso 2: WAV → MP3 con pydub
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="192k")

        with open(mp3_path, "rb") as f:
            return f.read()
    finally:
        # Limpiar archivos temporales
        for path in [midi_path, wav_path, mp3_path]:
            if os.path.exists(path):
                os.unlink(path)


def get_midi_player_html(midi_b64: str, visualizer_height: int = 180) -> str:
    """
    Genera el HTML del reproductor MIDI con visualizador waterfall.

    Usa html-midi-player (Magenta.js + Tone.js) con colores cálidos
    que combinan con la paleta naranja/crema de la interfaz.

    Args:
        midi_b64 (str): MIDI codificado en base64.
        visualizer_height (int): Altura en píxeles del visualizador.

    Returns:
        str: HTML listo para st.components.v1.html().

    Examples:
        >>> html = get_midi_player_html(midi_b64, visualizer_height=200)
        >>> st.components.v1.html(html, height=400)
    """
    cdn = (
        "https://cdn.jsdelivr.net/combine/"
        "npm/tone@14/build/Tone.js,"
        "npm/@magenta/music@1.23.1/es6/core.js,"
        "npm/html-midi-player@1.5.0"
    )
    data_uri = f"data:audio/midi;base64,{midi_b64}"

    return f"""
    <script src="{cdn}"></script>
    <style>
        body {{ margin: 0; background: transparent; }}
        midi-player {{
            display: block;
            width: 100%;
            margin: 12px 0 8px;
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e0dbd5;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }}
        midi-visualizer svg {{
            height: {visualizer_height}px;
            width: 100%;
            display: block;
            background: #faf7f4;
            border-radius: 12px;
            border: 1px solid #e0dbd5;
        }}
    </style>
    <script>
    function styleNotes() {{
        const viz = document.querySelector('midi-visualizer');
        if (!viz || !viz.shadowRoot) {{ setTimeout(styleNotes, 300); return; }}
        const warmColors = [
            '#be5c2b','#d4693a','#c8602e','#e8855a',
            '#d97b4a','#e89a6a','#c45a28','#f0a070',
        ];
        const svg = viz.shadowRoot.querySelector('svg');
        if (!svg) {{ setTimeout(styleNotes, 300); return; }}
        const rects = svg.querySelectorAll('rect');
        rects.forEach((r, i) => {{
            const fill = r.getAttribute('fill') || '';
            if (fill === 'white' || fill === '#fff' ||
                fill === '#000000' || fill === 'black' || fill === '') return;
            r.setAttribute('fill', warmColors[i % warmColors.length]);
            r.setAttribute('opacity', '0.82');
        }});
        setTimeout(styleNotes, 800);
    }}
    document.addEventListener('DOMContentLoaded', () => setTimeout(styleNotes, 800));
    setTimeout(styleNotes, 1200);
    </script>
    <midi-player src="{data_uri}" sound-font visualizer="#viz"></midi-player>
    <midi-visualizer id="viz" type="waterfall" src="{data_uri}"></midi-visualizer>
    """


def format_duration(seconds: float) -> str:
    """
    Formatea una duración en segundos a string legible.

    Args:
        seconds (float): Duración en segundos.

    Returns:
        str: Duración formateada. Ej: "1m 23s" o "45.2s".

    Examples:
        >>> format_duration(83.5)
        '1m 23s'
        >>> format_duration(45.2)
        '45.2s'
    """
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def tokens_to_approx_duration(n_tokens: int) -> str:
    """
    Estima la duración aproximada del MIDI a partir del número de tokens.

    Basado en: ~500 tokens ≈ 1 minuto de música (paper text2midi).

    Args:
        n_tokens (int): Número de tokens REMI+ generados.

    Returns:
        str: Duración estimada formateada con prefijo ~.

    Examples:
        >>> tokens_to_approx_duration(2000)
        '~4m 0s'
        >>> tokens_to_approx_duration(500)
        '~1m 0s'
    """
    approx_seconds = (n_tokens / 500) * 60
    return f"~{format_duration(approx_seconds)}"