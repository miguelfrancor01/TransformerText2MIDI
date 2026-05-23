"""
app.py - Interfaz web de text2midi con Streamlit.

Diseño de 3 columnas inspirado en aplicaciones de música modernas:
    - Sidebar izquierdo : gris oscuro, info del modelo.
    - Panel central     : fondo blanco/crema, card hero naranja para el input.
    - Panel derecho     : gris claro, reproductor MIDI.

Uso:
    streamlit run app.py
"""

import time

import streamlit as st

from src.generator import generate
from src.midi_utils import (
    get_midi_player_html,
    midi_to_base64,
    midi_to_mp3,
    tokens_to_approx_duration,
)
from src.model_loader import load_model

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Text2MIDI — Symbolic Music Generation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Poppins', sans-serif !important; }

.stApp { background-color: #f0ede8 !important; }
.main .block-container { padding-top: 1.5rem; max-width: 100%; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #2e2e2e !important;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
section[data-testid="stSidebar"] a { color: #e8855a !important; }

/* Panel central blanco */
[data-testid="column"]:nth-child(1) {
    background-color: #ffffff;
    border-radius: 20px;
    padding: 24px !important;
}

/* Panel derecho gris claro */
[data-testid="column"]:nth-child(2) {
    background-color: #f7f4f0;
    border-radius: 20px;
    padding: 24px !important;
}

/* ── 1. Input delicado ── */
.stTextArea textarea {
    background-color: #f8f6f3 !important;
    border: 1px solid rgba(190, 92, 43, 0.18) !important;
    color: #2a2a2a !important;
    border-radius: 14px !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
    box-shadow: 0 1px 4px rgba(190,92,43,0.06) !important;
    transition: border 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
    border: 1px solid rgba(190,92,43,0.45) !important;
    box-shadow: 0 2px 8px rgba(190,92,43,0.1) !important;
    background-color: #fff !important;
}
.stTextArea textarea::placeholder { color: rgba(190,92,43,0.35) !important; }

/* Botones ejemplo */
.stButton > button {
    background-color: rgba(255,255,255,0.7) !important;
    border: 1px solid #e0d8d0 !important;
    color: #555 !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #be5c2b !important;
    color: #be5c2b !important;
    background-color: rgba(190,92,43,0.05) !important;
}

/* Botón primario */
.stButton > button[kind="primary"] {
    background-color: #be5c2b !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 25px !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #d4693a !important;
    box-shadow: 0 4px 15px rgba(190,92,43,0.3) !important;
}

/* Botón secondary */
.stButton > button[kind="secondary"] {
    background-color: #f0ede8 !important;
    border: 1px solid #ddd !important;
    color: #888 !important;
    border-radius: 25px !important;
}

/* Slider y selectbox */
.stSlider label { color: #888 !important; font-size: 12px !important; }
.stSelectbox label { color: #888 !important; font-size: 12px !important; }
.stSelectbox [data-baseweb="select"] > div {
    background-color: #f8f6f3 !important;
    border: 1px solid rgba(190,92,43,0.18) !important;
    border-radius: 12px !important;
    color: #333 !important;
}

/* ── 2. Barra de progreso amarilla ── */
.stProgress > div > div {
    background-color: #e8b84b !important;
    border-radius: 4px;
}
.stProgress { background-color: #e8e0d8 !important; border-radius: 4px; }
/* Texto del progress en naranja sin fondo */
[data-testid="stProgressBar"] + div p,
.stProgress p {
    color: #be5c2b !important;
    font-weight: 500 !important;
    background: transparent !important;
}

/* ── 3. Alert success → naranja claro ── */
[data-testid="stAlert"] {
    background-color: #fdf3ec !important;
    border: 1px solid rgba(190,92,43,0.2) !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] p {
    color: #c96d3a !important;
    font-weight: 500 !important;
}
[data-testid="stAlert"] svg { color: #c96d3a !important; }

/* Warning */
.stWarning {
    background-color: #fff8f0 !important;
    border-color: rgba(190,92,43,0.3) !important;
}

/* Download button */
.stDownloadButton > button {
    background-color: #fff !important;
    border: 1.5px solid #be5c2b !important;
    color: #be5c2b !important;
    border-radius: 25px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background-color: #be5c2b !important;
    color: #fff !important;
}

/* Métricas */
[data-testid="stMetric"] {
    background: #fff;
    border-radius: 12px;
    padding: 12px 16px;
    border: 1px solid rgba(190,92,43,0.12);
}
[data-testid="stMetric"] label { color: #aaa !important; font-size: 11px !important; }
[data-testid="stMetricValue"] {
    color: #1a1a1a !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f0ede8; }
::-webkit-scrollbar-thumb { background: #be5c2b; border-radius: 2px; }
button[data-testid="BaseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Carga del modelo (cacheada) ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    """
    Carga el modelo text2midi y lo almacena en caché de Streamlit.

    Returns:
        Tuple: (model, text_tokenizer, remi_tokenizer, device)
    """
    return load_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Renderiza el sidebar oscuro con información del modelo."""
    with st.sidebar:
        st.markdown("""
        <div style="padding:24px 0 12px;">
            <div style="font-size:10px;letter-spacing:3px;color:#e8855a;
                        text-transform:uppercase;font-weight:700;">
                Datos Secuenciales
            </div>
            <div style="font-size:28px;font-weight:800;color:#fff;
                        line-height:1.2;margin-top:8px;">Text2MIDI</div>
            <div style="font-size:12px;color:#777;margin-top:4px;">
                Symbolic Music Generation
            </div>
        </div>
        <hr style="border-color:#444;margin:16px 0;">
        """, unsafe_allow_html=True)

        nav_items = ["Browse", "Songs", "Albums", "Artists"]
        st.markdown(
            '<div style="font-size:11px;color:#666;margin-bottom:8px;'
            'letter-spacing:1px;">Library</div>',
            unsafe_allow_html=True,
        )
        for item in nav_items:
            active = item == "Browse"
            bg = "background:#444;color:#fff;" if active else "color:#aaa;"
            weight = "600" if active else "400"
            st.markdown(f"""
            <div style="padding:8px 14px;border-radius:8px;{bg}
                        font-size:13px;font-weight:{weight};margin-bottom:4px;">
                {item}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:11px;color:#666;margin:16px 0 8px;'
            'letter-spacing:1px;">Model Info</div>',
            unsafe_allow_html=True,
        )

        specs = [
            ("Encoder",  "FLAN-T5 Base", "Frozen weights"),
            ("Decoder",  "Transformer",  "18 layers · 8 heads"),
            ("Params",   "272M total",   "159M trainable"),
            ("Vocab",    "REMI+",        "524 tokens"),
            ("Context",  "2,048 tokens", "≈ 4-5 min music"),
        ]
        for title, val, sub in specs:
            st.markdown(f"""
            <div style="margin-bottom:6px;padding:9px 12px;background:#3a3a3a;
                        border-radius:8px;border-left:2px solid #e8855a;">
                <div style="font-size:9px;color:#666;text-transform:uppercase;
                            letter-spacing:1px;">{title}</div>
                <div style="font-size:13px;color:#e8e8e8;font-weight:600;
                            margin-top:1px;">{val}</div>
                <div style="font-size:10px;color:#666;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <hr style="border-color:#444;margin:16px 0;">
        <div style="font-size:12px;line-height:2.2;">
            <a href="https://arxiv.org/abs/2412.16526" target="_blank"
               style="color:#e8855a;text-decoration:none;">📄 arXiv 2412.16526</a><br>
            <a href="https://github.com/AMAAI-Lab/Text2midi" target="_blank"
               style="color:#e8855a;text-decoration:none;">💻 GitHub</a><br>
            <a href="https://huggingface.co/amaai-lab/text2midi" target="_blank"
               style="color:#e8855a;text-decoration:none;">🤗 HuggingFace</a>
        </div>
        """, unsafe_allow_html=True)


# ── Panel central ─────────────────────────────────────────────────────────────

def render_generator_panel() -> None:
    """Renderiza el panel central blanco con card hero naranja."""

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #be5c2b 0%, #d4693a 50%, #c8602e 100%);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    ">
        <div style="position:absolute;top:-20px;right:-20px;width:160px;height:160px;
                    background:rgba(255,255,255,0.08);border-radius:50%;"></div>
        <div style="position:absolute;bottom:-40px;right:40px;width:100px;height:100px;
                    background:rgba(255,255,255,0.05);border-radius:50%;"></div>
        <div style="font-size:10px;letter-spacing:3px;color:rgba(255,255,255,0.7);
                    text-transform:uppercase;font-weight:600;margin-bottom:6px;">
            Curated by AI
        </div>
        <div style="font-size:26px;font-weight:800;color:#fff;line-height:1.2;">
            Generate Music
        </div>
        <div style="font-size:13px;color:rgba(255,255,255,0.75);margin-top:6px;">
            Describe your music and the model will generate a MIDI file.<br>
            Include tempo, key, instruments and mood for best results.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px;font-weight:600;color:#555;
                margin-bottom:10px;letter-spacing:0.5px;">
        Quick examples
    </div>
    """, unsafe_allow_html=True)

    examples = [
        "A happy pop song in A major at 120 BPM with piano and guitar",
        "A dark cinematic piece in C minor with strings and brass",
        "A relaxing jazz song with saxophone and piano at 90 BPM",
        "An energetic electronic track with synth bass at 140 BPM",
    ]
    ex_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with ex_cols[i % 2]:
            if st.button(f"♪  {ex[:34]}...", key=f"ex{i}"):
                st.session_state["caption_val"] = ex
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    caption = st.text_area(
        "caption",
        value=st.session_state.get("caption_val", ""),
        placeholder="Describe the music you want to generate...",
        height=110,
        label_visibility="collapsed",
    )

    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        temperature = st.slider(
            "Creativity",
            min_value=0.5, max_value=2.0, value=1.0, step=0.1,
            help="Higher = more creative and varied",
        )
    with ctrl2:
        max_tokens = st.selectbox(
            "Length",
            options=[500, 1000, 2000],
            index=1,
            help="~500 tokens ≈ 1 min",
        )

    b1, b2 = st.columns([3, 1])
    with b1:
        gen_btn = st.button(
            "🎵  Generate MIDI", type="primary", use_container_width=True
        )
    with b2:
        clr_btn = st.button("✕  Clear", use_container_width=True)

    if clr_btn:
        for key in ("caption_val", "midi_output", "mp3_output", "mp3_error", "elapsed", "n_tokens"):
            st.session_state.pop(key, None)
        st.rerun()

    progress_box = st.empty()
    status_box = st.empty()

    if gen_btn and caption.strip():
        with st.spinner("Loading model..."):
            model, text_tok, r_tok, device = get_model()

        pb = progress_box.progress(0, text="Processing text with FLAN-T5...")
        pb.progress(15, text="Generating MIDI tokens autoregressively...")
        t0 = time.time()

        midi_bytes, n_tokens = generate(
            caption=caption,
            model=model,
            text_tokenizer=text_tok,
            r_tokenizer=r_tok,
            device=device,
            max_len=max_tokens,
            temperature=temperature,
        )

        elapsed = time.time() - t0
        # ── 2. Texto del progress en naranja, sin fondo naranja ──
        pb.progress(100, text="")
        status_box.markdown(
            f'<p style="color:#be5c2b;font-weight:500;font-size:14px;margin:4px 0;">'
            f'✓ Done in {elapsed:.1f}s &nbsp;·&nbsp; {n_tokens} tokens generated'
            f'</p>',
            unsafe_allow_html=True,
        )

        st.session_state["midi_output"] = midi_bytes
        st.session_state["elapsed"] = elapsed
        st.session_state["n_tokens"] = n_tokens
        # Limpiar MP3 cacheado para que se regenere con el nuevo MIDI
        st.session_state.pop("mp3_output", None)
        st.session_state.pop("mp3_error", None)

    elif gen_btn:
        status_box.warning("Please write a description first.")


# ── Panel derecho ─────────────────────────────────────────────────────────────

def render_player_panel() -> None:
    """Renderiza el panel derecho con el reproductor MIDI."""

    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="display:flex;gap:3px;align-items:flex-end;">
            <div style="width:3px;height:14px;background:#be5c2b;border-radius:2px;"></div>
            <div style="width:3px;height:20px;background:#be5c2b;border-radius:2px;"></div>
            <div style="width:3px;height:11px;background:#be5c2b;border-radius:2px;"></div>
            <div style="width:3px;height:17px;background:#be5c2b;border-radius:2px;"></div>
        </div>
        <div style="font-size:16px;font-weight:700;color:#1a1a1a;">Now Playing</div>
    </div>
    """, unsafe_allow_html=True)

    if "midi_output" not in st.session_state:
        st.markdown("""
        <div style="height:300px;background:#ede9e4;border-radius:16px;
                    display:flex;flex-direction:column;align-items:center;
                    justify-content:center;margin-bottom:16px;">
            <div style="font-size:52px;color:#d4a080;margin-bottom:12px;
                        font-style:normal;line-height:1;">&#9835;</div>
            <div style="font-size:13px;color:#aaa;font-weight:500;">
                Your music will appear here
            </div>
            <div style="font-size:11px;color:#ccc;margin-top:4px;">
                Write a description and press Generate
            </div>
        </div>
        """, unsafe_allow_html=True)

        for title, artist in [("Waiting...", "—"), ("Ready to generate", "text2midi")]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        padding:10px 0;border-bottom:1px solid #e8e2db;">
                <div style="width:36px;height:36px;background:#e0dbd5;
                            border-radius:6px;flex-shrink:0;"></div>
                <div style="flex:1;">
                    <div style="font-size:13px;font-weight:500;color:#aaa;">{title}</div>
                    <div style="font-size:11px;color:#ccc;margin-top:1px;">{artist}</div>
                </div>
                <div style="font-size:12px;color:#ccc;">—:——</div>
            </div>
            """, unsafe_allow_html=True)
        return

    midi_bytes = st.session_state["midi_output"]
    midi_b64 = midi_to_base64(midi_bytes)

    # ── 4. Card sin emoji colorido, símbolo tipográfico naranja ──
    st.markdown("""
    <div style="background:#ede9e4;border-radius:16px;padding:16px;margin-bottom:16px;">
        <div style="width:100%;height:120px;
                    background:linear-gradient(135deg,#be5c2b,#d4693a);
                    border-radius:10px;margin-bottom:12px;
                    display:flex;align-items:center;justify-content:center;">
            <span style="font-size:48px;color:rgba(255,255,255,0.55);
                         font-family:serif;line-height:1;">&#9834;</span>
        </div>
        <div style="font-size:15px;font-weight:700;color:#1a1a1a;">AI Generated</div>
        <div style="font-size:12px;color:#888;margin-top:2px;">text2midi · REMI+</div>
    </div>
    """, unsafe_allow_html=True)

    if "elapsed" in st.session_state:
        m1, m2 = st.columns(2)
        m1.metric("Time", f"{st.session_state['elapsed']:.1f}s")
        m2.metric(
            "Est. duration",
            tokens_to_approx_duration(st.session_state.get("n_tokens", 0)),
        )

    # Botones de descarga MIDI y MP3
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="⬇  Download MIDI",
            data=midi_bytes,
            file_name="text2midi_output.mid",
            mime="audio/midi",
            use_container_width=True,
        )
    with dl2:
        if "mp3_output" not in st.session_state:
            with st.spinner("Converting to MP3..."):
                try:
                    st.session_state["mp3_output"] = midi_to_mp3(midi_bytes)
                except Exception as err:
                    st.session_state["mp3_output"] = None
                    st.session_state["mp3_error"] = str(err)

        if st.session_state.get("mp3_output"):
            st.download_button(
                label="⬇  Download MP3",
                data=st.session_state["mp3_output"],
                file_name="text2midi_output.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div style="padding:8px;text-align:center;font-size:12px;'
                'color:#ccc;border:1px solid #e0dbd5;border-radius:25px;">'
                'MP3 unavailable</div>',
                unsafe_allow_html=True,
            )

    # Reproductor con borde gris y colores cálidos en waterfall
    player_html = get_midi_player_html(midi_b64, visualizer_height=180)
    st.components.v1.html(player_html, height=380, scrolling=False)


# ── Layout principal ──────────────────────────────────────────────────────────

def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    render_sidebar()
    col_main, col_player = st.columns([3, 2], gap="medium")
    with col_main:
        render_generator_panel()
    with col_player:
        render_player_panel()


if __name__ == "__main__":
    main()