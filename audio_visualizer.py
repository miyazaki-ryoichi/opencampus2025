# app_mel_full.py
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")  # GUI不要の描画バックエンド
import matplotlib.pyplot as plt
import soundfile as sf
import gradio as gr

# ---- librosa 同梱サンプル（外部DL不要） ----
SAMPLE_FILES = {
    "Trumpet": librosa.ex("trumpet"),
    "Brahms": librosa.ex("brahms"),
    "Nutcracker": librosa.ex("nutcracker"),
    "Speech": librosa.ex("libri1"),
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 小ユーティリティ ----------
def _to_float_or_none(x):
    if x is None or isinstance(x, (int, float, np.number)):
        return None if x is None else float(x)
    try:
        s = str(x).strip()
        return None if s == "" else float(s)
    except Exception:
        return None

# ---------- 入力の読み込み ----------
def parse_audio(source, uploaded_file, recorded_file, sample_choice):
    if source == "録音":
        filepath = recorded_file               # gr.Audio(type="filepath")
    elif source == "ファイルアップロード":
        filepath = uploaded_file               # gr.File(type="filepath")
    elif source == "サンプル音":
        filepath = SAMPLE_FILES[sample_choice]
    else:
        raise ValueError("音声ソースが選択されていません。")

    if filepath is None:
        raise ValueError("ファイルが見つかりません。録音/アップロード/サンプルを確認してください。")

    y, sr = librosa.load(filepath, sr=None, mono=True)
    if y.size == 0:
        raise ValueError("読み込んだ音声が空でした。")
    m = float(np.max(np.abs(y)))
    if m > 0:
        y = (y / m).astype(np.float32, copy=False)
    return sr, y

# ---------- プロット＆ファイル作成（メルスペクトログラム） ----------
def make_plots(sr, y, t_start, t_end, limit_5k, show_f0):
    total_dur = len(y) / sr

    t_start = _to_float_or_none(t_start)
    t_end   = _to_float_or_none(t_end)
    if t_start is None or t_start < 0: t_start = 0.0
    if t_end   is None or t_end   > total_dur: t_end = total_dur
    if t_end <= t_start: t_end = min(t_start + 1.0, total_dur)  # 最低1秒

    s = int(t_start * sr)
    e = int(t_end   * sr)
    y_cut = y[s:e]

    # ---- 波形 ----
    fig_wave, axw = plt.subplots(figsize=(9, 2.2))
    tt = np.arange(len(y_cut))/sr + t_start
    axw.plot(tt, y_cut)
    axw.set_xlabel("Time [s]"); axw.set_ylabel("Amplitude"); axw.set_title("Waveform")
    fig_wave.tight_layout()

    # ---- メルスペクトログラム ----
    n_fft, hop = 2048, 512
    n_mels = 128
    fmax = 5000 if limit_5k else sr/2

    y_proc = y_cut if len(y_cut) >= n_fft else np.pad(y_cut, (0, max(0, n_fft-len(y_cut))), mode="constant")
    S = librosa.feature.melspectrogram(
        y=y_proc, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, fmin=0, fmax=fmax, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max) if S.size else np.zeros((1,1))

    ts = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop) + t_start
    mel_freqs_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=fmax)

    fig_spec, axs = plt.subplots(figsize=(9, 4.2))
    extent = [
        ts[0] if ts.size else t_start,
        ts[-1] if ts.size else t_end,
        mel_freqs_hz[0],
        mel_freqs_hz[-1],
    ]
    im = axs.imshow(S_db, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Frequency [Hz] (mel scale)")
    axs.set_title("Mel Spectrogram")
    fig_spec.colorbar(im, ax=axs, format="%+2.0f dB")

    # ---- F0（任意） ----
    if show_f0 and len(y_cut) > 0:
        try:
            f0 = librosa.yin(y_cut, fmin=50, fmax=min(1000, fmax-1), sr=sr,
                             frame_length=n_fft, hop_length=hop)
            t_f0 = np.arange(len(f0))*hop/sr + t_start
            mask = ~np.isnan(f0)
            if np.any(mask):
                axs.plot(t_f0[mask], f0[mask], linewidth=2.0)
        except Exception:
            pass

    fig_spec.tight_layout()

    # ---- 保存（WAV/PNG）----
    wav_path = os.path.join(OUTPUT_DIR, "output.wav")
    png_path = os.path.join(OUTPUT_DIR, "mel_spectrogram.png")
    sf.write(wav_path, y_cut, sr)
    fig_spec.savefig(png_path, dpi=150)

    # Figure は close しない（Gradio 側で描画）
    return fig_wave, fig_spec, wav_path, png_path, total_dur

# ---------- ハンドラ ----------
def process(source, uploaded_file, recorded_file, sample_choice,
            t_start, t_end, limit_5k, show_f0):
    sr, y = parse_audio(source, uploaded_file, recorded_file, sample_choice)
    fig_wave, fig_spec, wav_path, png_path, total_dur = make_plots(
        sr, y, t_start, t_end, limit_5k, show_f0
    )
    info = f"OK: 全長 {len(y)/sr:.2f}s → 表示 {t_start if t_start else 0:.2f}–{t_end if t_end else total_dur:.2f}s"
    return fig_wave, fig_spec, wav_path, png_path, info, (sr, y)

# ---------- 表示切替（ソース別UI） ----------
def toggle_inputs(src):
    return (
        gr.update(visible=(src == "ファイルアップロード")),  # uploaded_file
        gr.update(visible=(src == "録音")),                  # recorded_file
        gr.update(visible=(src == "サンプル音")),            # sample_choice
    )

# ---------- UI ----------
with gr.Blocks(title="スペクトログラム表示") as demo:
    gr.Markdown("## スペクトログラム表示")

    with gr.Row():
        with gr.Column():
            source = gr.Radio(
                ["録音", "ファイルアップロード", "サンプル音"],
                label="音声ソース", value="サンプル音"
            )

            uploaded_file = gr.File(label="音声ファイルをアップロード", type="filepath", visible=False)
            recorded_file = gr.Audio(sources=["microphone"], type="filepath", label="録音", visible=False)
            sample_choice = gr.Dropdown(
                choices=list(SAMPLE_FILES.keys()),
                value=list(SAMPLE_FILES.keys())[0],
                label="サンプル音を選択", visible=True
            )

            t_start = gr.Number(value=0.0, label="表示開始時間 [s]")
            t_end   = gr.Number(value=None, label="表示終了時間 [s]（空で末尾まで）")

            limit_5k = gr.Checkbox(label="5 kHz以下だけ表示（メルの fmax=5kHz）", value=True)
            show_f0  = gr.Checkbox(label="基本周波数(F0)ラインを表示", value=True)

            btn = gr.Button("解析する")

        with gr.Column():
            wave_plot = gr.Plot(label="波形")
            spec_plot = gr.Plot(label="メルスペクトログラム")
            wav_file  = gr.File(label="WAVダウンロード")
            png_file  = gr.File(label="メルスペクトログラムPNGダウンロード")
            status    = gr.Textbox(label="ステータス", interactive=False)
            audio_out = gr.Audio(label="音を再生", autoplay=False)

    # ソースに応じた表示切替
    source.change(toggle_inputs, inputs=[source], outputs=[uploaded_file, recorded_file, sample_choice])
    demo.load(toggle_inputs, inputs=[source], outputs=[uploaded_file, recorded_file, sample_choice])

    # 解析
    btn.click(
        process,
        inputs=[source, uploaded_file, recorded_file, sample_choice,
                t_start, t_end, limit_5k, show_f0],
        outputs=[wave_plot, spec_plot, wav_file, png_file, status, audio_out],
        queue=False
    )

if __name__ == "__main__":
    demo.launch()