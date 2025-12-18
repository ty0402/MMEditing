import os
import glob
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm



AUDIO_DIR_A = "static/audios/raw"
AUDIO_DIR_B = "static/audios/reorder"   


IMAGE_DIR = "static/images"


IMG_EXT = ".png"


FIG_SIZE = (10.0, 3.6)


WAVE_COLOR = "#00BFFF"
AXIS_COLOR = "#555555"
SPINE_COLOR = "#dddddd"
GRID_COLOR = "#eeeeee"


TRANSPARENT = True
# ===========================================


def _collect_wavs(directory: str):
    """递归收集目录内所有 wav。"""
    pattern = os.path.join(directory, "**/*.wav")
    return glob.glob(pattern, recursive=True)


def _build_name_map(wav_list):
    """
    构建 {stem: path} 映射。
    以“文件名（不含扩展名）”为同名判断标准。
    如果出现重复 stem，默认保留第一个。
    """
    mp = {}
    for p in wav_list:
        stem = Path(p).stem
        if stem not in mp:
            mp[stem] = p
    return mp


def _load_audio(audio_path):
    """加载音频，保持原采样率，单声道用于可视化。"""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return y, sr


def _compute_shared_ylim(y_a, y_b, eps=1e-6):
    """
    对一对音频计算共享幅度范围。
    这样 A/B 两张图的“响度/幅度刻度比例”完全一致。
    """
    max_a = float(np.max(np.abs(y_a))) if y_a is not None and len(y_a) else 0.0
    max_b = float(np.max(np.abs(y_b))) if y_b is not None and len(y_b) else 0.0
    m = max(max_a, max_b, eps)
    m *= 1.02  # 留一点点余量，防止贴边
    return (-m, m)


def _style_axes(ax):
    """统一坐标轴美化：保留左侧刻度 + 底部时间轴。"""
    # X 轴
    ax.set_xlabel("")
    ax.tick_params(axis="x", colors=AXIS_COLOR, labelsize=10)

    # Y 轴（关键：显示刻度）
    ax.tick_params(axis="y", colors=AXIS_COLOR, labelsize=9)

    # 边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(1)

    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_linewidth(1)

    # 非常淡的水平网格，让刻度更易读
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)


def generate_waveform_with_time(audio_path, output_path, ylim=None):

    try:
        y, sr = _load_audio(audio_path)

        fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="none")

        librosa.display.waveshow(
            y, sr=sr, color=WAVE_COLOR, alpha=0.9, ax=ax
        )

        if ylim is not None:
            ax.set_ylim(*ylim)

        y_min, y_max = ax.get_ylim()
        ticks = np.linspace(y_min, y_max, 5)
        ax.set_yticks(ticks)

        _style_axes(ax)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.12,
            transparent=TRANSPARENT,
            facecolor="none",
        )
        plt.close(fig)

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")


def main():
    wavs_a = _collect_wavs(AUDIO_DIR_A)
    wavs_b = _collect_wavs(AUDIO_DIR_B)

    if not wavs_a:
        print(f"⚠️ No .wav files found in {AUDIO_DIR_A}")
        return
    if not wavs_b:
        print(f"⚠️ No .wav files found in {AUDIO_DIR_B}")
        return


    map_a = _build_name_map(wavs_a)
    map_b = _build_name_map(wavs_b)

    common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common_keys:
        print("⚠️ No same-name wav pairs found between the two directories.")
        return


    tag_a = Path(AUDIO_DIR_A).name
    tag_b = Path(AUDIO_DIR_B).name

    out_dir_a = Path(IMAGE_DIR) / tag_a
    out_dir_b = Path(IMAGE_DIR) / tag_b
    out_dir_a.mkdir(parents=True, exist_ok=True)
    out_dir_b.mkdir(parents=True, exist_ok=True)

    print(f"🔎 Pairing by same filename:")
    print(f"   A: {AUDIO_DIR_A}")
    print(f"   B: {AUDIO_DIR_B}")
    print(f"✅ Found {len(common_keys)} pairs. Generating waveforms with shared scales...")

    for key in tqdm(common_keys):
        path_a = map_a[key]
        path_b = map_b[key]

        try:
            y_a, _ = _load_audio(path_a)
            y_b, _ = _load_audio(path_b)
            ylim = _compute_shared_ylim(y_a, y_b)

            # A 图
            out_a = out_dir_a / f"wave_{key}{IMG_EXT}"
            generate_waveform_with_time(path_a, str(out_a), ylim=ylim)

            # B 图
            out_b = out_dir_b / f"wave_{key}{IMG_EXT}"
            generate_waveform_with_time(path_b, str(out_b), ylim=ylim)

        except Exception as e:
            print(f"❌ Pair error for {key}: {e}")

    print("\n✅ All done!")
    print(f"   Images for A saved to: {out_dir_a}")
    print(f"   Images for B saved to: {out_dir_b}")
    print("ℹ️ Each pair uses identical Y-axis amplitude scale for fair loudness comparison.")


if __name__ == "__main__":
    main()
