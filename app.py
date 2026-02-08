import os
import sys
import json
import tempfile

import torch
import numpy as np
import soundfile as sf
import gradio as gr

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from soulxsinger.utils.file_utils import load_config
from soulxsinger.models.soulxsinger import SoulXSinger
from soulxsinger.utils.data_processor import DataProcessor

# 前処理パイプラインの読み込み（extras未インストール時はスキップ）
try:
    from preprocess.pipeline import PreprocessPipeline

    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

# --- グローバル状態 ---
_model = None
_config = None
_data_processor = None
_preprocess_pipeline = None

DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained_models", "SoulX-Singer", "model.pt")
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "soulxsinger", "config", "soulxsinger.yaml")
DEFAULT_PHONESET_PATH = os.path.join(ROOT_DIR, "soulxsinger", "utils", "phoneme", "phone_set.json")
DEFAULT_PREPROCESS_MODEL_DIR = os.path.join(ROOT_DIR, "pretrained_models", "SoulX-Singer-Preprocess")


def load_model(model_path: str, config_path: str, device: str):
    """モデルをロードしてグローバルに保持する。"""
    global _model, _config, _data_processor

    config = load_config(config_path)
    _config = config

    if not os.path.isfile(model_path):
        raise gr.Error(
            f"モデルファイルが見つかりません: {model_path}\n"
            "pretrained_models/ にモデルをダウンロードしてください。"
        )

    model = SoulXSinger(config).to(device)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    if "state_dict" not in checkpoint:
        raise gr.Error("チェックポイントに 'state_dict' キーがありません。")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    _model = model

    _data_processor = DataProcessor(
        hop_size=config.audio.hop_size,
        sample_rate=config.audio.sample_rate,
        phoneset_path=DEFAULT_PHONESET_PATH,
        device=device,
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    return f"モデルロード完了（{param_count:.1f}M パラメータ、デバイス: {device}）"


def _run_inference(
    prompt_audio_path: str,
    prompt_metadata_path: str,
    target_metadata_path: str,
    control_mode: str,
    auto_shift: bool,
    pitch_shift: int,
):
    """推論の共通ロジック。ファイルパスを受け取って生成音声パスを返す。"""
    if _model is None or _config is None or _data_processor is None:
        raise gr.Error("先にモデルをロードしてください。")

    with open(prompt_metadata_path, "r", encoding="utf-8") as f:
        prompt_meta_list = json.load(f)
    if not prompt_meta_list:
        raise gr.Error("プロンプトメタデータが空です。")
    prompt_meta = prompt_meta_list[0]

    with open(target_metadata_path, "r", encoding="utf-8") as f:
        target_meta_list = json.load(f)
    if not target_meta_list:
        raise gr.Error("ターゲットメタデータが空です。")

    infer_prompt_data = _data_processor.process(prompt_meta, prompt_audio_path)

    sample_rate = _config.audio.sample_rate
    generated_len = int(target_meta_list[-1]["time"][1] / 1000 * sample_rate)
    generated_merged = np.zeros(generated_len, dtype=np.float32)

    control = "melody" if control_mode == "メロディ制御（F0）" else "score"

    for target_meta in target_meta_list:
        start_sample_idx = int(target_meta["time"][0] / 1000 * sample_rate)
        infer_target_data = _data_processor.process(target_meta, None)

        infer_data = {
            "prompt": infer_prompt_data,
            "target": infer_target_data,
        }

        with torch.no_grad():
            generated_audio = _model.infer(
                infer_data,
                auto_shift=auto_shift,
                pitch_shift=pitch_shift,
                n_steps=_config.infer.n_steps,
                cfg=_config.infer.cfg,
                control=control,
            )

        generated_audio = generated_audio.squeeze().cpu().numpy()
        generated_merged[
            start_sample_idx : start_sample_idx + generated_audio.shape[0]
        ] = generated_audio

    output_path = os.path.join(tempfile.gettempdir(), "soulxsinger_output.wav")
    sf.write(output_path, generated_merged, sample_rate)

    return output_path


def infer_advanced(
    prompt_audio_path: str,
    prompt_metadata_file: str,
    target_metadata_file: str,
    control_mode: str,
    auto_shift: bool,
    pitch_shift: int,
):
    """詳細モード: 音声 + JSON で推論を実行。"""
    if prompt_audio_path is None:
        raise gr.Error("プロンプト音声ファイルを選択してください。")
    if prompt_metadata_file is None:
        raise gr.Error("プロンプトメタデータJSONを選択してください。")
    if target_metadata_file is None:
        raise gr.Error("ターゲットメタデータJSONを選択してください。")

    return _run_inference(
        prompt_audio_path,
        prompt_metadata_file,
        target_metadata_file,
        control_mode,
        auto_shift,
        pitch_shift,
    )


def _init_preprocess_pipeline(device: str, language: str, save_dir: str, vocal_sep: bool):
    """前処理パイプラインを初期化する。"""
    global _preprocess_pipeline

    if not PREPROCESS_AVAILABLE:
        raise gr.Error(
            "前処理パイプラインが利用できません。\n"
            "`uv sync --extra preprocess` を実行してインストールしてください。"
        )

    if not os.path.isdir(DEFAULT_PREPROCESS_MODEL_DIR):
        raise gr.Error(
            f"前処理モデルが見つかりません: {DEFAULT_PREPROCESS_MODEL_DIR}\n"
            "以下のコマンドでダウンロードしてください:\n"
            "uv run huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess "
            "--local-dir pretrained_models/SoulX-Singer-Preprocess"
        )

    _preprocess_pipeline = PreprocessPipeline(
        device=device,
        language=language,
        save_dir=save_dir,
        vocal_sep=vocal_sep,
    )


def infer_easy(
    prompt_audio_path: str,
    target_audio_path: str,
    language: str,
    vocal_sep: bool,
    control_mode: str,
    auto_shift: bool,
    pitch_shift: int,
):
    """簡単モード: 音声ファイルのみで前処理→推論を実行。"""
    if _model is None or _config is None or _data_processor is None:
        raise gr.Error("先にモデルをロードしてください。")
    if prompt_audio_path is None:
        raise gr.Error("プロンプト音声ファイルを選択してください。")
    if target_audio_path is None:
        raise gr.Error("ターゲット音声ファイルを選択してください。")

    device = str(next(_model.parameters()).device)

    # 前処理用の一時ディレクトリを作成
    work_dir = tempfile.mkdtemp(prefix="soulx_preprocess_")
    prompt_dir = os.path.join(work_dir, "prompt")
    target_dir = os.path.join(work_dir, "target")
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    # プロンプト音声を前処理（ボーカル分離なし — プロンプトは既にクリーンな音声を期待）
    gr.Info("プロンプト音声を前処理中...")
    try:
        _init_preprocess_pipeline(device, language, prompt_dir, vocal_sep=False)
        _preprocess_pipeline.run(
            audio_path=prompt_audio_path,
            vocal_sep=False,
            language=language,
        )
    except Exception as e:
        raise gr.Error(f"プロンプト音声の前処理に失敗しました: {e}")

    prompt_metadata_path = os.path.join(prompt_dir, "metadata.json")
    if not os.path.isfile(prompt_metadata_path):
        raise gr.Error("プロンプトメタデータの生成に失敗しました。")

    # ターゲット音声を前処理
    gr.Info("ターゲット音声を前処理中...")
    try:
        _init_preprocess_pipeline(device, language, target_dir, vocal_sep=vocal_sep)
        _preprocess_pipeline.run(
            audio_path=target_audio_path,
            vocal_sep=vocal_sep,
            language=language,
        )
    except Exception as e:
        raise gr.Error(f"ターゲット音声の前処理に失敗しました: {e}")

    target_metadata_path = os.path.join(target_dir, "metadata.json")
    if not os.path.isfile(target_metadata_path):
        raise gr.Error("ターゲットメタデータの生成に失敗しました。")

    # プロンプト音声パス: ボーカル分離していないのでそのまま使用
    # （前処理で vocal.wav が作られるが、元の音声をそのまま使う）
    prompt_vocal_path = prompt_audio_path

    # 推論を実行
    gr.Info("歌声を生成中...")
    return _run_inference(
        prompt_vocal_path,
        prompt_metadata_path,
        target_metadata_path,
        control_mode,
        auto_shift,
        pitch_shift,
    )


def build_ui():
    """Gradio UIを構築する。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(title="SoulX-Singer") as demo:
        gr.Markdown("# SoulX-Singer — ゼロショット歌声合成")
        gr.Markdown(
            "プロンプト音声の音色を模倣して、ターゲットメタデータに基づく歌声を生成します。"
        )

        # --- モデルロード ---
        with gr.Accordion("モデル設定", open=False):
            with gr.Row():
                model_path_input = gr.Textbox(
                    label="モデルパス",
                    value=DEFAULT_MODEL_PATH,
                )
                config_path_input = gr.Textbox(
                    label="設定ファイルパス",
                    value=DEFAULT_CONFIG_PATH,
                )
                device_input = gr.Dropdown(
                    label="デバイス",
                    choices=["cuda", "cpu"],
                    value=device,
                )
            load_btn = gr.Button("モデルをロード", variant="primary")
            _initial_status = ""
            if _model is not None:
                _dev = str(next(_model.parameters()).device)
                _pc = sum(p.numel() for p in _model.parameters()) / 1e6
                _initial_status = f"モデルロード完了（{_pc:.1f}M パラメータ、デバイス: {_dev}）"
            load_status = gr.Textbox(label="ステータス", interactive=False, value=_initial_status)

        load_btn.click(
            fn=load_model,
            inputs=[model_path_input, config_path_input, device_input],
            outputs=load_status,
        )

        gr.Markdown("---")

        # --- 入力（タブ切替） ---
        with gr.Tabs():
            # === 簡単モード ===
            with gr.TabItem("簡単モード"):
                if not PREPROCESS_AVAILABLE:
                    gr.Markdown(
                        "⚠️ **前処理パイプラインが利用できません。**\n\n"
                        "`uv sync --extra preprocess` を実行してインストールしてください。\n\n"
                        "前処理モデルのダウンロードも必要です:\n"
                        "```\n"
                        "uv run huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess "
                        "--local-dir pretrained_models/SoulX-Singer-Preprocess\n"
                        "```"
                    )

                gr.Markdown(
                    "音声ファイルのみで歌声を生成します。メタデータJSONは自動的に前処理で生成されます。"
                )
                with gr.Row():
                    with gr.Column():
                        easy_prompt_audio = gr.Audio(
                            label="プロンプト音声（音色の参照元）",
                            type="filepath",
                        )
                    with gr.Column():
                        easy_target_audio = gr.Audio(
                            label="ターゲット音声（歌わせたい曲）",
                            type="filepath",
                        )
                with gr.Row():
                    easy_language = gr.Dropdown(
                        label="言語",
                        choices=["Mandarin", "English", "Cantonese"],
                        value="Mandarin",
                    )
                    easy_vocal_sep = gr.Checkbox(
                        label="ボーカル分離（ターゲット音声）",
                        value=True,
                        info="ターゲット音声にBGMが含まれる場合はONにしてください",
                    )

            # === 詳細モード ===
            with gr.TabItem("詳細モード"):
                gr.Markdown(
                    "音声ファイルとメタデータJSONを直接指定して推論を実行します。"
                )
                adv_prompt_audio = gr.Audio(
                    label="プロンプト音声（音色の参照元）",
                    type="filepath",
                )
                adv_prompt_metadata = gr.File(
                    label="プロンプトメタデータ（JSON）",
                    file_types=[".json"],
                )
                adv_target_metadata = gr.File(
                    label="ターゲットメタデータ（JSON）",
                    file_types=[".json"],
                )

        # --- 共通パラメータ ---
        gr.Markdown("### パラメータ")
        with gr.Row():
            control_mode = gr.Radio(
                label="制御モード",
                choices=["メロディ制御（F0）", "スコア制御（MIDI）"],
                value="スコア制御（MIDI）",
            )
            auto_shift = gr.Checkbox(
                label="自動ピッチシフト",
                value=True,
                info="プロンプトとターゲットのF0中央値に基づき自動調整",
            )
            pitch_shift = gr.Slider(
                label="ピッチシフト（半音）",
                minimum=-12,
                maximum=12,
                step=1,
                value=0,
            )

        # --- 生成ボタン（モード別） ---
        with gr.Row():
            easy_infer_btn = gr.Button(
                "簡単モードで生成",
                variant="primary",
                size="lg",
                interactive=PREPROCESS_AVAILABLE,
            )
            adv_infer_btn = gr.Button(
                "詳細モードで生成",
                variant="primary",
                size="lg",
            )

        # --- 出力 ---
        gr.Markdown("### 出力")
        output_audio = gr.Audio(label="生成された歌声", type="filepath")

        # 簡単モードのイベント
        easy_infer_btn.click(
            fn=infer_easy,
            inputs=[
                easy_prompt_audio,
                easy_target_audio,
                easy_language,
                easy_vocal_sep,
                control_mode,
                auto_shift,
                pitch_shift,
            ],
            outputs=output_audio,
        )

        # 詳細モードのイベント
        adv_infer_btn.click(
            fn=infer_advanced,
            inputs=[
                adv_prompt_audio,
                adv_prompt_metadata,
                adv_target_metadata,
                control_mode,
                auto_shift,
                pitch_shift,
            ],
            outputs=output_audio,
        )

        # --- サンプル ---
        example_dir = os.path.join(ROOT_DIR, "example", "audio")
        if os.path.isdir(example_dir):
            gr.Markdown("---")
            gr.Markdown("### サンプル（簡単モード）")
            gr.Examples(
                examples=[
                    [
                        os.path.join(example_dir, "zh_prompt.mp3"),
                        os.path.join(example_dir, "zh_target.mp3"),
                        "Mandarin",
                        False,
                    ],
                    [
                        os.path.join(example_dir, "en_prompt.mp3"),
                        os.path.join(example_dir, "en_target.mp3"),
                        "English",
                        False,
                    ],
                ],
                inputs=[
                    easy_prompt_audio,
                    easy_target_audio,
                    easy_language,
                    easy_vocal_sep,
                ],
                label="サンプルデータ（簡単モード）",
            )

            gr.Markdown("### サンプル（詳細モード）")
            gr.Examples(
                examples=[
                    [
                        os.path.join(example_dir, "zh_prompt.mp3"),
                        os.path.join(example_dir, "zh_prompt.json"),
                        os.path.join(example_dir, "music.json"),
                        "スコア制御（MIDI）",
                        True,
                        0,
                    ],
                    [
                        os.path.join(example_dir, "en_prompt.mp3"),
                        os.path.join(example_dir, "en_prompt.json"),
                        os.path.join(example_dir, "en_target.json"),
                        "スコア制御（MIDI）",
                        True,
                        0,
                    ],
                ],
                inputs=[
                    adv_prompt_audio,
                    adv_prompt_metadata,
                    adv_target_metadata,
                    control_mode,
                    auto_shift,
                    pitch_shift,
                ],
                label="サンプルデータ（詳細モード）",
            )

    return demo


if __name__ == "__main__":
    # 起動時にモデルを自動ロード
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.isfile(DEFAULT_MODEL_PATH):
        print(f"モデルを自動ロード中... ({device})")
        try:
            status = load_model(DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH, device)
            print(status)
        except Exception as e:
            print(f"モデルの自動ロードに失敗しました: {e}")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7870, theme=gr.themes.Soft())
