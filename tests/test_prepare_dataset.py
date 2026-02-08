"""Tests for cli/prepare_dataset.py."""

import os
import json
import tempfile

import soundfile as sf
import numpy as np

from cli.prepare_dataset import find_source_dirs, convert_source, prepare_dataset


# --- Helpers ---

def make_segment(index="vocal_0_5000", language="Mandarin"):
    """Create a minimal segment metadata dict matching pipeline output."""
    return {
        "index": index,
        "language": language,
        "time": [0, 5000],
        "duration": "0.3 0.5 0.2",
        "text": "<SP> zh_ni3 <SP>",
        "phoneme": "<SP> zh_ni3 <SP>",
        "note_pitch": "0 60 0",
        "note_type": "1 2 1",
        "f0": " ".join(str(v) for v in [0.0] * 10 + [440.0] * 30 + [0.0] * 10),
    }


def make_preprocess_output(tmpdir, song_name="song_001", segments=None, sample_rate=24000):
    """Create a fake preprocessing pipeline output directory.

    Returns:
        Path to the created song directory.
    """
    if segments is None:
        segments = [
            make_segment("vocal_0_5000"),
            make_segment("vocal_5000_10000"),
        ]

    song_dir = os.path.join(tmpdir, song_name)
    wavs_dir = os.path.join(song_dir, "long_cut_wavs")
    os.makedirs(wavs_dir)

    # Write metadata.json
    with open(os.path.join(song_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)

    # Write wav files matching segment indices
    for seg in segments:
        wav_path = os.path.join(wavs_dir, f"{seg['index']}.wav")
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1
        sf.write(wav_path, audio, sample_rate)

    return song_dir


# --- find_source_dirs ---

class TestFindSourceDirs:
    def test_single_valid_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "mysong")
            result = find_source_dirs([song_dir])
            assert len(result) == 1
            assert result[0][0] == song_dir
            assert result[0][1] == "mysong"  # singer = dirname

    def test_multiple_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = make_preprocess_output(tmpdir, "song_A")
            d2 = make_preprocess_output(tmpdir, "song_B")
            result = find_source_dirs([d1, d2])
            assert len(result) == 2

    def test_skips_dir_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = os.path.join(tmpdir, "empty")
            os.makedirs(empty_dir)
            result = find_source_dirs([empty_dir])
            assert len(result) == 0

    def test_recursive_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_preprocess_output(tmpdir, "child_A")
            make_preprocess_output(tmpdir, "child_B")
            # tmpdir itself has no metadata.json, but children do
            result = find_source_dirs([tmpdir], recursive=True)
            assert len(result) == 2
            singers = {r[1] for r in result}
            assert singers == {"child_A", "child_B"}

    def test_recursive_skips_non_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_preprocess_output(tmpdir, "valid_song")
            # Create a file (not dir) in tmpdir
            with open(os.path.join(tmpdir, "notes.txt"), "w") as f:
                f.write("test")
            result = find_source_dirs([tmpdir], recursive=True)
            assert len(result) == 1

    def test_non_recursive_with_parent_dir(self):
        """Without --recursive, a parent dir without metadata.json is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_preprocess_output(tmpdir, "child")
            result = find_source_dirs([tmpdir], recursive=False)
            assert len(result) == 0


# --- convert_source ---

class TestConvertSource:
    def test_creates_metadata_and_wavs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "song_001")
            out_dir = os.path.join(tmpdir, "dataset")

            count = convert_source(song_dir, "singer_A", out_dir, "song_001")
            assert count == 2

            # Check metadata files exist
            assert os.path.isfile(os.path.join(out_dir, "metadata", "song_001_seg_000.json"))
            assert os.path.isfile(os.path.join(out_dir, "metadata", "song_001_seg_001.json"))

            # Check wav files exist
            assert os.path.isfile(os.path.join(out_dir, "wavs", "song_001_seg_000.wav"))
            assert os.path.isfile(os.path.join(out_dir, "wavs", "song_001_seg_001.wav"))

    def test_metadata_contains_singer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "song_001")
            out_dir = os.path.join(tmpdir, "dataset")

            convert_source(song_dir, "vocalist_X", out_dir, "song_001")

            with open(os.path.join(out_dir, "metadata", "song_001_seg_000.json"), "r") as f:
                meta = json.load(f)
            assert meta["singer"] == "vocalist_X"

    def test_metadata_preserves_original_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "song_001")
            out_dir = os.path.join(tmpdir, "dataset")

            convert_source(song_dir, "singer_A", out_dir, "song_001")

            with open(os.path.join(out_dir, "metadata", "song_001_seg_000.json"), "r") as f:
                meta = json.load(f)

            for key in ("index", "language", "time", "duration", "phoneme",
                        "note_pitch", "note_type", "f0"):
                assert key in meta, f"Missing field: {key}"

    def test_skips_segments_without_wav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            segments = [
                make_segment("vocal_0_5000"),
                make_segment("vocal_5000_10000"),
            ]
            song_dir = make_preprocess_output(tmpdir, "song_001", segments)

            # Delete one wav
            os.remove(os.path.join(song_dir, "long_cut_wavs", "vocal_5000_10000.wav"))

            out_dir = os.path.join(tmpdir, "dataset")
            count = convert_source(song_dir, "singer_A", out_dir, "song_001")
            assert count == 1

    def test_wav_is_valid_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "song_001")
            out_dir = os.path.join(tmpdir, "dataset")

            convert_source(song_dir, "singer_A", out_dir, "song_001")

            wav_path = os.path.join(out_dir, "wavs", "song_001_seg_000.wav")
            data, sr = sf.read(wav_path)
            assert sr == 24000
            assert len(data) > 0

    def test_start_idx_offset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "song_001")
            out_dir = os.path.join(tmpdir, "dataset")

            convert_source(song_dir, "singer_A", out_dir, "song_001", start_idx=5)

            assert os.path.isfile(os.path.join(out_dir, "metadata", "song_001_seg_005.json"))
            assert os.path.isfile(os.path.join(out_dir, "metadata", "song_001_seg_006.json"))

    def test_single_segment_metadata(self):
        """Handle metadata.json that is a single dict (not a list)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seg = make_segment("vocal_0_5000")
            song_dir = os.path.join(tmpdir, "song_001")
            wavs_dir = os.path.join(song_dir, "long_cut_wavs")
            os.makedirs(wavs_dir)

            # Write as a single dict, not a list
            with open(os.path.join(song_dir, "metadata.json"), "w") as f:
                json.dump(seg, f)
            audio = np.random.randn(24000).astype(np.float32) * 0.1
            sf.write(os.path.join(wavs_dir, "vocal_0_5000.wav"), audio, 24000)

            out_dir = os.path.join(tmpdir, "dataset")
            count = convert_source(song_dir, "singer_A", out_dir, "song_001")
            assert count == 1

    def test_empty_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = os.path.join(tmpdir, "song_001")
            os.makedirs(os.path.join(song_dir, "long_cut_wavs"))
            with open(os.path.join(song_dir, "metadata.json"), "w") as f:
                json.dump([], f)

            out_dir = os.path.join(tmpdir, "dataset")
            count = convert_source(song_dir, "singer_A", out_dir, "song_001")
            assert count == 0


# --- prepare_dataset (integration) ---

class TestPrepareDataset:
    def test_single_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "song_001")
            out_dir = os.path.join(tmpdir, "dataset")

            total = prepare_dataset([song_dir], out_dir)
            assert total == 2
            assert os.path.isdir(os.path.join(out_dir, "metadata"))
            assert os.path.isdir(os.path.join(out_dir, "wavs"))

    def test_multiple_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = make_preprocess_output(tmpdir, "song_A", [make_segment("vocal_0_3000")])
            d2 = make_preprocess_output(tmpdir, "song_B", [
                make_segment("vocal_0_4000"),
                make_segment("vocal_4000_8000"),
            ])
            out_dir = os.path.join(tmpdir, "dataset")

            total = prepare_dataset([d1, d2], out_dir)
            assert total == 3

    def test_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_preprocess_output(tmpdir, "artist_1")
            make_preprocess_output(tmpdir, "artist_2")
            out_dir = os.path.join(tmpdir, "dataset")

            total = prepare_dataset([tmpdir], out_dir, recursive=True)
            assert total == 4  # 2 segments * 2 sources

    def test_explicit_singer_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "song_001")
            out_dir = os.path.join(tmpdir, "dataset")

            prepare_dataset([song_dir], out_dir, singer="custom_singer")

            with open(os.path.join(out_dir, "metadata", "song_001_seg_000.json"), "r") as f:
                meta = json.load(f)
            assert meta["singer"] == "custom_singer"

    def test_auto_singer_from_dirname(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "my_artist")
            out_dir = os.path.join(tmpdir, "dataset")

            prepare_dataset([song_dir], out_dir)

            with open(os.path.join(out_dir, "metadata", "my_artist_seg_000.json"), "r") as f:
                meta = json.load(f)
            assert meta["singer"] == "my_artist"

    def test_no_valid_sources_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty = os.path.join(tmpdir, "empty")
            os.makedirs(empty)
            out_dir = os.path.join(tmpdir, "dataset")

            total = prepare_dataset([empty], out_dir)
            assert total == 0

    def test_output_compatible_with_svs_dataset(self):
        """Verify the output can be loaded by SVSDataset."""
        from soulxsinger.utils.data_processor import DataProcessor
        from soulxsinger.utils.dataset import SVSDataset

        phoneset_path = os.path.join(
            os.path.dirname(__file__), "..", "soulxsinger", "utils", "phoneme", "phone_set.json"
        )
        phoneset_path = os.path.abspath(phoneset_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            song_dir = make_preprocess_output(tmpdir, "song_001")
            out_dir = os.path.join(tmpdir, "dataset")

            prepare_dataset([song_dir], out_dir, singer="singer_A")

            processor = DataProcessor(
                hop_size=480, sample_rate=24000,
                phoneset_path=phoneset_path, device="cpu",
            )
            ds = SVSDataset(out_dir, processor)
            assert len(ds) == 2

            # Should be able to get a sample
            sample = ds[0]
            assert "target" in sample
            assert "prompt" in sample

    def test_different_singers_grouped_correctly(self):
        """Segments from different singers should be grouped separately."""
        from soulxsinger.utils.data_processor import DataProcessor
        from soulxsinger.utils.dataset import SVSDataset

        phoneset_path = os.path.join(
            os.path.dirname(__file__), "..", "soulxsinger", "utils", "phoneme", "phone_set.json"
        )
        phoneset_path = os.path.abspath(phoneset_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = make_preprocess_output(tmpdir, "singer_X", [
                make_segment("vocal_0_3000"),
                make_segment("vocal_3000_6000"),
            ])
            d2 = make_preprocess_output(tmpdir, "singer_Y", [
                make_segment("vocal_0_4000"),
            ])
            out_dir = os.path.join(tmpdir, "dataset")

            prepare_dataset([d1, d2], out_dir)

            processor = DataProcessor(
                hop_size=480, sample_rate=24000,
                phoneset_path=phoneset_path, device="cpu",
            )
            ds = SVSDataset(out_dir, processor)
            assert len(ds) == 3
            assert len(ds.singer_to_indices["singer_X"]) == 2
            assert len(ds.singer_to_indices["singer_Y"]) == 1
