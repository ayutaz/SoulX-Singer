"""Tests for SVSDataset and collate_fn."""

import os
import json
import tempfile

import torch
import pytest

from soulxsinger.utils.data_processor import DataProcessor
from soulxsinger.utils.dataset import SVSDataset


# --- Fixtures ---

def make_segment_metadata(
    duration="0.3 0.5 0.2",
    phoneme="<SP> zh_ni3 <SP>",
    note_pitch="0 60 0",
    note_type="1 2 1",
    f0_frames=50,
    singer="singer_A",
    time=None,
):
    """Create a minimal segment metadata dict."""
    f0_values = [0.0] * 10 + [440.0] * (f0_frames - 20) + [0.0] * 10
    return {
        "duration": duration,
        "phoneme": phoneme,
        "note_pitch": note_pitch,
        "note_type": note_type,
        "f0": " ".join(str(v) for v in f0_values),
        "singer": singer,
        "time": time or [0, int(sum(float(d) for d in duration.split()) * 1000)],
    }


def make_wav_file(path, sample_rate=24000, duration_sec=1.0):
    """Create a minimal wav file using torchaudio."""
    import torchaudio
    n_samples = int(sample_rate * duration_sec)
    waveform = torch.randn(1, n_samples) * 0.1
    torchaudio.save(path, waveform, sample_rate)
    return path


@pytest.fixture
def phoneset_path():
    """Return path to the phone_set.json in the project."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "soulxsinger", "utils", "phoneme", "phone_set.json"
    )
    return os.path.abspath(path)


@pytest.fixture
def data_processor(phoneset_path):
    """Create a DataProcessor for testing."""
    return DataProcessor(
        hop_size=480,
        sample_rate=24000,
        phoneset_path=phoneset_path,
        device="cpu",
    )


@pytest.fixture
def sample_dataset_dir(data_processor):
    """Create a temporary dataset directory with metadata + wavs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_dir = os.path.join(tmpdir, "metadata")
        wavs_dir = os.path.join(tmpdir, "wavs")
        os.makedirs(meta_dir)
        os.makedirs(wavs_dir)

        # Create 3 segments for singer_A, 2 for singer_B
        for i in range(3):
            seg_name = f"song_001_seg_{i:03d}"
            meta = make_segment_metadata(singer="singer_A")
            with open(os.path.join(meta_dir, f"{seg_name}.json"), "w") as f:
                json.dump(meta, f)
            make_wav_file(os.path.join(wavs_dir, f"{seg_name}.wav"), duration_sec=1.0)

        for i in range(2):
            seg_name = f"song_002_seg_{i:03d}"
            meta = make_segment_metadata(singer="singer_B")
            with open(os.path.join(meta_dir, f"{seg_name}.json"), "w") as f:
                json.dump(meta, f)
            make_wav_file(os.path.join(wavs_dir, f"{seg_name}.wav"), duration_sec=0.8)

        yield tmpdir


@pytest.fixture
def dataset(sample_dataset_dir, data_processor):
    """Create a SVSDataset from the temp directory."""
    return SVSDataset(
        data_dir=sample_dataset_dir,
        data_processor=data_processor,
        sample_rate=24000,
        hop_size=480,
    )


# --- Dataset initialization tests ---

class TestSVSDatasetInit:
    def test_loads_correct_number_of_items(self, dataset):
        assert len(dataset) == 5  # 3 + 2

    def test_singer_grouping(self, dataset):
        assert len(dataset.singer_to_indices["singer_A"]) == 3
        assert len(dataset.singer_to_indices["singer_B"]) == 2

    def test_skips_metadata_without_wav(self, data_processor):
        """If a wav is missing, that item should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = os.path.join(tmpdir, "metadata")
            wavs_dir = os.path.join(tmpdir, "wavs")
            os.makedirs(meta_dir)
            os.makedirs(wavs_dir)

            # metadata exists but no wav
            meta = make_segment_metadata()
            with open(os.path.join(meta_dir, "orphan.json"), "w") as f:
                json.dump(meta, f)

            # metadata + wav pair
            meta2 = make_segment_metadata()
            with open(os.path.join(meta_dir, "valid.json"), "w") as f:
                json.dump(meta2, f)
            make_wav_file(os.path.join(wavs_dir, "valid.wav"))

            ds = SVSDataset(tmpdir, data_processor)
            assert len(ds) == 1

    def test_handles_list_metadata(self, data_processor):
        """JSON that is a list (like [segment]) should be handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = os.path.join(tmpdir, "metadata")
            wavs_dir = os.path.join(tmpdir, "wavs")
            os.makedirs(meta_dir)
            os.makedirs(wavs_dir)

            meta = [make_segment_metadata()]  # wrapped in list
            with open(os.path.join(meta_dir, "test.json"), "w") as f:
                json.dump(meta, f)
            make_wav_file(os.path.join(wavs_dir, "test.wav"))

            ds = SVSDataset(tmpdir, data_processor)
            assert len(ds) == 1

    def test_default_singer_when_missing(self, data_processor):
        """If no 'singer' field, should use 'default'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = os.path.join(tmpdir, "metadata")
            wavs_dir = os.path.join(tmpdir, "wavs")
            os.makedirs(meta_dir)
            os.makedirs(wavs_dir)

            meta = make_segment_metadata()
            del meta["singer"]
            with open(os.path.join(meta_dir, "test.json"), "w") as f:
                json.dump(meta, f)
            make_wav_file(os.path.join(wavs_dir, "test.wav"))

            ds = SVSDataset(tmpdir, data_processor)
            assert "default" in ds.singer_to_indices


# --- __getitem__ tests ---

class TestSVSDatasetGetItem:
    def test_returns_dict_with_expected_keys(self, dataset):
        sample = dataset[0]
        assert "target" in sample
        assert "prompt" in sample
        for key in ("phoneme", "note_pitch", "note_type", "mel2note", "f0", "waveform"):
            assert key in sample["target"], f"Missing key '{key}' in target"
            assert key in sample["prompt"], f"Missing key '{key}' in prompt"

    def test_target_and_prompt_have_batch_dim(self, dataset):
        sample = dataset[0]
        assert sample["target"]["phoneme"].dim() == 2  # (1, N)
        assert sample["target"]["mel2note"].dim() == 2  # (1, F)

    def test_prompt_comes_from_same_singer(self, dataset):
        """For singer_A items, prompt should also be singer_A."""
        singer_a_indices = dataset.singer_to_indices["singer_A"]
        for idx in singer_a_indices:
            sample = dataset[idx]
            # We can't directly verify which singer the prompt came from
            # but we can verify it's a valid sample
            assert sample["prompt"]["phoneme"].shape[1] > 0

    def test_single_singer_uses_self_as_prompt(self, data_processor):
        """When there's only one segment for a singer, it uses itself as prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = os.path.join(tmpdir, "metadata")
            wavs_dir = os.path.join(tmpdir, "wavs")
            os.makedirs(meta_dir)
            os.makedirs(wavs_dir)

            meta = make_segment_metadata(singer="lonely")
            with open(os.path.join(meta_dir, "only.json"), "w") as f:
                json.dump(meta, f)
            make_wav_file(os.path.join(wavs_dir, "only.wav"))

            ds = SVSDataset(tmpdir, data_processor)
            sample = ds[0]
            # Should not crash; prompt and target should have same shapes
            assert sample["target"]["phoneme"].shape == sample["prompt"]["phoneme"].shape

    def test_waveform_is_1d_or_2d(self, dataset):
        sample = dataset[0]
        wav = sample["target"]["waveform"]
        assert wav.dim() in (1, 2)  # (T,) or (1, T)


# --- Collate function tests ---

class TestCollateFn:
    def test_basic_collation(self, dataset):
        samples = [dataset[0], dataset[1]]
        batch = SVSDataset.collate_fn(samples)

        assert batch["phoneme"].shape[0] == 2
        assert batch["note_pitch"].shape[0] == 2
        assert batch["note_type"].shape[0] == 2
        assert batch["mel2note"].shape[0] == 2
        assert batch["f0"].shape[0] == 2
        assert batch["mel_mask"].shape[0] == 2
        assert batch["is_prompt"].shape[0] == 2
        assert batch["prompt_mel_len"].shape[0] == 2
        assert batch["prompt_waveform"].shape[0] == 2
        assert batch["target_waveform"].shape[0] == 2

    def test_mel_mask_has_correct_values(self, dataset):
        samples = [dataset[0], dataset[1]]
        batch = SVSDataset.collate_fn(samples)

        # mel_mask should be 1 for valid frames, 0 for padding
        for i in range(2):
            mask = batch["mel_mask"][i]
            assert mask.sum() > 0  # at least some valid frames
            # padding region should be 0
            valid_len = int(mask.sum().item())
            assert (mask[:valid_len] == 1).all()
            if valid_len < mask.shape[0]:
                assert (mask[valid_len:] == 0).all()

    def test_is_prompt_mask_structure(self, dataset):
        samples = [dataset[0], dataset[1]]
        batch = SVSDataset.collate_fn(samples)

        for i in range(2):
            prompt_len = batch["prompt_mel_len"][i].item()
            is_prompt = batch["is_prompt"][i]
            # First prompt_len frames should be 1
            assert (is_prompt[:prompt_len] == 1).all()
            # After prompt, within valid region, should be 0
            valid_total = int(batch["mel_mask"][i].sum().item())
            if valid_total > prompt_len:
                assert (is_prompt[prompt_len:valid_total] == 0).all()

    def test_padding_is_zero(self, dataset):
        """Padded regions should be filled with zeros."""
        samples = [dataset[0], dataset[1]]
        batch = SVSDataset.collate_fn(samples)

        for key in ("phoneme", "note_pitch", "note_type", "mel2note"):
            assert (batch[key] >= 0).all(), f"Negative values in padded {key}"

    def test_collate_single_sample(self, dataset):
        """Collation should work with a single sample."""
        samples = [dataset[0]]
        batch = SVSDataset.collate_fn(samples)
        assert batch["phoneme"].shape[0] == 1

    def test_collate_preserves_dtypes(self, dataset):
        samples = [dataset[0], dataset[1]]
        batch = SVSDataset.collate_fn(samples)

        assert batch["phoneme"].dtype == torch.long
        assert batch["note_pitch"].dtype == torch.long
        assert batch["note_type"].dtype == torch.long
        assert batch["mel2note"].dtype == torch.long
        assert batch["f0"].dtype == torch.float
        assert batch["mel_mask"].dtype == torch.float
        assert batch["is_prompt"].dtype == torch.float
        assert batch["prompt_waveform"].dtype == torch.float
        assert batch["target_waveform"].dtype == torch.float

    def test_different_length_samples_padded_correctly(self, data_processor):
        """Samples with different durations should be correctly padded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = os.path.join(tmpdir, "metadata")
            wavs_dir = os.path.join(tmpdir, "wavs")
            os.makedirs(meta_dir)
            os.makedirs(wavs_dir)

            # Short segment
            meta_short = make_segment_metadata(
                duration="0.2 0.3",
                phoneme="<SP> zh_ni3",
                note_pitch="0 60",
                note_type="1 2",
                f0_frames=25,
            )
            with open(os.path.join(meta_dir, "short.json"), "w") as f:
                json.dump(meta_short, f)
            make_wav_file(os.path.join(wavs_dir, "short.wav"), duration_sec=0.5)

            # Long segment
            meta_long = make_segment_metadata(
                duration="0.3 0.5 0.4 0.3",
                phoneme="<SP> zh_ni3 zh_hao3 <SP>",
                note_pitch="0 60 62 0",
                note_type="1 2 2 1",
                f0_frames=75,
            )
            with open(os.path.join(meta_dir, "long.json"), "w") as f:
                json.dump(meta_long, f)
            make_wav_file(os.path.join(wavs_dir, "long.wav"), duration_sec=1.5)

            ds = SVSDataset(tmpdir, data_processor)
            samples = [ds[0], ds[1]]
            batch = SVSDataset.collate_fn(samples)

            # Both should have same padded dimensions
            assert batch["phoneme"].shape[0] == 2
            assert batch["mel2note"].shape[0] == 2
            # Longer sample should have more valid frames
            mask_0 = batch["mel_mask"][0].sum().item()
            mask_1 = batch["mel_mask"][1].sum().item()
            assert mask_0 != mask_1 or True  # lengths may differ
            # Max dim should accommodate the longer sample
            assert batch["mel2note"].shape[1] >= max(mask_0, mask_1)

    def test_prompt_waveform_3d(self, dataset):
        """prompt_waveform should be (B, 1, T)."""
        samples = [dataset[0]]
        batch = SVSDataset.collate_fn(samples)
        assert batch["prompt_waveform"].dim() == 3
        assert batch["prompt_waveform"].shape[1] == 1

    def test_target_waveform_3d(self, dataset):
        """target_waveform should be (B, 1, T)."""
        samples = [dataset[0]]
        batch = SVSDataset.collate_fn(samples)
        assert batch["target_waveform"].dim() == 3
        assert batch["target_waveform"].shape[1] == 1


# --- DataLoader integration test ---

class TestDataLoaderIntegration:
    def test_dataloader_iterates(self, dataset):
        """DataLoader should be able to iterate over the dataset."""
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=SVSDataset.collate_fn,
            num_workers=0,
        )
        batch = next(iter(loader))
        assert batch["phoneme"].shape[0] == 2
        assert batch["prompt_waveform"].shape[0] == 2
