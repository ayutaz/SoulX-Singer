import os
import copy
import json
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from soulxsinger.utils.data_processor import DataProcessor


class SVSDataset(Dataset):
    """Singing Voice Synthesis dataset for training.

    Directory structure:
        data_dir/
        ├── metadata/
        │   ├── song_001_seg_000.json
        │   ├── song_001_seg_001.json
        │   └── ...
        └── wavs/
            ├── song_001_seg_000.wav
            ├── song_001_seg_001.wav
            └── ...

    Each metadata JSON contains a single segment dict with keys:
        duration, phoneme, note_pitch, note_type, f0, time, etc.
    Optionally includes a "singer" field to group segments by singer
    for prompt selection. If absent, all segments are treated as one group.
    """

    def __init__(self, data_dir, data_processor: DataProcessor, sample_rate=24000, hop_size=480):
        self.data_dir = data_dir
        self.data_processor = data_processor
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        metadata_dir = os.path.join(data_dir, "metadata")
        wavs_dir = os.path.join(data_dir, "wavs")

        self.items = []
        self.singer_to_indices = defaultdict(list)

        for fname in sorted(os.listdir(metadata_dir)):
            if not fname.endswith(".json"):
                continue
            stem = fname[:-5]
            wav_path = os.path.join(wavs_dir, stem + ".wav")
            if not os.path.isfile(wav_path):
                continue
            meta_path = os.path.join(metadata_dir, fname)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, list):
                meta = meta[0]

            singer = meta.get("singer", "default")
            idx = len(self.items)
            self.items.append({
                "meta": meta,
                "wav_path": wav_path,
                "singer": singer,
            })
            self.singer_to_indices[singer].append(idx)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Process target metadata → tensors (phoneme, note_pitch, note_type, mel2note, f0)
        # deepcopy because DataProcessor.merge_phoneme() mutates the dict in-place
        target_data = self.data_processor.process(copy.deepcopy(item["meta"]), item["wav_path"])

        # Select a prompt from the same singer (different segment if possible)
        singer = item["singer"]
        candidates = self.singer_to_indices[singer]
        if len(candidates) > 1:
            prompt_idx = random.choice([c for c in candidates if c != idx])
        else:
            prompt_idx = idx
        prompt_item = self.items[prompt_idx]

        # Process prompt
        prompt_data = self.data_processor.process(copy.deepcopy(prompt_item["meta"]), prompt_item["wav_path"])

        return {
            "target": target_data,
            "prompt": prompt_data,
        }

    @staticmethod
    def collate_fn(batch):
        """Collate a list of samples into a padded batch.

        Returns a dict ready for SoulXSinger.forward().
        """
        batch_size = len(batch)

        # Compute max lengths for padding
        prompt_mel_lens = []
        target_mel_lens = []
        note_lens = []

        for sample in batch:
            pt = sample["prompt"]
            tg = sample["target"]
            pt_mel_len = pt["mel2note"].shape[1]
            tg_mel_len = tg["mel2note"].shape[1]
            prompt_mel_lens.append(pt_mel_len)
            target_mel_lens.append(tg_mel_len)
            note_lens.append(pt["phoneme"].shape[1] + tg["phoneme"].shape[1])

        max_note_len = max(note_lens)
        max_prompt_mel = max(prompt_mel_lens)
        max_target_mel = max(target_mel_lens)
        max_total_mel = max_prompt_mel + max_target_mel

        # Determine max waveform lengths from waveform tensors
        max_prompt_wav = 0
        max_target_wav = 0
        for sample in batch:
            if "waveform" in sample["prompt"]:
                max_prompt_wav = max(max_prompt_wav, sample["prompt"]["waveform"].shape[-1])
            if "waveform" in sample["target"]:
                max_target_wav = max(max_target_wav, sample["target"]["waveform"].shape[-1])

        device = batch[0]["target"]["phoneme"].device

        # Initialize padded tensors
        phoneme = torch.zeros(batch_size, max_note_len, dtype=torch.long, device=device)
        note_pitch = torch.zeros(batch_size, max_note_len, dtype=torch.long, device=device)
        note_type = torch.zeros(batch_size, max_note_len, dtype=torch.long, device=device)
        mel2note = torch.zeros(batch_size, max_total_mel, dtype=torch.long, device=device)
        f0 = torch.zeros(batch_size, max_total_mel, dtype=torch.float, device=device)
        mel_mask = torch.zeros(batch_size, max_total_mel, dtype=torch.float, device=device)
        is_prompt = torch.zeros(batch_size, max_total_mel, dtype=torch.float, device=device)
        prompt_mel_len_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
        prompt_waveform = torch.zeros(batch_size, 1, max_prompt_wav, device=device)
        target_waveform = torch.zeros(batch_size, 1, max_target_wav, device=device)

        for i, sample in enumerate(batch):
            pt = sample["prompt"]
            tg = sample["target"]

            pt_mel_len = pt["mel2note"].shape[1]
            tg_mel_len = tg["mel2note"].shape[1]
            pt_note_len = pt["phoneme"].shape[1]
            tg_note_len = tg["phoneme"].shape[1]
            total_note_len = pt_note_len + tg_note_len
            total_mel_len = pt_mel_len + tg_mel_len

            # Concatenate phoneme, note_pitch, note_type (prompt + target)
            concat_phoneme = torch.cat([pt["phoneme"].squeeze(0), tg["phoneme"].squeeze(0)], dim=0)
            phoneme[i, :total_note_len] = concat_phoneme

            concat_note_pitch = torch.cat([pt["note_pitch"].squeeze(0), tg["note_pitch"].squeeze(0)], dim=0)
            note_pitch[i, :total_note_len] = concat_note_pitch

            concat_note_type = torch.cat([pt["note_type"].squeeze(0), tg["note_type"].squeeze(0)], dim=0)
            note_type[i, :total_note_len] = concat_note_type

            # mel2note: target needs offset by prompt note length
            pt_m2n = pt["mel2note"].squeeze(0)
            tg_m2n = tg["mel2note"].squeeze(0) + pt_note_len
            concat_mel2note = torch.cat([pt_m2n, tg_m2n], dim=0)
            mel2note[i, :total_mel_len] = concat_mel2note

            # F0
            pt_f0 = pt["f0"].squeeze(0) if "f0" in pt else torch.zeros(pt_mel_len, device=device)
            tg_f0 = tg["f0"].squeeze(0) if "f0" in tg else torch.zeros(tg_mel_len, device=device)
            concat_f0 = torch.cat([pt_f0, tg_f0], dim=0)
            f0[i, :total_mel_len] = concat_f0

            # Masks
            mel_mask[i, :total_mel_len] = 1.0
            is_prompt[i, :pt_mel_len] = 1.0
            prompt_mel_len_tensor[i] = pt_mel_len

            # Waveforms
            if "waveform" in pt:
                wav = pt["waveform"]
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                prompt_waveform[i, :, :wav.shape[-1]] = wav
            if "waveform" in tg:
                wav = tg["waveform"]
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                target_waveform[i, :, :wav.shape[-1]] = wav

        return {
            "phoneme": phoneme,
            "note_pitch": note_pitch,
            "note_type": note_type,
            "mel2note": mel2note,
            "f0": f0,
            "mel_mask": mel_mask,
            "is_prompt": is_prompt,
            "prompt_mel_len": prompt_mel_len_tensor,
            "prompt_waveform": prompt_waveform,
            "target_waveform": target_waveform,
        }
