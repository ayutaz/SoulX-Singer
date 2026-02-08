"""Convert preprocessing pipeline outputs into training dataset format.

The preprocessing pipeline produces:
    song_dir/
    ├── metadata.json           # Array of segment dicts
    └── long_cut_wavs/
        ├── vocal_0_6710.wav    # Named by time range (= index field)
        └── vocal_6710_12000.wav

This script converts one or more such directories into the format
expected by SVSDataset:
    dataset_dir/
    ├── metadata/
    │   ├── song_001_seg_000.json
    │   └── ...
    └── wavs/
        ├── song_001_seg_000.wav
        └── ...

Usage:
    # Single song
    uv run python -m cli.prepare_dataset \
        --sources data/preprocessed/song_001 \
        --output data/dataset

    # Multiple songs
    uv run python -m cli.prepare_dataset \
        --sources data/preprocessed/song_001 data/preprocessed/song_002 \
        --output data/dataset

    # All subdirectories under a parent (--recursive)
    uv run python -m cli.prepare_dataset \
        --sources data/preprocessed \
        --output data/dataset \
        --recursive

    # Assign singer names from directory names (default)
    # song_001/ → singer="song_001"

    # Explicit singer name for all sources
    uv run python -m cli.prepare_dataset \
        --sources data/preprocessed/song_001 \
        --output data/dataset \
        --singer "singer_A"
"""

import os
import json
import shutil
import argparse
import logging

logger = logging.getLogger(__name__)


def find_source_dirs(sources, recursive=False):
    """Resolve source directories containing metadata.json.

    Args:
        sources: List of paths (files or directories).
        recursive: If True and a source has no metadata.json,
                   scan its immediate subdirectories.

    Returns:
        List of (source_dir, singer_name) tuples.
    """
    result = []
    for src in sources:
        src = os.path.abspath(src)
        meta_path = os.path.join(src, "metadata.json")
        if os.path.isfile(meta_path):
            singer = os.path.basename(src)
            result.append((src, singer))
        elif recursive and os.path.isdir(src):
            for child in sorted(os.listdir(src)):
                child_path = os.path.join(src, child)
                child_meta = os.path.join(child_path, "metadata.json")
                if os.path.isdir(child_path) and os.path.isfile(child_meta):
                    result.append((child_path, child))
        else:
            logger.warning(f"Skipping {src}: no metadata.json found")
    return result


def convert_source(source_dir, singer, output_dir, prefix, start_idx=0):
    """Convert a single preprocessing output directory.

    Args:
        source_dir: Path to preprocessed song directory.
        singer: Singer name to embed in metadata.
        output_dir: Root output dataset directory.
        prefix: Filename prefix for this source (e.g. "song_001").
        start_idx: Starting segment index for numbering.

    Returns:
        Number of segments written.
    """
    meta_path = os.path.join(source_dir, "metadata.json")
    wavs_dir = os.path.join(source_dir, "long_cut_wavs")

    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not isinstance(segments, list):
        segments = [segments]

    out_meta_dir = os.path.join(output_dir, "metadata")
    out_wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(out_meta_dir, exist_ok=True)
    os.makedirs(out_wavs_dir, exist_ok=True)

    count = 0
    for i, seg in enumerate(segments):
        # Find source wav: long_cut_wavs/{index}.wav
        index = seg.get("index", "")
        src_wav = os.path.join(wavs_dir, f"{index}.wav")
        if not os.path.isfile(src_wav):
            logger.warning(
                f"Skipping segment {index}: wav not found at {src_wav}"
            )
            continue

        seg_name = f"{prefix}_seg_{start_idx + i:03d}"

        # Write individual segment metadata with singer field
        out_seg = dict(seg)
        out_seg["singer"] = singer
        out_meta_path = os.path.join(out_meta_dir, f"{seg_name}.json")
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(out_seg, f, ensure_ascii=False, indent=2)

        # Copy wav
        out_wav_path = os.path.join(out_wavs_dir, f"{seg_name}.wav")
        shutil.copy2(src_wav, out_wav_path)

        count += 1

    return count


def prepare_dataset(sources, output_dir, recursive=False, singer=None):
    """Main entry point for dataset preparation.

    Args:
        sources: List of source directory paths.
        output_dir: Output dataset directory.
        recursive: Whether to scan subdirectories.
        singer: Optional explicit singer name (overrides directory-based).

    Returns:
        Total number of segments written.
    """
    source_dirs = find_source_dirs(sources, recursive=recursive)

    if not source_dirs:
        logger.error("No valid source directories found.")
        return 0

    total = 0
    for idx, (src_dir, auto_singer) in enumerate(source_dirs):
        effective_singer = singer if singer else auto_singer
        prefix = os.path.basename(src_dir)

        count = convert_source(
            source_dir=src_dir,
            singer=effective_singer,
            output_dir=output_dir,
            prefix=prefix,
        )
        logger.info(f"  {src_dir}: {count} segments (singer={effective_singer})")
        total += count

    logger.info(f"Total: {total} segments written to {output_dir}")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Convert preprocessing outputs to training dataset format"
    )
    parser.add_argument(
        "--sources", type=str, nargs="+", required=True,
        help="One or more preprocessed song directories"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output dataset directory"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Scan subdirectories of each source for metadata.json"
    )
    parser.add_argument(
        "--singer", type=str, default=None,
        help="Explicit singer name (default: use directory name)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    prepare_dataset(
        sources=args.sources,
        output_dir=args.output,
        recursive=args.recursive,
        singer=args.singer,
    )


if __name__ == "__main__":
    main()
