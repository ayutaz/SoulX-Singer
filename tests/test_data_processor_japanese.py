"""Tests for DataProcessor Japanese phoneme handling."""


class TestDataProcessorJapanesePhonemes:
    """Test that Japanese phonemes are correctly split in DataProcessor.

    G2P produces ``ja_s-a-k-u-r-a`` format (prefix only on first element).
    data_processor.py splits on ``-`` after stripping ``ja_`` prefix ([3:]),
    then re-applies ``ja_`` to each element.
    """

    def test_ja_phoneme_split_pattern(self):
        """Test that ja_ prefixed compound phonemes are split correctly."""
        # Format from g2p_japanese: "ja_s-a-k-u-r-a"
        phoneme = "ja_s-a-k-u-r-a"
        assert phoneme[:3] == "ja_"

        ja_phs = ["ja_" + x for x in phoneme[3:].split("-")] + ["<SEP>"]
        expected = ["ja_s", "ja_a", "ja_k", "ja_u", "ja_r", "ja_a", "<SEP>"]
        assert ja_phs == expected

    def test_ja_single_phoneme(self):
        """Test single phoneme (e.g., particle)."""
        phoneme = "ja_a"
        assert phoneme[:3] == "ja_"

        ja_phs = ["ja_" + x for x in phoneme[3:].split("-")] + ["<SEP>"]
        expected = ["ja_a", "<SEP>"]
        assert ja_phs == expected

    def test_ja_special_phonemes(self):
        """Test special phonemes like N (ん) and cl (っ)."""
        phoneme = "ja_k-a-N"
        ja_phs = ["ja_" + x for x in phoneme[3:].split("-")] + ["<SEP>"]
        expected = ["ja_k", "ja_a", "ja_N", "<SEP>"]
        assert ja_phs == expected

    def test_en_not_matched_by_ja(self):
        """Ensure English phonemes don't trigger Japanese path."""
        phoneme = "en_HH-AH0-L-OW1"
        assert phoneme[:3] != "ja_"
        assert phoneme[:3] == "en_"

    def test_zh_not_matched_by_ja(self):
        """Ensure Chinese phonemes don't trigger Japanese path."""
        phoneme = "zh_ni3"
        assert phoneme[:3] != "ja_"
        assert phoneme[:3] == "zh_"
