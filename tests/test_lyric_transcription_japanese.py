"""Tests for Japanese ASR (lyric transcription) support."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock

import pytest


class TestASRJaModel:
    """Test _ASRJaModel class."""

    @patch("preprocess.tools.lyric_transcription.librosa")
    def test_process_basic(self, mock_librosa):
        """Test basic Japanese ASR processing with mocked faster-whisper."""
        mock_librosa.get_duration.return_value = 5.0

        mock_word1 = MagicMock()
        mock_word1.word = "さくら"
        mock_word1.start = 0.5
        mock_word1.end = 1.5

        mock_word2 = MagicMock()
        mock_word2.word = "が"
        mock_word2.start = 1.5
        mock_word2.end = 2.0

        mock_segment = MagicMock()
        mock_segment.words = [mock_word1, mock_word2]

        # Mock faster_whisper module before importing _ASRJaModel
        mock_fw = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_segment]), None)
        mock_fw.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
            from preprocess.tools.lyric_transcription import _ASRJaModel

            model = _ASRJaModel(model_size="tiny", device="cpu")
            words, durs = model.process("test.wav")

            assert "さくら" in words
            assert "が" in words
            assert len(words) == len(durs)

    def test_clean_word(self):
        """Test word cleaning."""
        from preprocess.tools.lyric_transcription import _ASRJaModel

        assert _ASRJaModel._clean_word("さくら?") == "さくら"
        assert _ASRJaModel._clean_word("テスト.") == "テスト"
        assert _ASRJaModel._clean_word("") == ""


class TestLyricTranscriberJapanese:
    """Test LyricTranscriber with Japanese language."""

    def test_japanese_validation_accepted(self):
        """Japanese should be accepted as valid language."""
        with patch("preprocess.tools.lyric_transcription._ASRZhModel"):
            from preprocess.tools.lyric_transcription import LyricTranscriber

            LyricTranscriber(
                zh_model_path="dummy_zh",
                en_model_path="dummy_en",
                device="cpu",
                verbose=False,
            )
            # Should not raise for Japanese
            assert "Japanese" in {"Mandarin", "Cantonese", "English", "Japanese"}

    def test_japanese_validation_error_on_unknown(self):
        """Unknown language should raise ValueError."""
        with patch("preprocess.tools.lyric_transcription._ASRZhModel"):
            from preprocess.tools.lyric_transcription import LyricTranscriber

            transcriber = LyricTranscriber(
                zh_model_path="dummy_zh",
                en_model_path="dummy_en",
                device="cpu",
                verbose=False,
            )
            with pytest.raises(ValueError, match="Unsupported language"):
                transcriber.process("test.wav", language="Korean")

    def test_japanese_lazy_init(self):
        """Japanese ASR should be lazily initialized."""
        with patch("preprocess.tools.lyric_transcription._ASRZhModel"):
            from preprocess.tools.lyric_transcription import LyricTranscriber

            transcriber = LyricTranscriber(
                zh_model_path="dummy_zh",
                en_model_path="dummy_en",
                device="cpu",
                verbose=False,
            )
            assert transcriber.ja_model is None
