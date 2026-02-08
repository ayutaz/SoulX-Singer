"""Tests for Japanese G2P support."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


def _make_fullcontext_labels(phonemes):
    """Create mock HTS fullcontext labels from a list of phoneme strings."""
    labels = []
    for ph in ["sil"] + phonemes + ["sil"]:
        labels.append(f"xx^xx-{ph}+xx=xx")
    return labels


def _patch_pyopenjtalk(fullcontext_return):
    """Create a mock pyopenjtalk and inject it into the g2p module."""
    mock_jtalk = MagicMock()
    mock_jtalk.extract_fullcontext.return_value = fullcontext_return
    return patch("preprocess.tools.g2p._pyopenjtalk", mock_jtalk), mock_jtalk


class TestG2pJapanese:
    """Test g2p_japanese function."""

    def test_basic_word(self):
        """Test basic Japanese word conversion."""
        from preprocess.tools.g2p import g2p_japanese

        patcher, mock_jtalk = _patch_pyopenjtalk(
            _make_fullcontext_labels(["s", "a", "k", "u", "r", "a"])
        )
        with patcher:
            result = g2p_japanese("さくら")
            assert result == "ja_s-a-k-u-r-a"
            mock_jtalk.extract_fullcontext.assert_called_once_with("さくら")

    def test_empty_result_returns_sp(self):
        """If pyopenjtalk returns only sil/pau, return <SP>."""
        from preprocess.tools.g2p import g2p_japanese

        patcher, _ = _patch_pyopenjtalk(_make_fullcontext_labels([]))
        with patcher:
            result = g2p_japanese("。")
            assert result == "<SP>"

    def test_pau_filtered(self):
        """Pause phonemes should be filtered out."""
        from preprocess.tools.g2p import g2p_japanese

        patcher, _ = _patch_pyopenjtalk(
            _make_fullcontext_labels(["k", "a", "pau", "k", "i"])
        )
        with patcher:
            result = g2p_japanese("かき")
            assert result == "ja_k-a-k-i"

    def test_special_phonemes(self):
        """Test special phonemes like N and cl."""
        from preprocess.tools.g2p import g2p_japanese

        patcher, _ = _patch_pyopenjtalk(
            _make_fullcontext_labels(["n", "i", "h", "o", "N"])
        )
        with patcher:
            result = g2p_japanese("にほん")
            assert result == "ja_n-i-h-o-N"


class TestG2pTransformJapanese:
    """Test g2p_transform with Japanese language."""

    def test_japanese_routing(self):
        """Japanese words should be routed through g2p_japanese."""
        from preprocess.tools.g2p import g2p_transform

        patcher, _ = _patch_pyopenjtalk(
            _make_fullcontext_labels(["s", "a", "k", "u", "r", "a"])
        )
        with patcher:
            result = g2p_transform(["さくら"], "Japanese")
            assert result == ["ja_s-a-k-u-r-a"]

    def test_sp_preserved(self):
        """<SP> tokens should pass through unchanged."""
        from preprocess.tools.g2p import g2p_transform

        result = g2p_transform(["<SP>"], "Japanese")
        assert result == ["<SP>"]

    def test_punctuation_stripped(self):
        """Punctuation should be stripped before G2P."""
        from preprocess.tools.g2p import g2p_transform

        patcher, mock_jtalk = _patch_pyopenjtalk(
            _make_fullcontext_labels(["k", "a"])
        )
        with patcher:
            g2p_transform(["か?"], "Japanese")
            mock_jtalk.extract_fullcontext.assert_called_once_with("か")

    def test_does_not_affect_mandarin(self):
        """Mandarin routing should be unaffected."""
        from preprocess.tools.g2p import g2p_transform

        # Mandarin path should NOT use pyopenjtalk
        patcher, mock_jtalk = _patch_pyopenjtalk(
            _make_fullcontext_labels([])
        )
        with patcher:
            g2p_transform(["<SP>"], "Mandarin")
            mock_jtalk.extract_fullcontext.assert_not_called()
