"""Label loading. The full PhiUSIIL file is covered by the dataset-marked test."""

from __future__ import annotations

from pathlib import Path

from phishguard.train import load_urls


def test_labels_are_flipped_so_one_means_phishing(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(
        "URL,label,unused\nhttps://ok.example,1,no\nhttp://bad.example,0,no\n",
        encoding="utf-8",
    )
    urls, labels = load_urls(csv_path)
    assert urls == ["https://ok.example", "http://bad.example"]
    assert labels == [0, 1]
