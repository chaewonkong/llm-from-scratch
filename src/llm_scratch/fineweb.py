from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_PARQUET = Path.home() / "data/fineweb-edu/sample/10BT/000_00000.parquet"


def load_text(path: Path, n_bytes: int) -> str:
    """Read row groups until we have n_bytes of UTF-8 text (never loads the whole 2GB file)."""
    pf = pq.ParquetFile(path)
    docs, size = [], 0
    for rg in range(pf.num_row_groups):
        for doc in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            docs.append(doc)
            size += len(doc.encode("utf-8"))
            if size >= n_bytes:
                return "\n\n".join(docs)
    return "\n\n".join(docs)
