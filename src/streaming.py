
from __future__ import annotations
import time
from typing import Iterator, Tuple
import pandas as pd

def stream_series(df: pd.DataFrame, delay_sec: float = 0.05, batch_size: int = 1) -> Iterator[Tuple[pd.DatetimeIndex, pd.Series]]:
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i : i + batch_size]
        yield chunk.index, chunk["value"]
        time.sleep(delay_sec)
