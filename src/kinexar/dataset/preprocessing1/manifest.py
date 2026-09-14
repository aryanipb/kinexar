from __future__ import annotations

from pathlib import Path

from kinexar.dataset.preprocessing.pipeline import av2_shared
from kinexar.dataset.preprocessing.pipeline import dump


def build_shared_manifest(
    av2_root: str | Path,
    output_path: str | Path,
    *,
    history_steps: int = 50,
    future_steps: int = 60,
    force: bool = False,
) -> Path:
    av2_root = Path(av2_root).expanduser()
    output_path = Path(output_path).expanduser()
    if output_path.exists() and not force:
        return output_path
    dump(
        av2_shared(
            av2_root,
            history_steps=int(history_steps),
            future_steps=int(future_steps),
        ),
        output_path,
    )
    return output_path
