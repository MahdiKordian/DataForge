# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""DataForge core: data-preparation logic with no Streamlit dependency."""
from .splitting import balance_train, split_frame
__all__ = ["balance_train", "split_frame"]
