"""ComfyUI nodes for local and API-backed multimodal LLM inference."""

from .generic_nodes import (
    NODE_CLASS_MAPPINGS as GENERIC_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GENERIC_NODE_DISPLAY_NAME_MAPPINGS,
)

# Register one clear, unified node set. Model discovery helpers are kept
# separate and do not register any additional nodes.
NODE_CLASS_MAPPINGS = dict(GENERIC_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(GENERIC_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
