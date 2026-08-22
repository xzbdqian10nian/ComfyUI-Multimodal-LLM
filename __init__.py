"""ComfyUI nodes for local and API-backed multimodal LLM inference."""

from .generic_nodes import (
    NODE_CLASS_MAPPINGS as GENERIC_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GENERIC_NODE_DISPLAY_NAME_MAPPINGS,
)

# Only expose the unified nodes.  The previous Qwen38* classes remain in
# ``nodes.py`` as implementation code for now, but are deliberately not
# registered with ComfyUI: registering them creates a second, near-identical
# set of nodes (the deprecated flag is only metadata and does not hide a node
# in every frontend version).  This keeps the node menu unambiguous.
NODE_CLASS_MAPPINGS = dict(GENERIC_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(GENERIC_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
