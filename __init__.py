"""Top-level package for s3_serverless_inline_storage_utils."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    
]

__author__ = """Uday Jain"""
__email__ = "uj9.uday@gmail.com"
__version__ = "0.0.1"

from .src.s3_serverless_inline_storage_utils.nodes import NODE_CLASS_MAPPINGS as NODES_MAPPINGS
from .src.s3_serverless_inline_storage_utils.nodes import NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_MAPPINGS
from .src.s3_serverless_inline_storage_utils.s3_video_combine import NODE_CLASS_MAPPINGS as S3_VIDEO_MAPPINGS
from .src.s3_serverless_inline_storage_utils.s3_video_combine import NODE_DISPLAY_NAME_MAPPINGS as S3_VIDEO_DISPLAY_MAPPINGS

# Combine all node mappings
NODE_CLASS_MAPPINGS = {**NODES_MAPPINGS, **S3_VIDEO_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY_MAPPINGS, **S3_VIDEO_DISPLAY_MAPPINGS}


