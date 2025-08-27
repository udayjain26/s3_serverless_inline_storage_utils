import os
import sys
import json
import subprocess
import numpy as np
import datetime
import tempfile
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from typing import Dict, Any, Tuple, Optional
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Import VHS utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ComfyUI-VideoHelperSuite", "videohelpersuite"))
from utils import ffmpeg_path, ENCODE_ARGS, imageOrLatent, floatOrInt, ContainsAll


def tensor_to_bytes(tensor):
    """Convert tensor to bytes - copied from VHS for consistency"""
    tensor = tensor.cpu().numpy() * 255 + 0.5
    return np.clip(tensor, 0, 255).astype(np.uint8)


class S3VideoCombine:
    """
    Fast video creation using VHS performance + S3 upload with inline credentials
    
    Extends VideoHelperSuite's VideoCombine with direct S3 upload capability
    for maximum performance in serverless environments.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (floatOrInt, {"default": 8, "min": 1, "step": 1}),
                "filename_prefix": ("STRING", {"default": "s3_video"}),
                "bucket_name": ("STRING", {"default": "", "tooltip": "S3 bucket name"}),
                "access_key_id": ("STRING", {"default": "", "tooltip": "AWS Access Key ID"}),
                "secret_access_key": ("STRING", {"default": "", "tooltip": "AWS Secret Access Key"}),
                "region": ("STRING", {"default": "us-east-1", "tooltip": "AWS region"}),
            },
            "optional": {
                "endpoint_url": ("STRING", {
                    "default": "", 
                    "tooltip": "Custom S3 endpoint URL (for S3-compatible services like Supabase)"
                }),
                "folder_path": ("STRING", {
                    "default": "uploads", 
                    "tooltip": "Folder path within bucket"
                }),
                "video_quality": ("INT", {
                    "default": 23,
                    "min": 15,
                    "max": 35,
                    "step": 1,
                    "tooltip": "Video quality (15=highest, 35=lowest)"
                }),
                "video_format": (["mp4", "webm", "avi", "mov"], {
                    "default": "mp4",
                    "tooltip": "Output video format"
                }),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }),
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("s3_url", "status", "file_size_mb")
    FUNCTION = "combine_and_upload"
    OUTPUT_NODE = True
    CATEGORY = "S3 Serverless Storage"
    DESCRIPTION = "Fast video creation using VHS performance + S3 upload"

    def _create_s3_client(self, access_key_id: str, secret_access_key: str, 
                         region: str, endpoint_url: Optional[str] = None) -> boto3.client:
        """Create S3 client - same as original S3VideoUpload"""
        try:
            addressing_style = 'path' if endpoint_url and 'supabase' in endpoint_url.lower() else 'virtual'
            
            config = Config(
                region_name=region,
                signature_version='s3v4',
                s3={'addressing_style': addressing_style},
                retries={'max_attempts': 5, 'mode': 'adaptive'},
                connect_timeout=60,
                read_timeout=60,
                max_pool_connections=50,
                parameter_validation=False,
                tcp_keepalive=True
            )
            
            client_kwargs = {
                'service_name': 's3',
                'aws_access_key_id': access_key_id,
                'aws_secret_access_key': secret_access_key,
                'config': config,
                'verify': True
            }
            
            if endpoint_url and endpoint_url.strip():
                client_kwargs['endpoint_url'] = endpoint_url.strip()
            
            return boto3.client(**client_kwargs)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create S3 client: {str(e)}")

    def _upload_to_s3(self, s3_client: boto3.client, local_path: str, 
                     bucket_name: str, s3_key: str, video_format: str) -> str:
        """Upload video to S3 and return URL"""
        try:
            content_type_map = {
                'mp4': 'video/mp4',
                'webm': 'video/webm',
                'avi': 'video/x-msvideo',
                'mov': 'video/quicktime'
            }
            content_type = content_type_map.get(video_format, 'video/mp4')
            
            extra_args = {'ContentType': content_type}
            s3_client.upload_file(local_path, bucket_name, s3_key, ExtraArgs=extra_args)
            
            # Generate S3 URL
            endpoint_url = s3_client._endpoint.host
            if 'supabase' in endpoint_url.lower():
                s3_url = f"{endpoint_url}/object/public/{bucket_name}/{s3_key}"
            elif endpoint_url.startswith('https://s3.'):
                s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
            else:
                s3_url = f"{endpoint_url}/{bucket_name}/{s3_key}"
            
            return s3_url
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise RuntimeError(f"Bucket '{bucket_name}' does not exist")
            elif error_code == 'AccessDenied':
                raise RuntimeError("Access denied. Check your credentials and bucket permissions")
            else:
                raise RuntimeError(f"S3 upload failed: {str(e)}")

    def combine_and_upload(self, images, frame_rate: float, filename_prefix: str,
                          bucket_name: str, access_key_id: str, secret_access_key: str,
                          region: str, endpoint_url: str = "", folder_path: str = "uploads",
                          video_quality: int = 23, video_format: str = "mp4",
                          prompt=None, extra_pnginfo=None, unique_id=None):
        """
        Fast video creation using VHS streaming approach + S3 upload
        """
        try:
            # Validate inputs
            if not all([bucket_name.strip(), access_key_id.strip(), secret_access_key.strip()]):
                return "", "Error: Missing S3 credentials", 0
                
            if images is None or len(images) == 0:
                return "", "Error: No images provided", 0
            
            if ffmpeg_path is None:
                return "", "Error: FFmpeg not found", 0

            # Get first image for dimensions - VHS style
            first_image = images[0]
            images_iter = iter(images)
            
            # Convert to numpy for consistency with VHS
            images_bytes = map(tensor_to_bytes, images_iter)
            
            # Create temporary video file
            temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=f".{video_format}")
            os.close(temp_video_fd)
            
            try:
                # Get dimensions
                dimensions = (first_image.shape[1], first_image.shape[0])  # Width, Height
                
                # Build FFmpeg command - VHS streaming style
                args = [
                    ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{dimensions[0]}x{dimensions[1]}", 
                    "-r", str(frame_rate), 
                    "-i", "-"  # Read from stdin
                ]
                
                # Add encoding options
                args += [
                    "-c:v", "libx264",
                    "-crf", str(video_quality),
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",  # Optimize for web
                    temp_video_path
                ]
                
                # Stream to FFmpeg - VHS approach
                env = os.environ.copy()
                with subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE, env=env) as proc:
                    try:
                        # Convert first image and add it back to the stream
                        first_image_bytes = tensor_to_bytes(first_image).tobytes()
                        proc.stdin.write(first_image_bytes)
                        
                        # Stream remaining images
                        for image_bytes in images_bytes:
                            proc.stdin.write(image_bytes.tobytes())
                        
                        proc.stdin.close()
                        stdout, stderr = proc.communicate(timeout=120)
                        
                        if proc.returncode != 0:
                            raise RuntimeError(f"FFmpeg failed: {stderr.decode(*ENCODE_ARGS)}")
                            
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        raise RuntimeError("FFmpeg process timed out")
                    except BrokenPipeError:
                        stdout, stderr = proc.communicate()
                        raise RuntimeError(f"FFmpeg pipe error: {stderr.decode(*ENCODE_ARGS)}")
                
                # Check output file
                if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                    raise RuntimeError("Failed to create video file")
                
                file_size_bytes = os.path.getsize(temp_video_path)
                file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
                
                # Upload to S3
                s3_client = self._create_s3_client(access_key_id, secret_access_key, region, endpoint_url)
                
                folder_path = folder_path.strip().strip('/')
                if folder_path:
                    folder_path += '/'
                
                filename = f"{filename_prefix}.{video_format}"
                s3_key = f"{folder_path}{filename}"
                
                s3_url = self._upload_to_s3(s3_client, temp_video_path, bucket_name, s3_key, video_format)
                
                status_msg = f"Success: {len(images)} frames → {file_size_mb}MB video uploaded to S3"
                return s3_url, status_msg, int(file_size_mb)
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    try:
                        os.unlink(temp_video_path)
                    except OSError:
                        pass
                        
        except Exception as e:
            error_msg = f"Failed: {str(e)}"
            return "", error_msg, 0


# Node mapping
NODE_CLASS_MAPPINGS = {
    "S3VideoCombine": S3VideoCombine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S3VideoCombine": "S3 Video Combine (VHS Performance)"
}