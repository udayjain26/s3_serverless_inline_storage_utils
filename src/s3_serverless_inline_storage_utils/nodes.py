import os
import json
import tempfile
import hashlib
import shutil
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config
import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))), "comfy"))

import comfy.sd
import comfy.utils
import folder_paths


class S3ImageUpload:
    """
    Upload images directly to S3 with inline credentials
    
    A production-ready node for uploading images to any S3-compatible storage
    without relying on environment variables. All credentials are provided
    directly in the workflow for maximum flexibility.
    """
    
    def __init__(self):
        self.compress_level = 4
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to upload to S3"}),
                "filename_prefix": ("STRING", {
                    "default": "comfy_image", 
                    "tooltip": "Prefix for uploaded filenames"
                }),
                "bucket_name": ("STRING", {
                    "default": "", 
                    "tooltip": "S3 bucket name"
                }),
                "access_key_id": ("STRING", {
                    "default": "", 
                    "tooltip": "AWS Access Key ID"
                }),
                "secret_access_key": ("STRING", {
                    "default": "", 
                    "tooltip": "AWS Secret Access Key"
                }),
                "region": ("STRING", {
                    "default": "us-east-1", 
                    "tooltip": "AWS region"
                }),
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
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("s3_urls", "upload_status", "uploaded_count")
    FUNCTION = "upload_images"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, False, False)
    CATEGORY = "S3 Serverless Storage"
    DESCRIPTION = "Upload images to S3 with inline credentials"

    def _create_s3_client(self, access_key_id: str, secret_access_key: str, 
                         region: str, endpoint_url: Optional[str] = None) -> boto3.client:
        """Create and return S3 client with provided credentials"""
        try:
            addressing_style = 'path' if endpoint_url and 'supabase' in endpoint_url.lower() else 'virtual'
            
            config = Config(
                region_name=region,
                signature_version='s3v4',
                s3={'addressing_style': addressing_style},
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                }
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

    def _validate_inputs(self, bucket_name: str, access_key_id: str, 
                        secret_access_key: str) -> None:
        """Validate required inputs"""
        if not bucket_name.strip():
            raise ValueError("Bucket name is required")
        if not access_key_id.strip():
            raise ValueError("Access Key ID is required")
        if not secret_access_key.strip():
            raise ValueError("Secret Access Key is required")

    def _generate_filename(self, prefix: str) -> str:
        """Generate filename using only the prefix"""
        return f"{prefix}.webp"

    def _save_image(self, image: np.ndarray, file_path: str) -> None:
        """Save image to temporary file as WebP with maximum quality"""
        i = 255.0 * image
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        img.save(file_path, format="WEBP", quality=100, method=6, lossless=False)

    def _upload_to_s3(self, s3_client: boto3.client, local_path: str, 
                     bucket_name: str, s3_key: str) -> str:
        """Upload file to S3 and return the S3 URL"""
        try:
            extra_args = {
                'ContentType': 'image/webp'
            }
            
            s3_client.upload_file(local_path, bucket_name, s3_key, ExtraArgs=extra_args)
            
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
        except Exception as e:
            raise RuntimeError(f"Upload failed: {str(e)}")

    def upload_images(self, images, filename_prefix: str, bucket_name: str,
                     access_key_id: str, secret_access_key: str, region: str,
                     endpoint_url: str = "", folder_path: str = "uploads",
                     prompt: Optional[Dict] = None,
                     extra_pnginfo: Optional[Dict] = None) -> Tuple[list, str, int]:
        """
        Main upload function
        
        Returns:
            - List of S3 URLs for uploaded images
            - Upload status message
            - Number of successfully uploaded images
        """
        try:
            self._validate_inputs(bucket_name, access_key_id, secret_access_key)
            
            s3_client = self._create_s3_client(
                access_key_id, secret_access_key, region, endpoint_url
            )
            
            folder_path = folder_path.strip().strip('/')
            if folder_path:
                folder_path += '/'
            
            s3_urls = []
            uploaded_count = 0
            
            for i, image in enumerate(images):
                temp_file = None
                try:
                    # Handle both tensor and numpy array inputs
                    if hasattr(image, 'cpu'):
                        image_np = image.cpu().numpy()
                    else:
                        image_np = image
                    
                    if len(images) > 1:
                        filename = self._generate_filename(f"{filename_prefix}_{i+1}")
                    else:
                        filename = self._generate_filename(filename_prefix)
                    s3_key = f"{folder_path}{filename}"
                    
                    with tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix=".webp"
                    ) as temp_file:
                        temp_file_path = temp_file.name
                        
                        self._save_image(image_np, temp_file_path)
                        
                        s3_url = self._upload_to_s3(
                            s3_client, temp_file_path, bucket_name, s3_key
                        )
                        
                        s3_urls.append(s3_url)
                        uploaded_count += 1
                        
                finally:
                    if temp_file and os.path.exists(temp_file.name):
                        try:
                            os.unlink(temp_file.name)
                        except OSError:
                            pass
            
            status_msg = f"Successfully uploaded {uploaded_count}/{len(images)} images to S3"
            return s3_urls, status_msg, uploaded_count
            
        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            print(f"S3ImageUpload Error: {error_msg}")
            return [], error_msg, 0


class S3VideoUpload:
    """
    Upload videos directly to S3 with inline credentials
    
    A production-ready node for uploading videos to any S3-compatible storage
    without relying on environment variables.
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"tooltip": "Path to video file to upload"}),
                "filename_prefix": ("STRING", {
                    "default": "comfy_video", 
                    "tooltip": "Prefix for uploaded filename"
                }),
                "bucket_name": ("STRING", {
                    "default": "", 
                    "tooltip": "S3 bucket name"
                }),
                "access_key_id": ("STRING", {
                    "default": "", 
                    "tooltip": "AWS Access Key ID"
                }),
                "secret_access_key": ("STRING", {
                    "default": "", 
                    "tooltip": "AWS Secret Access Key"
                }),
                "region": ("STRING", {
                    "default": "us-east-1", 
                    "tooltip": "AWS region"
                }),
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
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("s3_url", "upload_status", "file_size_mb")
    FUNCTION = "upload_video"
    OUTPUT_NODE = True
    CATEGORY = "S3 Serverless Storage"
    DESCRIPTION = "Upload video to S3 with inline credentials"

    def _create_s3_client(self, access_key_id: str, secret_access_key: str, 
                         region: str, endpoint_url: Optional[str] = None) -> boto3.client:
        """Create and return S3 client with provided credentials"""
        try:
            addressing_style = 'path' if endpoint_url and 'supabase' in endpoint_url.lower() else 'virtual'
            
            config = Config(
                region_name=region,
                signature_version='s3v4',
                s3={'addressing_style': addressing_style},
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                }
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

    def _validate_inputs(self, bucket_name: str, access_key_id: str, 
                        secret_access_key: str, video_path: str) -> None:
        """Validate required inputs"""
        if not bucket_name.strip():
            raise ValueError("Bucket name is required")
        if not access_key_id.strip():
            raise ValueError("Access Key ID is required")
        if not secret_access_key.strip():
            raise ValueError("Secret Access Key is required")
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

    def _generate_video_filename(self, prefix: str, original_path: str) -> str:
        """Generate filename using only the prefix and original extension"""
        file_ext = os.path.splitext(original_path)[1].lower()
        if not file_ext:
            file_ext = '.mp4'
        
        return f"{prefix}{file_ext}"

    def _upload_video_to_s3(self, s3_client: boto3.client, local_path: str, 
                           bucket_name: str, s3_key: str) -> str:
        """Upload video file to S3 and return the S3 URL"""
        try:
            file_ext = os.path.splitext(local_path)[1].lower()
            content_type_map = {
                '.mp4': 'video/mp4',
                '.avi': 'video/x-msvideo',
                '.mov': 'video/quicktime',
                '.wmv': 'video/x-ms-wmv',
                '.flv': 'video/x-flv',
                '.webm': 'video/webm'
            }
            content_type = content_type_map.get(file_ext, 'video/mp4')
            
            extra_args = {
                'ContentType': content_type
            }
            
            s3_client.upload_file(local_path, bucket_name, s3_key, ExtraArgs=extra_args)
            
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
        except Exception as e:
            raise RuntimeError(f"Upload failed: {str(e)}")

    def upload_video(self, video: str, filename_prefix: str, bucket_name: str,
                    access_key_id: str, secret_access_key: str, region: str,
                    endpoint_url: str = "", folder_path: str = "uploads") -> Tuple[str, str, int]:
        """
        Main video upload function
        
        Returns:
            - S3 URL for uploaded video
            - Upload status message
            - File size in MB
        """
        try:
            self._validate_inputs(bucket_name, access_key_id, secret_access_key, video)
            
            s3_client = self._create_s3_client(
                access_key_id, secret_access_key, region, endpoint_url
            )
            
            folder_path = folder_path.strip().strip('/')
            if folder_path:
                folder_path += '/'
            
            filename = self._generate_video_filename(filename_prefix, video)
            s3_key = f"{folder_path}{filename}"
            
            file_size_bytes = os.path.getsize(video)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            
            s3_url = self._upload_video_to_s3(s3_client, video, bucket_name, s3_key)
            
            status_msg = f"Successfully uploaded video ({file_size_mb}MB) to S3"
            return s3_url, status_msg, int(file_size_mb)
            
        except Exception as e:
            error_msg = f"Video upload failed: {str(e)}"
            print(f"S3VideoUpload Error: {error_msg}")
            return "", error_msg, 0


class S3ImageLoad:
    """
    Load images from S3 using signed URLs or public URLs
    
    A production-ready node for downloading images from S3-compatible storage
    using direct URLs or signed URLs.
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "", 
                    "tooltip": "S3 URL or signed URL to image"
                }),
            },
            "optional": {
                "timeout": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 300,
                    "tooltip": "Request timeout in seconds"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "filename", "status")
    FUNCTION = "load_image"
    CATEGORY = "S3 Serverless Storage"
    DESCRIPTION = "Load image from S3 URL or signed URL"

    def _download_image(self, url: str, timeout: int) -> Tuple[str, str]:
        """Download image from URL and return temp file path and filename"""
        try:
            headers = {
                'User-Agent': 'ComfyUI-S3-ImageLoad/1.0'
            }
            
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
            
            filename_from_url = os.path.basename(url.split('?')[0])
            if not filename_from_url or '.' not in filename_from_url:
                ext_map = {
                    'image/jpeg': '.jpg',
                    'image/jpg': '.jpg', 
                    'image/png': '.png',
                    'image/webp': '.webp',
                    'image/gif': '.gif'
                }
                ext = ext_map.get(content_type, '.jpg')
                filename_from_url = f"downloaded_image{ext}"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename_from_url)[1]) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            return temp_file_path, filename_from_url
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Request timed out after {timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download image: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Download error: {str(e)}")

    def _load_image_from_path(self, image_path: str) -> np.ndarray:
        """Load image from file path and convert to ComfyUI format"""
        try:
            img = Image.open(image_path)
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img).astype(np.float32) / 255.0
            
            if len(img_array.shape) == 3:
                img_array = img_array[None, ...]
            
            return img_array
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {str(e)}")

    def load_image(self, url: str, timeout: int = 30) -> Tuple[np.ndarray, str, str]:
        """
        Main image loading function
        
        Returns:
            - Image as numpy array in ComfyUI format
            - Original filename
            - Status message
        """
        temp_file_path = None
        try:
            if not url.strip():
                raise ValueError("URL is required")
            
            url = url.strip()
            
            temp_file_path, filename = self._download_image(url, timeout)
            
            image_array = self._load_image_from_path(temp_file_path)
            
            status_msg = f"Successfully loaded image: {filename}"
            return image_array, filename, status_msg
            
        except Exception as e:
            error_msg = f"Image load failed: {str(e)}"
            print(f"S3ImageLoad Error: {error_msg}")
            
            dummy_image = np.zeros((1, 64, 64, 3), dtype=np.float32)
            return dummy_image, "error.jpg", error_msg
            
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass


class StringCheckpointLoader:
    """
    Load checkpoint by providing a string filename
    
    This node allows you to load a checkpoint/safetensors file by providing
    the filename as a string input, rather than selecting from a dropdown.
    It looks for the file in the default checkpoints directory.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_filename": ("STRING", {
                    "default": "",
                    "tooltip": "Filename of the checkpoint to load (e.g., 'model.safetensors')"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space."
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "S3 Serverless Storage"
    DESCRIPTION = "Load a checkpoint by providing the filename as a string input"

    def load_checkpoint(self, ckpt_filename: str):
        """
        Load checkpoint from the default checkpoints directory using string filename
        
        Args:
            ckpt_filename: The filename of the checkpoint to load
            
        Returns:
            Tuple containing (MODEL, CLIP, VAE)
        """
        try:
            if not ckpt_filename.strip():
                raise ValueError("Checkpoint filename is required")
            
            filename = ckpt_filename.strip()
            
            # Get the full path using folder_paths
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", filename)
            
            # Load the checkpoint using ComfyUI's standard loading mechanism
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path, 
                output_vae=True, 
                output_clip=True, 
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            
            return out[:3]  # Return MODEL, CLIP, VAE
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Checkpoint file '{filename}' not found in checkpoints directory. {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{filename}': {str(e)}")


class StringLoraLoader:
    """
    Load LoRA by providing a string filename
    
    This node allows you to load a LoRA file by providing the filename as a string input,
    rather than selecting from a dropdown. It looks for the file in the default loras directory.
    """
    
    def __init__(self):
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_filename": ("STRING", {
                    "default": "",
                    "tooltip": "Filename of the LoRA to load (e.g., 'lora_name.safetensors')"
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, 
                    "min": -100.0, 
                    "max": 100.0, 
                    "step": 0.01, 
                    "tooltip": "How strongly to modify the diffusion model. This value can be negative."
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0, 
                    "min": -100.0, 
                    "max": 100.0, 
                    "step": 0.01, 
                    "tooltip": "How strongly to modify the CLIP model. This value can be negative."
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"
    CATEGORY = "S3 Serverless Storage"
    DESCRIPTION = "Load a LoRA by providing the filename as a string input"

    def load_lora(self, model, clip, lora_filename: str, strength_model: float, strength_clip: float):
        """
        Load LoRA from the default loras directory using string filename
        
        Args:
            model: The diffusion model to apply LoRA to
            clip: The CLIP model to apply LoRA to
            lora_filename: The filename of the LoRA to load (can be empty for no LoRA)
            strength_model: Strength for model modification
            strength_clip: Strength for CLIP modification
            
        Returns:
            Tuple containing (MODEL, CLIP)
        """
        try:
            # Handle empty filename or zero strengths - just return original models
            if strength_model == 0 and strength_clip == 0:
                return (model, clip)
            
            filename = lora_filename.strip()
            if not filename:
                # Empty string means no LoRA, return original models
                return (model, clip)
            
            # Get the full path using folder_paths
            lora_path = folder_paths.get_full_path_or_raise("loras", filename)
            
            # Load LoRA with caching (exactly like original LoraLoader)
            lora = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    self.loaded_lora = None
            
            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            
            # Apply LoRA to model and clip using the same method as original LoraLoader
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
            return (model_lora, clip_lora)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"LoRA file '{filename}' not found in loras directory. {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA '{filename}': {str(e)}")


NODE_CLASS_MAPPINGS = {
    "S3ImageUpload": S3ImageUpload,
    "S3VideoUpload": S3VideoUpload,
    "S3ImageLoad": S3ImageLoad,
    "StringCheckpointLoader": StringCheckpointLoader,
    "StringLoraLoader": StringLoraLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S3ImageUpload": "S3 Image Upload (Inline Credentials)",
    "S3VideoUpload": "S3 Video Upload (Inline Credentials)", 
    "S3ImageLoad": "S3 Image Load from URL",
    "StringCheckpointLoader": "String Checkpoint Loader",
    "StringLoraLoader": "String LoRA Loader"
}
