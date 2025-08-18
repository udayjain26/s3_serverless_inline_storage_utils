# S3 Serverless-Inline Storage Utils

**Production-ready ComfyUI nodes for S3-compatible storage with inline credentials**

Upload images and videos, and load images from any S3-compatible storage service (AWS S3, Supabase, MinIO, etc.) directly from your ComfyUI workflows without relying on environment variables. All credentials are provided inline for maximum flexibility and control.

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started)
2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
3. Look up "S3 Serverless-Inline Storage Utils" in ComfyUI-Manager, or manually clone this repository under `ComfyUI/custom_nodes`
4. Restart ComfyUI
5. Find the nodes in the **"S3 Serverless Storage"** category

## Available Nodes

### 🖼️ S3 Image Upload (Inline Credentials)
Upload images to S3 with inline credentials. Automatically converts all images to WebP format for optimal storage efficiency.

**Inputs:**
- `images` - ComfyUI IMAGE tensor(s) to upload
- `filename_prefix` - Base filename (e.g., "test123" → "test123.webp")
- `bucket_name` - S3 bucket name
- `access_key_id` - AWS Access Key ID
- `secret_access_key` - AWS Secret Access Key  
- `region` - AWS region (default: "us-east-1")
- `endpoint_url` (optional) - Custom S3 endpoint for services like Supabase
- `folder_path` (optional) - Folder within bucket (default: "uploads")

**Outputs:**
- `s3_urls` - List of uploaded image URLs
- `upload_status` - Status message
- `uploaded_count` - Number of successfully uploaded images

### 🎥 S3 Video Upload (Inline Credentials)
Upload video files to S3 with inline credentials. Preserves original video format and quality.

**Inputs:**
- `video` - Local path to video file
- `filename_prefix` - Base filename (e.g., "myvideo" → "myvideo.mp4")
- `bucket_name` - S3 bucket name
- `access_key_id` - AWS Access Key ID
- `secret_access_key` - AWS Secret Access Key
- `region` - AWS region (default: "us-east-1")
- `endpoint_url` (optional) - Custom S3 endpoint
- `folder_path` (optional) - Folder within bucket (default: "uploads")

**Outputs:**
- `s3_url` - Uploaded video URL
- `upload_status` - Status message
- `file_size_mb` - Video file size in MB

### 📥 S3 Image Load from URL
Load images from S3 URLs or signed URLs directly into your workflow.

**Inputs:**
- `url` - S3 URL or signed URL to image
- `timeout` (optional) - Request timeout in seconds (default: 30)

**Outputs:**
- `image` - ComfyUI IMAGE tensor
- `filename` - Original filename
- `status` - Load status message

## Usage Examples

### Basic Image Upload to AWS S3
```
filename_prefix: "my_artwork"
bucket_name: "my-bucket"
access_key_id: "AKIA..."
secret_access_key: "wJalrX..."
region: "us-west-2"
```
**Result:** `my_artwork.webp` uploaded to S3

### Upload to Supabase Storage
```
filename_prefix: "generated_image"
bucket_name: "images"
access_key_id: "your_supabase_s3_key"
secret_access_key: "your_supabase_s3_secret"
region: "us-east-1"
endpoint_url: "https://your-project.supabase.co/storage/v1/s3"
```

### Load Image from S3 URL
```
url: "https://my-bucket.s3.amazonaws.com/uploads/image.webp"
```
or
```
url: "https://presigned-url.s3.amazonaws.com/image.jpg?expires=..."
```

## S3-Compatible Services

This extension works with any S3-compatible storage service:

- **AWS S3** - Leave `endpoint_url` empty
- **Supabase Storage** - Set `endpoint_url` to your Supabase S3 endpoint
- **MinIO** - Set `endpoint_url` to your MinIO server
- **DigitalOcean Spaces** - Set appropriate endpoint
- **Wasabi** - Set appropriate endpoint
- **Any S3-compatible service**

## Security Best Practices

### ⚠️ Credential Security Warning

**NEVER expose workflows containing credentials publicly!**

When sharing workflows that use these nodes:

1. **Remove all credential values** from the workflow JSON before sharing
2. **Use environment-specific credentials** - don't hardcode production keys
3. **Consider using temporary/limited-scope access keys** for workflows
4. **Implement proper bucket policies** to restrict access as needed
5. **Use signed URLs** for temporary access instead of long-lived credentials when possible

### Recommended Approach for Sharing Workflows

Instead of sharing workflows with embedded credentials, provide templates like:

```json
{
  "filename_prefix": "YOUR_FILENAME_HERE",
  "bucket_name": "YOUR_BUCKET_NAME", 
  "access_key_id": "YOUR_ACCESS_KEY_ID",
  "secret_access_key": "YOUR_SECRET_ACCESS_KEY",
  "region": "YOUR_REGION",
  "endpoint_url": "YOUR_ENDPOINT_IF_NEEDED"
}
```

### Bucket Configuration Tips

- **Set up proper CORS policies** if accessing uploaded files from web browsers
- **Configure bucket policies** to limit access to specific operations
- **Enable versioning** if you need to track file changes
- **Set up lifecycle policies** to manage storage costs

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd s3_serverless_inline_storage_utils
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name. 
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
``` 

## Writing custom nodes

An example custom node is located in [node.py](src/s3_serverless_inline_storage_utils/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile). 
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

