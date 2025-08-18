# S3 Serverless-Inline Storage Utils

Upload images and videos to S3 directly from ComfyUI workflows. No environment variables needed - all credentials go directly in the nodes.

## Installation

1. Install via ComfyUI-Manager or clone to `ComfyUI/custom_nodes`
2. Restart ComfyUI
3. Look for nodes in "S3 Serverless Storage" category

## Nodes

**S3 Image Upload** - Upload images as WebP files
- Put your S3 credentials directly in the node
- Images automatically convert to WebP format
- Clean filenames: `test123` becomes `test123.webp`

**S3 Video Upload** - Upload video files  
- Same inline credentials approach
- Keeps original video format
- Clean filenames: `myvideo` with `clip.mp4` becomes `myvideo.mp4`

**S3 Image Load** - Download images from S3 URLs
- Works with public URLs and signed URLs
- No credentials needed for public files

## Usage

### For AWS S3:
- bucket_name: your bucket name
- access_key_id: your access key  
- secret_access_key: your secret key
- region: your AWS region
- endpoint_url: leave empty

### For Supabase:
- bucket_name: your bucket name
- access_key_id: your supabase S3 key
- secret_access_key: your supabase S3 secret  
- region: us-east-1
- endpoint_url: https://yourproject.supabase.co/storage/v1/s3

### For other S3-compatible services:
Fill in your service's S3 endpoint URL.

## Security Note

Don't share workflows with your credentials filled in. Clear the credential fields before sharing workflow files with others.

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

