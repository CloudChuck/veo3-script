import argparse
import torch
from diffusers import StableVideoDiffusionPipeline
import imageio
from pathlib import Path
from PIL import Image
from google.cloud import storage

# --------- Parse Prompt ----------
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()
prompt = args.prompt

print(f"▶ Generating video for prompt:\n{prompt}")

# --------- Load Model ----------
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_model_cpu_offload()

# --------- Generate Video ----------
video_frames = pipe(prompt=prompt, num_frames=25).frames[0]

# --------- Save to MP4 ----------
output_path = Path("output.mp4")
imageio.mimsave(output_path, video_frames, fps=8)
print(f"✅ Saved video to: {output_path.resolve()}")

# --------- Upload to GCS ----------
bucket_name = "n8n-video-output-bucket"
dest_blob_name = f"video_{Path(output_path).stem}.mp4"

client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(dest_blob_name)
blob.upload_from_filename(str(output_path))

print(f"✅ Uploaded to GCS: gs://{bucket_name}/{dest_blob_name}")
