import argparse
import torch
from diffusers import TextToVideoSDPipeline  # <-- CHANGED
import imageio
from pathlib import Path
from google.cloud import storage

# --------- Parse Prompt ----------
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()
prompt = args.prompt

print(f"▶ Generating video for prompt:\n{prompt}")

# --------- Load Model ----------
# Using a Text-to-Video model instead
pipe = TextToVideoSDPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",  # <-- CHANGED
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_model_cpu_offload()

# --------- Generate Video ----------
# The call remains the same, but now it's using the correct type of model
video_frames = pipe(prompt=prompt, num_frames=25, num_inference_steps=25).frames

# --------- Save to MP4 ----------
output_path = Path("output.mp4")
imageio.mimsave(output_path, video_frames, fps=8)
print(f"✅ Saved video to: {output_path.resolve()}")

# --------- Upload to GCS ----------
bucket_name = "n8n-video-output-bucket"
# Create a unique name for the blob
dest_blob_name = f"video_{Path(output_path).stem}_{Path(output_path).stat().st_mtime}.mp4"

client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(dest_blob_name)
blob.upload_from_filename(str(output_path))

print(f"✅ Uploaded to GCS: gs://{bucket_name}/{dest_blob_name}")
