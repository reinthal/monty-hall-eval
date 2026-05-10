"""Template Modal app — runs on cloud GPU."""

import modal

app = modal.App("my-project")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", "numpy", "tqdm")
)


@app.function(image=image, gpu="A100")
def gpu_task():
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1000, 1000, device="cuda")
        result = x @ x.T
        print(f"Matmul result shape: {result.shape}")
    return "done"


@app.local_entrypoint()
def main():
    result = gpu_task.remote()
    print(f"Result: {result}")
