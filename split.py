from pathlib import Path
src = Path('public/model_int8.onnx_data')   # adjust if your file lives elsewhere
if not src.exists():
    raise SystemExit(f"Source not found: {src}")
out_dir = Path('public')
out_dir.mkdir(exist_ok=True)
chunk_size = 64 * 1024 * 1024  # 64 MB
i = 1
with src.open('rb') as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        name = f'model_int8.onnx_data.part{str(i).zfill(2)}'
        out = out_dir / name
        out.write_bytes(chunk)
        print('wrote', out)
        i += 1
print('done, parts written:', i-1)