# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
from pathlib import Path

# 모델 파일을 포함
model_datas = []

spec_root = os.path.abspath(SPECPATH)
model_dir = os.path.abspath(os.path.join(spec_root, '..', 'model'))

print(f"Looking for model in: {model_dir}")

if os.path.exists(model_dir):
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.startswith('.'):
                continue

            src_path = os.path.join(root, file)
            rel_to_model = os.path.relpath(src_path, model_dir)
            dest_path = os.path.join('model', rel_to_model)
            dest_dir = os.path.dirname(dest_path)

            model_datas.append((src_path, dest_dir))
            print(f"Adding model file: {file} -> {dest_dir}")
else:
    print(f"WARNING: Model directory not found at {model_dir}")

print(f"Total model files to include: {len(model_datas)}")

from PyInstaller.utils.hooks import copy_metadata, collect_all

metadata_packages = [
    'transformers',
    'tokenizers',
    'tqdm',
    'regex',
    'filelock',
    'numpy',
    'huggingface-hub',
    'safetensors',
    'pyyaml',
    'requests',
    'packaging',
    'torch',
    'fastapi',
    'uvicorn',
    'pydantic',
    'sklearn',
    'scikit-learn',
]

metadata_datas = []
for pkg in metadata_packages:
    try:
        metadata_datas += copy_metadata(pkg)
    except Exception:
        pass

# torch 전체 수집 (기본 hook 동작에 최대한 맡김)
try:
    torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
    print(f"PyTorch data files: {len(torch_datas)}")
    print(f"PyTorch binaries: {len(torch_binaries)}")
    print(f"PyTorch hidden imports: {len(torch_hiddenimports)}")
except Exception as e:
    print(f"WARNING: PyTorch collect_all 실패: {e}")
    torch_datas = []
    torch_binaries = []
    torch_hiddenimports = []

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=torch_binaries,
    datas=model_datas + metadata_datas + torch_datas + [
        ('train.py', '.'),
    ],
    hiddenimports=torch_hiddenimports + [
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'sklearn.utils._weight_vector',
        'sklearn.neighbors._partition_nodes',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# VC++ 런타임 DLL은 시스템 vc_redist를 사용하도록 번들에서 제거
vc_runtime_names = {
    'vcruntime140.dll',
    'vcruntime140_1.dll',
    'msvcp140.dll',
    'vcomp140.dll',
    'api-ms-win-crt-runtime-l1-1-0.dll',
    'api-ms-win-crt-stdio-l1-1-0.dll',
    'api-ms-win-crt-heap-l1-1-0.dll',
    'api-ms-win-crt-convert-l1-1-0.dll',
    'api-ms-win-crt-locale-l1-1-0.dll',
    'api-ms-win-crt-time-l1-1-0.dll',
    'api-ms-win-crt-filesystem-l1-1-0.dll',
    'api-ms-win-crt-math-l1-1-0.dll',
    'api-ms-win-crt-string-l1-1-0.dll',
    'api-ms-win-crt-utility-l1-1-0.dll',
    'ucrtbase.dll',
}

def _is_vc_runtime(binary_tuple):
    src = os.path.basename(binary_tuple[0]).lower()
    return src in vc_runtime_names

filtered_binaries = [b for b in a.binaries if not _is_vc_runtime(b)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='youtube_live_filter_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    filtered_binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='youtube_live_filter_server',
)

