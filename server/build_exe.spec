# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 모델 파일들을 포함
model_datas = []
import os
from pathlib import Path

# 현재 spec 파일의 절대 경로 기준으로 모델 디렉토리 찾기
spec_root = os.path.abspath(SPECPATH)
model_dir = os.path.join(spec_root, '..', 'model')
model_dir = os.path.abspath(model_dir)

print(f"Looking for model in: {model_dir}")

if os.path.exists(model_dir):
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            # .gitkeep 같은 불필요한 파일 제외
            if file.startswith('.'):
                continue
            
            src_path = os.path.join(root, file)
            # 목적지는 항상 'model' 폴더 안에
            rel_to_model = os.path.relpath(src_path, model_dir)
            dest_path = os.path.join('model', rel_to_model)
            dest_dir = os.path.dirname(dest_path)
            
            model_datas.append((src_path, dest_dir))
            print(f"Adding model file: {file} -> {dest_dir}")
else:
    print(f"WARNING: Model directory not found at {model_dir}")
    
print(f"Total model files to include: {len(model_datas)}")

# Collect metadata for packages that need it
from PyInstaller.utils.hooks import copy_metadata, collect_submodules

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
    'accelerate',
    'peft',
]

metadata_datas = []
for pkg in metadata_packages:
    try:
        metadata_datas += copy_metadata(pkg)
    except:
        pass

# Collect ALL torch submodules (including _dynamo)
torch_hidden = []
dynamo_hidden = []
dynamo_polyfill_hidden = []
numpy_hidden = []
try:
    torch_hidden = collect_submodules('torch')
    print(f"Collected {len(torch_hidden)} torch submodules")
    # Ensure _dynamo and its polyfills are explicitly collected (PyInstaller sometimes misses nested namespace packages)
    dynamo_hidden = collect_submodules('torch._dynamo')
    print(f"Collected {len(dynamo_hidden)} torch._dynamo submodules")
    dynamo_polyfill_hidden = collect_submodules('torch._dynamo.polyfills')
    print(f"Collected {len(dynamo_polyfill_hidden)} torch._dynamo.polyfills submodules")
    # Numpy submodules (ensure numpy C-extension packages are included)
    numpy_hidden = collect_submodules('numpy')
    print(f"Collected {len(numpy_hidden)} numpy submodules")
except Exception as e:
    print(f"Warning: Could not collect torch submodules: {e}")

import sys
import site

# torch._dynamo.polyfills 디렉터리를 포함
torch_dynamo_datas = []
try:
    import torch._dynamo.polyfills
    polyfills_dir = os.path.dirname(torch._dynamo.polyfills.__file__)
    for root, dirs, files in os.walk(polyfills_dir):
        for file in files:
            if file.endswith('.py'):
                src = os.path.join(root, file)
                rel = os.path.relpath(root, os.path.dirname(polyfills_dir))
                torch_dynamo_datas.append((src, os.path.join('torch', '_dynamo', rel)))
    print(f"Added {len(torch_dynamo_datas)} torch._dynamo.polyfills files")
except Exception as e:
    print(f"Warning: Could not add torch._dynamo.polyfills: {e}")

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=model_datas + metadata_datas + torch_dynamo_datas + [
        ('train.py', '.'),
    ],
    hiddenimports=[
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
        # torch dynamo/polyfills explicit (PyInstaller may miss these)
        'torch._dynamo',
        'torch._dynamo.polyfills',
        'torch._dynamo.polyfills.loader',
        'torch._dynamo.polyfills.fx',
        'torch._dynamo.polyfills.indexing',
        'torch._dynamo.polyfills.modules',
        'torch._dynamo.polyfills.utils',
        'torch._dynamo.polyfills.tensor',
        'sklearn.utils._weight_vector',
        'sklearn.neighbors._partition_nodes',
        # accelerate & peft (for training)
        'accelerate',
        'accelerate.utils',
        'peft',
        'peft.utils',
        # transformers training
        'transformers.trainer',
        'transformers.trainer_callback',
        'transformers.training_args',
        'transformers.integrations',
    ] + torch_hidden + dynamo_hidden + dynamo_polyfill_hidden + numpy_hidden,  # Add ALL torch + _dynamo + polyfills + numpy
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='youtube_live_filter_server',
    debug=True,
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
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='youtube_live_filter_server',
)

