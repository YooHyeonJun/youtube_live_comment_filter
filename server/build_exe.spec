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
from PyInstaller.utils.hooks import copy_metadata

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
    except:
        pass

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=model_datas + metadata_datas + [
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='youtube_live_filter_server',
)

