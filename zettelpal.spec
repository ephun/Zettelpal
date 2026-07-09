# -*- mode: python ; coding: utf-8 -*-
# Zettelpal PyInstaller spec file
# Build with: pyinstaller zettelpal.spec

import os

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
spec_dir = os.path.dirname(os.path.abspath(SPEC))

# CustomTkinter ships its theme/asset JSON files as package data that must be
# bundled or the GUI fails to start in the frozen build.
ctk_datas = collect_data_files('customtkinter')

a = Analysis(
    ['launcher.py'],
    pathex=[spec_dir],
    binaries=[],
    datas=[
        ('zettelpal/zettelpal.ico', 'zettelpal'),
    ] + ctk_datas,
    hiddenimports=[
        'whisper',
        'sentence_transformers',
        'torch',
        'sklearn',
        'sklearn.metrics',
        'sklearn.metrics.pairwise',
        'numpy',
        'requests',
        'typer',
        'pydantic',
        'pydantic_settings',
        'customtkinter',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.scrolledtext',
        'tkinter.messagebox',
        # Whisper dependencies
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        # Torch dependencies
        'torch._C',
        'torch.utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Zettelpal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='zettelpal/zettelpal.ico',
)
