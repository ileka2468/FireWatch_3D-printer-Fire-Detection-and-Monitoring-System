# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for FireWatch - 3D Print Fire Detection System.

Build with:  pyinstaller firewatch.spec
Output:      dist/FireWatch/FireWatch.exe
"""

import os

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include test images
        ('test_images', 'test_images'),
        # Include the window icon
        ('icon.ico', '.'),
        # Include web dashboard assets
        ('web/templates', 'web/templates'),
        ('web/static', 'web/static'),
    ],
    hiddenimports=[
        'customtkinter',
        'PIL',
        'PIL._tkinter_finder',
        'cv2',
        'ollama',
        'httpx',
        'pydantic',
        'flask',
        'jinja2',
        'werkzeug',
        'blinker',
        'itsdangerous',
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
    name='FireWatch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window — GUI app
    icon='icon.ico',  # Set the app icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FireWatch',
)
