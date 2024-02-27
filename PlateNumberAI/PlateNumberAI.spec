# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['PlateNumberAI.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/pashn/source/repos/PlateNumberAI/PlateNumberAI/plate_number/Lib/site-packages/ultralytics/cfg/default.yaml', 'ultralytics/cfg')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PlateNumberAI',
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PlateNumberAI',
)
