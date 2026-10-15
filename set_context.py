#!/usr/bin/env python3
"""Switch only DeepSeek's catalog budget between the test and normal windows."""
import argparse
import datetime
import json
import os
import pathlib
import tempfile
import tomllib


def atomic_write(path, data):
    mode = path.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.context-', delete=False) as f:
        f.write(data)
        temp = pathlib.Path(f.name)
    temp.chmod(mode)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['small', 'normal', 'status'])
    args = parser.parse_args()
    codex_home = pathlib.Path(os.environ.get('CODEX_HOME', pathlib.Path.home() / '.codex')).expanduser().resolve()
    root = codex_home / 'deepseek-bridge'
    path = root / 'models.json'
    config = tomllib.loads((codex_home / 'config.toml').read_text())
    if pathlib.Path(config.get('model_catalog_json', '')).resolve() != path.resolve():
        raise SystemExit('The installed bridge is not the active model catalog; no files changed.')
    original = path.read_bytes()
    catalog = json.loads(original)
    model = next(m for m in catalog['models'] if m['slug'] == 'deepseek-v4.1-flash')
    if args.mode != 'status':
        window, threshold = (32000, 16000) if args.mode == 'small' else (500000, 400000)
        model.update(context_window=window, max_context_window=window,
                     effective_context_window_percent=100, auto_compact_token_limit=threshold)
        encoded = (json.dumps(catalog, ensure_ascii=False, indent=2) + '\n').encode()
        if encoded != original:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup = root / 'backups' / ('context-' + stamp)
            backup.mkdir(parents=True, mode=0o700)
            (backup / 'models.json').write_bytes(original)
            atomic_write(path, encoded)
        state_path = root / 'install-state.json'
        state = json.loads(state_path.read_text())
        state.update(deepseek_effective_context_tokens=window,
                     deepseek_auto_compact_threshold_tokens=threshold,
                     deepseek_context_mode=args.mode)
        atomic_write(state_path, (json.dumps(state, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({
        'model': model['slug'],
        'effective_context_tokens': model['context_window'] * model['effective_context_window_percent'] // 100,
        'auto_compact_threshold': model['auto_compact_token_limit'],
        'default_reasoning': model['default_reasoning_level'],
        'desktop_reload_required': args.mode != 'status',
    }))


if __name__ == '__main__':
    main()
