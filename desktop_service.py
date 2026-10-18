#!/usr/bin/env python3
"""Socket-activated gateway; runtime settings and credentials stay outside Git."""
import argparse
import json
import os
import pathlib
import stat
import subprocess
import bridge


def load_key(root, settings):
    if settings['platform'] == 'darwin':
        result = subprocess.run(
            ['/usr/bin/security', 'find-generic-password', '-a', settings['keychain_account'],
             '-s', settings['keychain_service'], '-w'], capture_output=True, timeout=20)
        if result.returncode or not result.stdout.strip():
            raise RuntimeError('Relay Keychain entry is unavailable')
        return result.stdout.decode().strip()
    path = root / 'relay.key'
    if path.stat().st_uid != os.getuid() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RuntimeError('relay.key must belong to this user and have mode 0600')
    key = path.read_text().strip()
    if not key:
        raise RuntimeError('relay.key is empty')
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-dir', type=pathlib.Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.install_dir.resolve()
    settings = json.loads((root / 'settings.json').read_text())
    os.environ['DEEPSEEK_API_KEY'] = load_key(root, settings)
    os.environ['CODEX_BRIDGE_TOKEN'] = (root / 'gateway.token').read_text().strip()
    bridge.serve(argparse.Namespace(
        relay=settings['relay'], model='deepseek-v4.1-flash', catalog=str(root/'models.json'),
        port=settings['port'], launchd_socket='Listener' if settings['platform']=='darwin' else None,
        systemd_socket=settings['platform']=='linux', follow_pid=None,
        audit=str(root/'logs/gateway.jsonl'), max_output=8192, idle_timeout=90,
        context_store=str(root/'contexts')))


if __name__ == '__main__':
    main()
