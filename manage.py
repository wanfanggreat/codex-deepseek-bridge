#!/usr/bin/env python3
"""Inspect or uninstall a bridge installed by install.py."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tomllib
from install import atomic_write


def restored_config(content, state, backup):
    if hashlib.sha256(content.encode()).hexdigest() == state['installed_config_sha256']:
        return backup
    for key in ('managed_root_block','managed_tables_block'):
        block = state[key]
        if content.count(block) != 1:
            raise ValueError('Managed blocks were edited; automatic uninstall stopped without changes')
        content = content.replace(block,'',1)
    content = state['removed_root_lines']+content+'\n'+state['original_agents_section']
    tomllib.loads(content)
    return content.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','uninstall','rollback'])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--install-dir',type=pathlib.Path,
                        default=pathlib.Path(os.environ.get('CODEX_HOME',pathlib.Path.home()/'.codex'))/'deepseek-bridge')
    args = parser.parse_args()
    root = args.install_dir.expanduser().resolve()
    state = json.loads((root/'install-state.json').read_text())
    config = root.parent/'config.toml'
    if state['platform']=='darwin':
        service_command = ['/bin/launchctl','print','gui/'+str(os.getuid())+'/'+state['label']]
    else:
        service_command = ['systemctl','--user','is-active',state['unit']+'.socket']
    if args.action=='status':
        service = subprocess.run(service_command,capture_output=True)
        catalog = json.loads((root/'models.json').read_text())
        model = next(m for m in catalog['models'] if m['slug']=='deepseek-v4.1-flash')
        print(json.dumps({'configured':tomllib.loads(config.read_text()).get('model_provider')=='local_deepseek_bridge',
                          'socket_registered':service.returncode==0,'context_tokens':model['context_window'],
                          'auto_compact_threshold':model['auto_compact_token_limit'],
                          'reasoning':model['default_reasoning_level']}))
        return
    backup = (pathlib.Path(state['backup'])/'config.toml').read_bytes()
    restored = restored_config(config.read_text(),state,backup)
    if args.dry_run:
        print(json.dumps({'uninstall_valid':True,'matches_original':hashlib.sha256(restored).hexdigest()==state['original_config_sha256']}))
        return
    if state['platform']=='darwin':
        subprocess.run(['/bin/launchctl','bootout','gui/'+str(os.getuid())+'/'+state['label']],capture_output=True)
    else:
        subprocess.run(['systemctl','--user','disable','--now',state['unit']+'.socket'],capture_output=True)
        subprocess.run(['systemctl','--user','stop',state['unit']+'.service'],capture_output=True)
    for path in map(pathlib.Path,state['services']):
        path.unlink(missing_ok=True)
    if state['platform']=='linux':
        subprocess.run(['systemctl','--user','daemon-reload'],capture_output=True)
    if state['original_config_existed'] or restored.strip():
        atomic_write(config,restored,state['original_config_mode'])
    else:
        config.unlink()
    settings = json.loads((root/'settings.json').read_text())
    if state['keychain_entry_created']:
        subprocess.run(['/usr/bin/security','delete-generic-password','-a',settings['keychain_account'],
                        '-s',settings['keychain_service']],capture_output=True)
    for name in ('relay.key','gateway.token'):
        (root/name).unlink(missing_ok=True)
    state['uninstalled'] = True
    atomic_write(root/'install-state.json',(json.dumps(state,indent=2)+'\n').encode())
    print('Bridge uninstalled. Backups and source retained. Restart Codex and create a new chat.')


if __name__=='__main__':
    main()
