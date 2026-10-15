#!/usr/bin/env python3
"""Install the native Codex bridge for macOS or Linux without modifying Codex."""
import argparse
import datetime
import getpass
import hashlib
import json
import os
import pathlib
import plistlib
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request
from launch import build_catalog

LABEL = 'com.codex.deepseek-bridge'
UNIT = 'codex-deepseek-bridge'
ROOT_START = '# BEGIN CODEX DEEPSEEK BRIDGE ROOT\n'
ROOT_END = '# END CODEX DEEPSEEK BRIDGE ROOT\n'
TABLE_START = '# BEGIN CODEX DEEPSEEK BRIDGE TABLES\n'
TABLE_END = '# END CODEX DEEPSEEK BRIDGE TABLES\n'
SOURCE = pathlib.Path(__file__).resolve().parent


def atomic_write(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.bridge-', delete=False) as f:
        f.write(data)
        temp = pathlib.Path(f.name)
    temp.chmod(mode)
    os.replace(temp, path)


def remove_keys(text, keys):
    kept, removed = [], []
    for line in text.splitlines(keepends=True):
        match = re.match(r'^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=', line)
        if match and match[1] in keys:
            # Managed values are scalar strings/bools/integers; reject unusual
            # multiline forms before mutating any credentials or services.
            parsed = tomllib.loads(line)
            if set(parsed) != {match[1]}:
                raise ValueError('Unsupported managed configuration line')
            removed.append(line)
        else:
            kept.append(line)
    return ''.join(kept), ''.join(removed)


def make_config(original, root, token, port):
    parsed = tomllib.loads(original)
    if parsed.get('model_provider', 'openai') != 'openai' or parsed.get('model_catalog_json'):
        raise ValueError('An existing provider/catalog is active; use launch.py for an isolated run instead.')
    if ROOT_START in original or TABLE_START in original:
        raise ValueError('A bridge configuration already exists')
    for name in ('local_deepseek_bridge', 'local_deepseek_chat'):
        if name in parsed.get('model_providers', {}):
            raise ValueError('Reserved provider already exists: '+name)
    if 'deepseek' in parsed.get('agents', {}):
        raise ValueError('An agents.deepseek role already exists')
    headers = list(re.finditer(r'(?m)^\s*(\[[^\n]+\])[^\n]*\n?', original))
    first = headers[0].start() if headers else len(original)
    prefix, removed_root = remove_keys(original[:first], {'model_provider'})
    remaining = original[first:]
    agent_section = ''
    for index, header in enumerate(headers):
        if header[1].strip() == '[agents]':
            end = headers[index+1].start() if index+1 < len(headers) else len(original)
            agent_section = original[header.start():end]
            remaining = original[first:header.start()] + original[end:]
            break
    body = agent_section.split('\n', 1)[1] if agent_section else ''
    body, _ = remove_keys(body, {'enabled', 'default_subagent_model',
                                  'default_subagent_reasoning_effort', 'max_concurrent_threads_per_session'})
    root_block = (ROOT_START + 'model_provider = "local_deepseek_bridge"\n'
                  + 'model_catalog_json = '+json.dumps(str(root/'models.json'))+'\n'+ROOT_END)
    providers = ''
    for name in ('local_deepseek_bridge', 'local_deepseek_chat'):
        providers += (f'[model_providers.{name}]\nname = "Local Codex + DeepSeek bridge"\n'
                      f'base_url = "http://127.0.0.1:{port}"\nwire_api = "responses"\n'
                      'requires_openai_auth = true\nsupports_websockets = false\n'
                      'request_max_retries = 1\nstream_max_retries = 0\n'
                      f'http_headers = {{ "X-Bridge-Key" = "{token}" }}\n')
        if name == 'local_deepseek_chat':
            providers += f'[model_providers.{name}.capabilities]\nremote_compaction = "unsupported"\n'
    maximum = max(2, parsed.get('agents', {}).get('max_concurrent_threads_per_session', 2))
    tables = (TABLE_START+providers+'[agents]\n'+body
              +'\nenabled = true\nmax_concurrent_threads_per_session = '+str(maximum)
              +'\ndefault_subagent_model = "deepseek-v4.1-flash"\n'
              +'default_subagent_reasoning_effort = "high"\n'
              +'[agents.deepseek]\ndescription = "DeepSeek Chat relay worker in the Codex harness."\n'
              +'config_file = '+json.dumps(str(root/'deepseek-agent.toml'))+'\n'+TABLE_END)
    result = root_block+prefix+remaining+'\n'+tables
    tomllib.loads(result)
    return result, {'managed_root_block':root_block, 'managed_tables_block':tables,
                    'removed_root_lines':removed_root, 'original_agents_section':agent_section}


def render_services(platform, root, python, port, home):
    if platform == 'darwin':
        path = home/'Library/LaunchAgents'/(LABEL+'.plist')
        data = {'Label':LABEL,'ProgramArguments':[python,str(root/'desktop_service.py'),'--install-dir',str(root)],
                'WorkingDirectory':str(root),'ProcessType':'Background','ThrottleInterval':5,
                'EnvironmentVariables':{'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'},
                'StandardOutPath':str(root/'logs/service.stdout'),'StandardErrorPath':str(root/'logs/service.stderr'),
                'Sockets':{'Listener':{'SockFamily':'IPv4','SockNodeName':'127.0.0.1',
                                      'SockServiceName':str(port),'SockType':'stream'}}}
        return {path:plistlib.dumps(data)}
    if any(c in str(root)+python for c in '\n\r%'):
        raise ValueError('systemd installation paths cannot contain newlines or percent signs')
    def quote(value):
        return '"'+str(value).replace('\\','\\\\').replace('"','\\"')+'"'
    folder = home/'.config/systemd/user'
    service = ('[Unit]\nDescription=Codex DeepSeek local protocol bridge\n'
               '[Service]\nType=simple\nExecStart='+quote(python)+' '+quote(root/'desktop_service.py')
               +' --install-dir '+quote(root)+'\nWorkingDirectory='+quote(root)+'\n'
               'Environment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=PYTHONUNBUFFERED=1\n'
               'UMask=0077\nNoNewPrivileges=true\n')
    activation = ('[Unit]\nDescription=Codex DeepSeek IPv4 loopback socket\n'
                  f'[Socket]\nListenStream=127.0.0.1:{port}\nAccept=no\n'
                  f'Service={UNIT}.service\n[Install]\nWantedBy=sockets.target\n')
    return {folder/(UNIT+'.service'):service.encode(), folder/(UNIT+'.socket'):activation.encode()}


def validate_relay(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Relay must be an HTTPS base URL without credentials, query or fragment')
    return url.rstrip('/')


def find_codex(explicit):
    if explicit:
        found = shutil.which(explicit)
        if not found:
            raise ValueError('Codex executable was not found')
        return found
    found = shutil.which('codex')
    bundled = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex'
    if not found and sys.platform=='darwin' and pathlib.Path(bundled).is_file():
        found = bundled
    if not found:
        raise ValueError('Install Codex and log in first, or supply --codex /absolute/path')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relay', required=True, help='HTTPS base URL including /v1 when needed')
    parser.add_argument('--codex', help='Path to the Codex executable')
    parser.add_argument('--port', type=int, default=24589)
    parser.add_argument('--dry-run', action='store_true', help='Validate the plan without credentials or writes')
    args = parser.parse_args()
    if sys.platform not in ('darwin','linux'):
        raise SystemExit('Supported platforms: macOS and Linux')
    os.umask(0o077)
    relay = validate_relay(args.relay)
    if not 1024<=args.port<=65535:
        raise SystemExit('Choose a port between 1024 and 65535')
    home = pathlib.Path.home()
    codex_home = pathlib.Path(os.environ.get('CODEX_HOME',home/'.codex')).expanduser().resolve()
    config = codex_home/'config.toml'
    root = codex_home/'deepseek-bridge'
    if root.exists():
        raise SystemExit('Install directory already exists. No changes made; use its manage.py status or uninstall first.')
    codex = find_codex(args.codex)
    original = config.read_bytes() if config.exists() else b''
    token = secrets.token_urlsafe(32)
    content, blocks = make_config(original.decode(),root,token,args.port)
    catalog = build_catalog(codex)
    services = render_services(sys.platform,root,sys.executable,args.port,home)
    if any(p.exists() for p in services):
        raise SystemExit('A target service file already exists; no changes made')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',args.port))
    with tempfile.TemporaryDirectory(prefix='codex-version-discovery-') as discovery:
        version = subprocess.check_output([codex,'--version'],text=True,
                                          env=dict(os.environ,CODEX_HOME=discovery)).strip()
    if sys.platform=='linux':
        result = subprocess.run(['systemctl','--user','show-environment'],capture_output=True)
        if result.returncode:
            raise SystemExit('systemd user manager is unavailable. Use launch.py instead.')
    if args.dry_run:
        print(json.dumps({'dry_run':True,'platform':sys.platform,'codex':version,
                          'install_dir':str(root),'services':[str(p) for p in services],
                          'context_tokens':500000,'reasoning':'high'}))
        return
    key = os.environ.pop('DEEPSEEK_API_KEY','') or getpass.getpass('Relay API key (hidden, never put it in command arguments): ')
    if not key or any(c in key for c in '\r\n'):
        raise SystemExit('A nonempty single-line API key is required')
    settings = {'platform':sys.platform,'relay':relay,'port':args.port,
                'keychain_service':'com.codex.deepseek.relay','keychain_account':'codex-deepseek'}
    state = dict(blocks, original_config_existed=config.exists(),
                 original_config_mode=(config.stat().st_mode & 0o777) if config.exists() else 0o600,
                 original_config_sha256=hashlib.sha256(original).hexdigest(),
                 installed_config_sha256=hashlib.sha256(content.encode()).hexdigest(),
                 platform=sys.platform,label=LABEL,unit=UNIT,services=[str(p) for p in services],
                 keychain_entry_created=False,version=version)
    key_created = False
    config_written = False
    try:
        root.mkdir(parents=True,mode=0o700)
        (root/'logs').mkdir(mode=0o700)
        backup = root/'backups'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup.mkdir(parents=True,mode=0o700)
        atomic_write(backup/'config.toml',original)
        state['backup'] = str(backup)
        for name in ('bridge.py','desktop_service.py','manage.py','set_context.py','install.py','launch.py'):
            atomic_write(root/name,(SOURCE/name).read_bytes(),0o700)
        atomic_write(root/'models.json',(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n').encode())
        atomic_write(root/'settings.json',(json.dumps(settings,indent=2)+'\n').encode())
        atomic_write(root/'gateway.token',(token+'\n').encode())
        atomic_write(root/'deepseek-agent.toml',b'model = "deepseek-v4.1-flash"\nmodel_provider = "local_deepseek_chat"\nweb_search = "disabled"\n[features]\napps = false\nplugins = false\nmemories = false\n')
        if sys.platform=='darwin':
            existing = subprocess.run(['/usr/bin/security','find-generic-password','-a',settings['keychain_account'],
                                       '-s',settings['keychain_service'],'-w'],capture_output=True)
            if existing.returncode==0:
                if existing.stdout.decode().strip()!=key:
                    raise RuntimeError('A different Keychain credential already exists; it was preserved')
            else:
                # security's interactive input keeps the secret out of argv.
                command = 'add-generic-password -a '+shlex.quote(settings['keychain_account'])+' -s '+shlex.quote(settings['keychain_service'])+' -w '+shlex.quote(key)+'\n'
                saved = subprocess.run(['/usr/bin/security','-i'],input=command.encode(),capture_output=True)
                check = subprocess.run(['/usr/bin/security','find-generic-password','-a',settings['keychain_account'],
                                        '-s',settings['keychain_service'],'-w'],capture_output=True)
                key_created = saved.returncode == 0
                if saved.returncode or check.returncode or check.stdout.decode().strip()!=key:
                    raise RuntimeError('Could not verify the new Keychain entry')
                key_created = state['keychain_entry_created'] = True
        else:
            atomic_write(root/'relay.key',(key+'\n').encode())
        del key
        atomic_write(root/'install-state.json',(json.dumps(state,indent=2)+'\n').encode())
        # The added local capability must not become readable to other users.
        atomic_write(config,content.encode(),0o600)
        config_written = True
        for path,data in services.items():
            atomic_write(path,data)
        if sys.platform=='darwin':
            subprocess.run(['/bin/launchctl','bootstrap','gui/'+str(os.getuid()),str(next(iter(services)))],check=True,capture_output=True)
        else:
            subprocess.run(['systemctl','--user','daemon-reload'],check=True,capture_output=True)
            subprocess.run(['systemctl','--user','enable','--now',UNIT+'.socket'],check=True,capture_output=True)
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=20) as response:
            if json.load(response).get('status')!='ok':
                raise RuntimeError('Gateway health check failed')
    except Exception as exc:
        if sys.platform=='darwin':
            subprocess.run(['/bin/launchctl','bootout','gui/'+str(os.getuid())+'/'+LABEL],capture_output=True)
        else:
            subprocess.run(['systemctl','--user','disable','--now',UNIT+'.socket'],capture_output=True)
            subprocess.run(['systemctl','--user','stop',UNIT+'.service'],capture_output=True)
        for path in services:
            if path.exists():
                path.unlink()
        if sys.platform=='linux':
            subprocess.run(['systemctl','--user','daemon-reload'],capture_output=True)
        if config_written:
            if state['original_config_existed']:
                atomic_write(config,original,state['original_config_mode'])
            else:
                config.unlink()
        if key_created:
            subprocess.run(['/usr/bin/security','delete-generic-password','-a',settings['keychain_account'],
                            '-s',settings['keychain_service']],capture_output=True)
        if root.exists():
            shutil.rmtree(root)
        raise SystemExit('Installation failed and changes were rolled back: '+type(exc).__name__)
    print(json.dumps({'installed':True,'platform':sys.platform,'install_dir':str(root),
                      'context_tokens':500000,'reasoning':'high','restart_codex_and_create_new_chat':True}))


if __name__=='__main__':
    main()
