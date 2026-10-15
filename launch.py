#!/usr/bin/env python3
"""Run an isolated Codex profile with a process-scoped DeepSeek bridge."""
import argparse
import copy
import getpass
import json
import os
import pathlib
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent


def toml_value(value):
    if isinstance(value, dict):
        return '{ ' + ', '.join(json.dumps(k) + ' = ' + toml_value(v) for k, v in value.items()) + ' }'
    return json.dumps(value)


def build_catalog(codex):
    # Metadata discovery can initialize Codex's home even for --bundled.
    # Isolate those incidental files; read the target account's cache below.
    with tempfile.TemporaryDirectory(prefix='codex-model-discovery-') as discovery:
        discovery_env = dict(os.environ, CODEX_HOME=discovery)
        raw = subprocess.check_output([codex, 'debug', 'models', '--bundled'],
                                      text=True, env=discovery_env)
    catalog = json.loads(raw)
    # Account metadata can supply capabilities newer than the bundled defaults.
    cached = pathlib.Path(os.environ.get('CODEX_HOME', pathlib.Path.home() / '.codex')) / 'models_cache.json'
    if cached.exists():
        try:
            current = json.loads(cached.read_text())
            if current.get('models'):
                catalog = {'models': current['models']}
        except (OSError, ValueError):
            pass
    # Native v2 can carry the actual delegated task only in OpenAI-encrypted
    # agent_message content. Cross-provider tasks must use Codex's local v1
    # orchestration with ordinary Responses function calls instead.
    for m in catalog['models']:
        if m['slug'].startswith('gpt-') or m['slug'] == 'codex-auto-review':
            m.update({'multi_agent_version': 'v1', 'use_responses_lite': False,
                      'tool_mode': None, 'prefer_websockets': False})
            messages = m.setdefault('model_messages', {})
            hint = ('When native subagent tools are deferred, discover them with tool_search '
                    'before concluding that delegation is unavailable.')
            if isinstance(messages, dict):
                messages['persistent_instructions'] = (messages.get('persistent_instructions') or '') + '\n' + hint
    base = copy.deepcopy(next(m for m in catalog['models'] if m['slug'] == 'gpt-6.1-sol'))
    base.update({
        'slug': 'deepseek-v4.1-flash', 'display_name': 'DeepSeek V4.1 Flash (relay)',
        'description': 'DeepSeek through a local Responses to Chat Completions bridge.',
        'visibility': 'list', 'supported_in_api': True, 'priority': 100,
        'upgrade': None, 'default_reasoning_level': 'high',
        'supported_reasoning_levels': [
            {'effort': 'low', 'description': 'Lower thinking effort'},
            {'effort': 'high', 'description': 'Default thinking effort'},
            {'effort': 'max', 'description': 'Maximum thinking effort'}],
        'additional_speed_tiers': [], 'service_tiers': [],
        'context_window': 500000, 'max_context_window': 500000,
        'effective_context_window_percent': 100, 'auto_compact_token_limit': 400000,
        'input_modalities': ['text'], 'supports_image_detail_original': False,
        'supports_search_tool': False, 'prefer_websockets': False,
        'use_responses_lite': False, 'tool_mode': None, 'multi_agent_version': 'v1',
        'multi_agent_reasoning_effort': 'high', 'experimental_supported_tools': [],
        'supports_reasoning_effort_updates': True,
        'supports_experimental_context': False, 'support_verbosity': False,
        'default_reasoning_summary': 'none',
        'shell_type': 'unified_exec', 'apply_patch_tool_type': 'freeform',
        'base_instructions': 'You are a coding agent running inside the Codex harness. Follow the task and all applicable instructions. Use the available tools when needed, inspect their real outputs, and report only verified results. Keep delegated tasks bounded. Never invent tool outputs. Use apply_patch for file changes. Do not expose credentials.'})
    for key in ('model_messages', 'availability_nux', 'available_access_programs', 'auto_review_model_override'):
        base.pop(key, None)
    catalog['models'] = [m for m in catalog['models'] if m['slug'] != base['slug']] + [base]
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='gpt-6.1-sol', choices=['gpt-6.1-sol', 'deepseek-v4.1-flash'])
    parser.add_argument('--relay', required=True, help='HTTPS relay base URL, normally ending in /v1')
    parser.add_argument('--exec', dest='execute', action='store_true')
    parser.add_argument('--audit-dir', type=pathlib.Path)
    parser.add_argument('--sandbox', default='read-only', choices=['read-only', 'workspace-write'])
    parser.add_argument('--cwd', type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument('--prompt')
    parser.add_argument('--timeout', type=int, default=0, help='Stop the entire run after this many seconds; 0 disables the limit')
    args = parser.parse_args()
    os.umask(0o077)
    codex = shutil.which('codex')
    if not codex and sys.platform == 'darwin':
        bundled = pathlib.Path('/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex')
        if bundled.is_file():
            codex = str(bundled)
    if not codex:
        raise SystemExit('codex executable was not found on PATH')
    key = os.environ.pop('DEEPSEEK_API_KEY', '')
    if not key:
        key = getpass.getpass('DeepSeek relay API key (not saved): ')
    if not key:
        raise SystemExit('An API key is required')
    runtime = pathlib.Path(tempfile.mkdtemp(prefix='codex-deepseek-'))
    catalog_path = runtime / 'models.json'
    catalog_path.write_text(json.dumps(build_catalog(codex), ensure_ascii=False))
    audit_dir = args.audit_dir
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)
    gateway_env = os.environ.copy()
    gateway_env['DEEPSEEK_API_KEY'] = key
    token = secrets.token_urlsafe(32)
    gateway_env['CODEX_BRIDGE_TOKEN'] = token
    gateway_args = [sys.executable, str(ROOT / 'bridge.py'), '--port', '0', '--relay', args.relay,
                    '--catalog', str(catalog_path), '--follow-pid', str(os.getpid())]
    if audit_dir:
        gateway_args += ['--audit', str(audit_dir / 'gateway.jsonl')]
    gateway = subprocess.Popen(gateway_args, env=gateway_env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    gateway_env.pop('DEEPSEEK_API_KEY', None)
    del key
    cli = None
    try:
        ready = json.loads(gateway.stdout.readline())
        port = ready['port']
        # Ignore user configuration so the validation session has no personal MCP,
        # plugins, memories, notifications, or credential-bearing shell environment.
        overrides = {
            'model': args.model, 'model_provider': 'deepseek_bridge',
            'model_catalog_json': str(catalog_path),
            'model_providers.deepseek_bridge.name': 'Local GPT + DeepSeek bridge',
            'model_providers.deepseek_bridge.base_url': 'http://127.0.0.1:' + str(port),
            'model_providers.deepseek_bridge.wire_api': 'responses',
            'model_providers.deepseek_bridge.requires_openai_auth': True,
            'model_providers.deepseek_bridge.env_http_headers': {'X-Bridge-Key': 'CODEX_BRIDGE_TOKEN'},
            'model_providers.deepseek_bridge.supports_websockets': False,
            'model_providers.deepseek_bridge.request_max_retries': 0,
            'model_providers.deepseek_bridge.stream_max_retries': 0,
            'model_reasoning_effort': 'high' if args.model == 'deepseek-v4.1-flash' else 'low',
            'web_search': 'disabled',
            'agents.enabled': True, 'agents.max_concurrent_threads_per_session': 2,
            'agents.default_subagent_model': 'deepseek-v4.1-flash',
            'agents.default_subagent_reasoning_effort': 'high',
            'features.apps': False, 'features.plugins': False, 'features.memories': False,
            'skills.bundled.enabled': False,
            'skills.include_instructions': False,
            'shell_environment_policy.inherit': 'core',
            'approval_policy': 'never',
        }
        command = [codex]
        if args.execute:
            command += ['exec', '--ignore-user-config', '--ignore-rules', '--ephemeral',
                        '--skip-git-repo-check', '--json', '--color', 'never']
        else:
            # Interactive users get the same independent config as exec through a
            # temporary user home. Auth uses the original home, so prefer exec for now.
            raise SystemExit('This validation launcher currently requires --exec')
        command += ['--sandbox', args.sandbox, '--cd', str(args.cwd.resolve())]
        for k, v in overrides.items():
            command += ['-c', k + '=' + toml_value(v)]
        command.append(args.prompt or '-')
        clean_env = os.environ.copy()
        clean_env['CODEX_BRIDGE_TOKEN'] = token
        clean_env.pop('DEEPSEEK_API_KEY', None)
        clean_env.pop('OPENAI_API_KEY', None)
        for name in list(clean_env):
            if name.startswith('CODEX_') and name not in ('CODEX_HOME', 'CODEX_BRIDGE_TOKEN'):
                clean_env.pop(name, None)
        out = open(audit_dir / 'codex.jsonl', 'w') if audit_dir else None
        err = open(audit_dir / 'codex.stderr', 'w') if audit_dir else None
        cli = subprocess.Popen(command, env=clean_env, stdout=out, stderr=err,
                               stdin=subprocess.DEVNULL if args.prompt else None, start_new_session=True)
        try:
            code = cli.wait(timeout=args.timeout or None)
        except subprocess.TimeoutExpired:
            os.killpg(cli.pid, signal.SIGTERM)
            try:
                cli.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(cli.pid, signal.SIGKILL)
                cli.wait()
            code = 124
        if out:
            out.close()
            err.close()
        if audit_dir:
            print(json.dumps({'exit_code': code, 'audit_dir': str(audit_dir.resolve()),
                              'model': args.model, 'gateway_stopped_on_exit': True}))
        return code
    except (KeyboardInterrupt, SystemExit):
        if cli and cli.poll() is None:
            os.killpg(cli.pid, signal.SIGTERM)
            cli.wait(timeout=10)
        raise
    finally:
        if gateway.poll() is None:
            gateway.terminate()
        try:
            gateway.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gateway.kill()
            gateway.wait()
        shutil.rmtree(runtime)


if __name__ == '__main__':
    sys.exit(main())
