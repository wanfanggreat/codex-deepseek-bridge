"""Portable install planning, config preservation, and credential file checks."""
import argparse
import hashlib
import json
import os
import pathlib
import plistlib
import socket
import subprocess
import sys
import tempfile
import tomllib
import unittest
import urllib.request
from unittest.mock import patch
import bridge
from desktop_service import load_key
from install import make_config, render_services, validate_relay
from manage import restored_config
from launch import build_catalog, provider_overrides


class InstallTests(unittest.TestCase):
    def plan(self, original):
        content, state = make_config(original,pathlib.Path('/tmp/test-user/.codex/deepseek-bridge'),'fake-capability',24589)
        state['installed_config_sha256'] = hashlib.sha256(content.encode()).hexdigest()
        return content,state

    def test_existing_agent_preferences_and_roles_preserved(self):
        old = ('model = "gpt-6.1-sol"\nmodel_reasoning_effort = "max"\nmodel_provider = "openai"\n'
               '[mcp_servers.example]\ncommand = "example"\n'
               '[agents]\nenabled = false\nmax_concurrent_threads_per_session = 4\njob_max_runtime_seconds = 123\n'
               '[agents.editor]\nconfig_file = "/tmp/editor.toml"\n'
               '[features]\nmemories = true\n')
        content,state = self.plan(old)
        parsed = tomllib.loads(content)
        self.assertEqual(parsed['model_reasoning_effort'],'max')
        self.assertEqual(parsed['agents']['max_concurrent_threads_per_session'],4)
        self.assertEqual(parsed['agents']['job_max_runtime_seconds'],123)
        self.assertEqual(parsed['agents']['editor']['config_file'],'/tmp/editor.toml')
        self.assertEqual(parsed['agents']['default_subagent_reasoning_effort'],'high')
        self.assertEqual(parsed['mcp_servers'],tomllib.loads(old)['mcp_servers'])
        self.assertEqual(parsed['model_providers']['local_deepseek_chat']['capabilities']['remote_compaction'],'unsupported')
        self.assertEqual(parsed['model_providers']['local_deepseek_bridge']['capabilities']['remote_compaction'],'v2')
        self.assertEqual(restored_config(content,state,old.encode()),old.encode())

    def test_uninstall_preserves_later_unrelated_preferences(self):
        old = 'model = "gpt-6.1-sol"\nmodel_reasoning_effort = "high"\nmodel_provider = "openai"\n[agents]\nmax_concurrent_threads_per_session = 3\n'
        content,state = self.plan(old)
        edited = content.replace('model_reasoning_effort = "high"','model_reasoning_effort = "low"')
        restored = tomllib.loads(restored_config(edited,state,old.encode()).decode())
        expected = tomllib.loads(old)
        expected['model_reasoning_effort'] = 'low'
        self.assertEqual(restored,expected)

    def test_modified_managed_block_refuses_uninstall(self):
        content,state = self.plan('model = "gpt-6.1-sol"\n')
        with self.assertRaises(ValueError):
            restored_config(content.replace('fake-capability','edited-capability'),state,b'')

    def test_empty_and_missing_final_newline_configs(self):
        for old in ['', 'model = "gpt-6.1-sol"', '[features]\nmemories = true']:
            with self.subTest(old=old):
                content,state = self.plan(old)
                tomllib.loads(content)
                self.assertEqual(restored_config(content,state,old.encode()),old.encode())

    def test_existing_custom_providers_or_roles_are_not_overwritten(self):
        for old in ['model_provider = "other"\n', 'model_catalog_json = "/tmp/other.json"\n',
                    '[agents.deepseek]\nconfig_file = "mine.toml"\n']:
            with self.subTest(old=old),self.assertRaises(ValueError):
                self.plan(old)

    def test_macos_socket_is_loopback_and_python_path_portable(self):
        root=pathlib.Path('/Users/test/.codex/deepseek-bridge')
        services=render_services('darwin',root,'/custom/python3',24589,pathlib.Path('/Users/test'))
        data=plistlib.loads(next(iter(services.values())))
        self.assertEqual(data['ProgramArguments'][0],'/custom/python3')
        self.assertEqual(data['Sockets']['Listener']['SockNodeName'],'127.0.0.1')
        self.assertNotIn('liming',next(iter(services.values())).decode())

    def test_linux_socket_unit_and_paths_with_spaces(self):
        services=render_services('linux',pathlib.Path('/home/test user/.codex/deepseek-bridge'),
                                 '/opt/python/bin/python3',24589,pathlib.Path('/home/test user'))
        socket_unit=next(v.decode() for k,v in services.items() if k.suffix=='.socket')
        service=next(v.decode() for k,v in services.items() if k.suffix=='.service')
        self.assertIn('ListenStream=127.0.0.1:24589',socket_unit)
        self.assertIn('UMask=0077',service)
        self.assertIn('"/home/test user/.codex/deepseek-bridge/desktop_service.py"',service)
        with self.assertRaises(ValueError):
            render_services('linux',pathlib.Path('/home/test%user/.codex'),'python',24589,pathlib.Path('/home/test'))

    def test_linux_private_key_permissions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)
            (root/'relay.key').write_text('fake-test-key\n')
            (root/'relay.key').chmod(0o600)
            self.assertEqual(load_key(root,{'platform':'linux'}),'fake-test-key')
            (root/'relay.key').chmod(0o644)
            with self.assertRaises(RuntimeError):
                load_key(root,{'platform':'linux'})

    def test_relay_url_validation(self):
        self.assertEqual(validate_relay('https://relay.example/v1/'),'https://relay.example/v1')
        for url in ['http://relay.example','https://key@relay.example','https://relay.example?key=x']:
            with self.subTest(url=url),self.assertRaises(ValueError):
                validate_relay(url)

    def test_systemd_rejects_foreign_socket_activation_before_serving(self):
        args=argparse.Namespace(relay='https://relay.example/v1',catalog='',systemd_socket=True)
        with tempfile.TemporaryDirectory() as temp:
            path=pathlib.Path(temp)/'models.json';path.write_text('{"models":[]}')
            args.catalog=str(path)
            with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'fake','LISTEN_PID':'0','LISTEN_FDS':'1'}):
                with self.assertRaises(SystemExit):
                    bridge.serve(args)

    def test_systemd_fd3_handoff_serves_loopback_health_without_api_request(self):
        source = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp, socket.socket() as listener:
            catalog = pathlib.Path(temp)/'models.json'
            catalog.write_text('{"models":[]}')
            listener.bind(('127.0.0.1',0))
            listener.listen()
            port = listener.getsockname()[1]
            code = (
                'import argparse,os,bridge;'
                f'os.dup2({listener.fileno()},3);'
                'os.environ["LISTEN_PID"]=str(os.getpid());'
                'os.environ["LISTEN_FDS"]="1";'
                'os.environ["DEEPSEEK_API_KEY"]="fake-test-key";'
                'os.environ["CODEX_BRIDGE_TOKEN"]="fake-test-capability";'
                'bridge.serve(argparse.Namespace('
                f'catalog={str(catalog)!r},relay="https://relay.example/v1",'
                'systemd_socket=True,launchd_socket=None,model="ds",max_output=8192,'
                'audit=None,follow_pid=None,idle_timeout=0))')
            process = subprocess.Popen([sys.executable,'-c',code],cwd=source,
                                       pass_fds=(listener.fileno(),),stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,text=True)
            try:
                ready = json.loads(process.stdout.readline())
                self.assertEqual(ready['port'],port)
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=5) as response:
                    self.assertEqual(json.load(response)['status'],'ok')
            finally:
                process.terminate()
                process.communicate(timeout=10)

    def test_catalog_discovery_does_not_initialize_target_home(self):
        with tempfile.TemporaryDirectory() as temp:
            def fake_codex(*args,**kwargs):
                discovered = pathlib.Path(kwargs['env']['CODEX_HOME'])
                self.assertNotEqual(discovered,pathlib.Path(temp))
                (discovered/'incidental-log').write_text('metadata only')
                return '{"models":[{"slug":"gpt-6.1-sol"}]}'
            with patch.dict(os.environ,{'CODEX_HOME':temp}), patch('launch.subprocess.check_output',side_effect=fake_codex):
                catalog = build_catalog('fake-codex')
            model = next(m for m in catalog['models'] if m['slug']=='deepseek-v4.1-flash')
            self.assertEqual(model['context_window'],500000)
            self.assertEqual(model['default_reasoning_level'],'high')
            self.assertEqual(list(pathlib.Path(temp).iterdir()),[])

    def test_cli_gpt_and_deepseek_compaction_providers_are_independent(self):
        settings = provider_overrides(24589)
        self.assertEqual(settings['model_providers.deepseek_bridge.capabilities.remote_compaction'],'v2')
        self.assertEqual(settings['model_providers.deepseek_chat.capabilities.remote_compaction'],'unsupported')
        self.assertEqual(settings['model_providers.deepseek_bridge.base_url'],
                         settings['model_providers.deepseek_chat.base_url'])


if __name__=='__main__':
    unittest.main()
