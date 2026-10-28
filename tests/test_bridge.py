"""Protocol and credential-boundary tests; no live API requests."""
import io
import json
import pathlib
import socket
import ssl
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from bridge import ContextStore, Gateway, Handler, ProtocolError, convert_usage, relay_events, to_chat


def upstream(*chunks):
    data = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
    return io.BytesIO(data.encode())


class ConversionTests(unittest.TestCase):
    def test_namespaced_calls_and_results_round_trip(self):
        body = {'instructions': 'system instructions', 'tools': [{'type': 'namespace', 'name': 'collaboration',
                'tools': [{'type': 'function', 'name': 'send_message', 'parameters': {'type': 'object'}}]}],
                'input': [{'type': 'function_call', 'namespace': 'collaboration', 'name': 'send_message',
                           'call_id': 'call_1', 'arguments': '{"target":"worker"}'},
                          {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'delivered'}]}
        chat, mapping = to_chat(body, 'ds', 8192)
        self.assertEqual(chat['messages'][1]['tool_calls'][0]['function']['name'], 'collaboration__send_message')
        self.assertEqual(chat['messages'][2]['tool_call_id'], 'call_1')
        self.assertEqual(mapping['collaboration__send_message']['namespace'], 'collaboration')

    def test_fragmented_custom_patch_stream(self):
        _, mapping = to_chat({'input': 'x', 'tools': [{'type': 'custom', 'name': 'apply_patch'}]}, 'ds', 8192)
        raw = upstream({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_patch',
                       'function': {'name': 'apply_patch', 'arguments': '{"input":"hello'}}]}}]},
                       {'choices': [{'delta': {'tool_calls': [{'index': 0,
                       'function': {'arguments': '\\nworld"}'}}]}, 'finish_reason': 'tool_calls'}]})
        events = list(relay_events(raw, 'ds', mapping))
        item = events[-1]['response']['output'][0]
        self.assertEqual(item['type'], 'custom_tool_call')
        self.assertEqual(item['input'], 'hello\nworld')
        self.assertEqual(item['call_id'], 'call_patch')
        self.assertEqual([e['sequence_number'] for e in events], list(range(len(events))))

    def test_length_is_incomplete_not_success(self):
        events = list(relay_events(upstream({'choices': [{'delta': {'content': 'partial'}, 'finish_reason': 'length'}]}), 'ds', {}))
        self.assertEqual(events[-1]['type'], 'response.incomplete')
        self.assertEqual(events[-1]['response']['incomplete_details']['reason'], 'max_output_tokens')

    def test_broken_stream_fails(self):
        with self.assertRaises(ProtocolError):
            list(relay_events(upstream({'choices': [{'delta': {'content': 'partial'}}]}), 'ds', {}))

    def test_unsupported_inputs_are_rejected(self):
        for body in [{'previous_response_id': 'resp_old'},
                     {'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'x'}]}]},
                     {'tools': [{'type': 'code_interpreter'}]}]:
            with self.subTest(body=body), self.assertRaises(ProtocolError):
                to_chat(body, 'ds', 8192)

    def test_inherited_hosted_search_is_not_advertised_to_chat(self):
        chat, mapping = to_chat({'input': 'text task', 'tools': [{'type': 'web_search'}]}, 'ds', 8192)
        self.assertNotIn('tools', chat)
        self.assertEqual(mapping, {})

    def test_model_and_output_limit_cannot_escape(self):
        chat, _ = to_chat({'input': 'x', 'model': 'other', 'max_output_tokens': 1000000,
                          'reasoning': {'effort': 'high'}, 'service_tier': 'priority'}, 'ds', 8192)
        self.assertEqual(chat['model'], 'ds')
        self.assertEqual(chat['max_tokens'], 8192)
        self.assertNotIn('service_tier', chat)
        self.assertNotIn('reasoning', chat)

    def test_thinking_default_and_three_explicit_efforts(self):
        default, _ = to_chat({'input': 'x'}, 'ds', 8192)
        self.assertEqual(default['reasoning_effort'], 'high')
        for effort in ('low', 'high', 'max'):
            with self.subTest(effort=effort):
                chat, _ = to_chat({'input': 'x', 'reasoning': {'effort': effort}}, 'ds', 8192)
                self.assertEqual(chat['reasoning_effort'], effort)
                self.assertEqual(chat['thinking'], {'type': 'enabled'})
        for effort in ('medium', 'ultra', 'none'):
            with self.subTest(effort=effort), self.assertRaises(ProtocolError):
                to_chat({'input': 'x', 'reasoning': {'effort': effort}}, 'ds', 8192)

    def test_thinking_reasoning_and_tool_calls_share_assistant_message(self):
        body = {'input': [
            {'type': 'message', 'role': 'user', 'content': 'Do the task'},
            {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'portable reasoning'}]},
            {'type': 'message', 'role': 'assistant', 'content': 'I will use a tool.'},
            {'type': 'function_call', 'name': 'probe', 'call_id': 'call_1', 'arguments': '{}'},
            {'type': 'function_call', 'name': 'probe', 'call_id': 'call_2', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'result one'},
            {'type': 'function_call_output', 'call_id': 'call_2', 'output': 'result two'},
        ]}
        chat, _ = to_chat(body, 'ds', 8192)
        self.assertEqual(len(chat['messages']), 4)
        assistant = chat['messages'][1]
        self.assertEqual(assistant['reasoning_content'], 'portable reasoning')
        self.assertEqual(assistant['content'], 'I will use a tool.')
        self.assertEqual(len(assistant['tool_calls']), 2)
        self.assertEqual(chat['messages'][2]['role'], 'tool')

    def test_missing_usage_details_remain_unreported(self):
        usage = convert_usage({'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})
        self.assertNotIn('output_tokens_details', usage)
        self.assertNotIn('input_tokens_details', usage)
        usage = convert_usage({'prompt_tokens_details': None, 'completion_tokens_details': None,
                               'prompt_cache_hit_tokens': 7})
        self.assertEqual(usage['input_tokens_details']['cached_tokens'], 7)
        usage = convert_usage({'completion_tokens_details': {'reasoning_tokens': 0}})
        self.assertEqual(usage['output_tokens_details']['reasoning_tokens'], 0)

    def test_native_agent_message_uses_plaintext_only(self):
        chat, _ = to_chat({'input': [{'type': 'agent_message', 'author': 'parent',
                         'recipient': 'child', 'content': [{'type': 'input_text', 'text': 'Read the test file'}]}]}, 'ds', 8192)
        self.assertIn('Read the test file', chat['messages'][0]['content'])
        self.assertNotIn('opaque-gpt-data', json.dumps(chat))
        with self.assertRaises(ProtocolError):
            to_chat({'input': [{'type': 'agent_message', 'content': [
                    {'type': 'input_text', 'text': 'Payload:'},
                    {'type': 'encrypted_content', 'encrypted_content': 'opaque'}]}]}, 'ds', 8192)


class FakeUpstream(io.BytesIO):
    status = 200
    headers = {'Content-Type': 'text/event-stream'}


class CaptureOpener:
    def __init__(self):
        self.requests = []

    def open(self, req, **_):
        self.requests.append(req)
        raw = upstream({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})
        return FakeUpstream(raw.getvalue())


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.server = Gateway(('127.0.0.1', 0), Handler)
        self.server.api_key, self.server.token = 'fake-relay-secret', 'local-capability'
        self.server.model, self.server.relay = 'ds', 'https://relay.example/v1'
        self.server.max_output, self.server.catalog = 8192, {'models': []}
        self.server.gpt_models = {'gpt-test'}
        self.server.audit_path, self.server.audit_lock = None, threading.Lock()
        self.server.opener = CaptureOpener()
        self.context_temp = tempfile.TemporaryDirectory()
        self.server.context_store = ContextStore(pathlib.Path(self.context_temp.name)/'contexts')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.context_temp.cleanup()

    def request(self, model, token='local-capability', body=None):
        payload = {'model': model, 'input': 'x', 'stream': False}
        if body:
            payload.update(body)
        req = urllib.request.Request('http://127.0.0.1:' + str(self.server.server_port) + '/responses',
              data=json.dumps(payload).encode(),
              headers={'Content-Type': 'application/json', 'Authorization': 'Bearer fake-chatgpt-token',
                       'ChatGPT-Account-ID': 'fake-account', 'Cookie': 'fake-cookie', 'X-Bridge-Key': token})
        return urllib.request.urlopen(req, timeout=5)

    def test_deepseek_headers_are_rebuilt(self):
        with self.request('ds') as response:
            self.assertEqual(json.load(response)['output'][0]['content'][0]['text'], 'ok')
        req = self.server.opener.requests[0]
        headers = {k.lower(): v for k, v in req.header_items()}
        self.assertEqual(headers['authorization'], 'Bearer fake-relay-secret')
        self.assertNotIn('chatgpt-account-id', headers)
        self.assertNotIn('cookie', headers)
        self.assertNotIn('x-bridge-key', headers)

    def test_gpt_auth_preserved_relay_secret_absent(self):
        with self.request('gpt-test') as response:
            response.read()
        req = self.server.opener.requests[0]
        headers = {k.lower(): v for k, v in req.header_items()}
        self.assertEqual(headers['authorization'], 'Bearer fake-chatgpt-token')
        self.assertEqual(headers['chatgpt-account-id'], 'fake-account')
        self.assertNotIn('x-bridge-key', headers)
        self.assertNotIn('fake-relay-secret', json.dumps(headers))
        self.assertTrue(req.full_url.startswith('https://chatgpt.com/backend-api/codex/'))

    def test_unauthorized_and_unknown_routes_never_forward(self):
        for model, token, code in [('ds', 'wrong', 401), ('unknown', 'local-capability', 400)]:
            with self.subTest(model=model), self.assertRaises(urllib.error.HTTPError) as cm:
                self.request(model, token)
            self.assertEqual(cm.exception.code, code)
            cm.exception.close()
        self.assertEqual(self.server.opener.requests, [])

    def test_upstream_error_preserves_code_without_echoing_body(self):
        def fail(*args, **kwargs):
            payload = json.dumps({'error': {'code': 'system_memory_overloaded',
                               'message': 'private fake-relay-secret body'}}).encode()
            raise urllib.error.HTTPError('https://relay.example', 503, 'Unavailable', {}, io.BytesIO(payload))
        self.server.opener.open = fail
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.request('ds')
        self.assertEqual(cm.exception.code, 503)
        body = cm.exception.read().decode()
        cm.exception.close()
        self.assertIn('system_memory_overloaded', body)
        self.assertNotIn('fake-relay-secret', body)

    def test_native_compaction_http_then_continuation(self):
        with self.request('ds',body={'input':[{'role':'user','content':'Summarize task'},
                                             {'type':'compaction_trigger'}]}) as response:
            compacted=json.load(response)
        item=compacted['output'][0]
        self.assertEqual(item['type'],'compaction')
        self.assertEqual(len(self.server.opener.requests),1)
        with self.request('ds',body={'input':[item,{'role':'user','content':'Continue'}]}) as response:
            self.assertEqual(json.load(response)['status'],'completed')
        replay=json.loads(self.server.opener.requests[-1].data)
        self.assertIn('Context summary retained',replay['messages'][0]['content'])
        self.assertNotIn(item['encrypted_content'],json.dumps(replay))
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.request('gpt-test',body={'input':[item]})
        self.assertEqual(cm.exception.code,400)
        cm.exception.close()
        self.assertEqual(len(self.server.opener.requests),2)

    def test_summary_budget_is_independent_of_ordinary_output_limit(self):
        with self.request('ds', body={'max_output_tokens': 256}) as response:
            json.load(response)
        self.assertEqual(json.loads(self.server.opener.requests[-1].data)['max_tokens'], 256)
        with self.request('ds', body={'max_output_tokens': 256,
                'input': [{'role': 'user', 'content': 'Task'}, {'type': 'compaction_trigger'}]}) as response:
            self.assertEqual(json.load(response)['status'], 'completed')
        summary_request = json.loads(self.server.opener.requests[-1].data)
        self.assertEqual(summary_request['max_tokens'], 16384)
        self.assertEqual(summary_request['reasoning_effort'], 'high')
        self.assertNotIn('tools', summary_request)
        with self.request('ds') as response:
            json.load(response)
        self.assertEqual(json.loads(self.server.opener.requests[-1].data)['max_tokens'], 8192)

    def scripted_compaction(self, finishes):
        requests = self.server.opener.requests
        def opened(req, **_):
            requests.append(req)
            finish = finishes[len(requests) - 1]
            raw = upstream({'choices': [{'delta': {'content': 'fixture summary'},
                                        'finish_reason': finish}]})
            return FakeUpstream(raw.getvalue())
        self.server.opener.open = opened

    def test_http_summary_retry_uses_original_history_and_shorter_prompt(self):
        self.scripted_compaction(['length', 'stop'])
        with self.request('ds', body={'input': [
                {'role': 'user', 'content': 'Original task marker 731'},
                {'type': 'compaction_trigger'}]}) as response:
            compacted = json.load(response)
        self.assertEqual(compacted['status'], 'completed')
        self.assertEqual(len(self.server.opener.requests), 2)
        first, retry = [json.loads(r.data) for r in self.server.opener.requests]
        self.assertEqual(first['messages'][:-1], retry['messages'][:-1])
        self.assertIn('at most 2000 tokens', retry['messages'][-1]['content'])
        self.assertEqual(retry['max_tokens'], 16384)
        self.assertNotIn('tools', retry)
        self.assertEqual(len(list(self.server.context_store.directory.iterdir())), 1)

    def test_streamed_compaction_failure_is_a_native_terminal_not_a_silent_eof(self):
        self.scripted_compaction(['length', 'length'])
        with self.request('ds', body={'stream': True,
                'input': [{'role': 'user', 'content': 'Task'}, {'type': 'compaction_trigger'}]}) as response:
            events = [json.loads(line[6:]) for line in response if line.startswith(b'data: ')]
        self.assertEqual(events[-1]['type'], 'response.failed')
        failed = events[-1]['response']
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['id'], events[0]['response']['id'])
        self.assertIn('summary output limit', failed['error']['message'])
        self.assertNotIn('response.completed', [e['type'] for e in events])
        self.assertEqual(len(self.server.opener.requests), 2)
        self.assertEqual(list(self.server.context_store.directory.iterdir()), [])

    def test_nonstream_compaction_failure_reports_upstream_failure_and_preserves_history(self):
        self.scripted_compaction(['length', 'length'])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.request('ds', body={'input': [
                {'role': 'user', 'content': 'Task'}, {'type': 'compaction_trigger'}]})
        self.assertEqual(cm.exception.code, 502)
        body = json.loads(cm.exception.read())
        cm.exception.close()
        self.assertIn('original context was not compacted', body['error']['message'])
        self.assertEqual(list(self.server.context_store.directory.iterdir()), [])

    def test_temporary_dns_failure_before_connect_is_retried_once(self):
        original_open = self.server.opener.open
        attempts = []
        def open_request(req, **kwargs):
            attempts.append(req)
            if len(attempts) == 1:
                raise urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, 'private DNS detail'))
            return original_open(req, **kwargs)
        self.server.opener.open = open_request
        with self.request('ds') as response:
            self.assertEqual(json.load(response)['status'], 'completed')
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].data, attempts[1].data)

    def test_refused_connection_retry_is_bounded_and_reports_reason(self):
        attempts = []
        def open_request(req, **_):
            attempts.append(req)
            raise urllib.error.URLError(ConnectionRefusedError(61, 'private proxy URL'))
        self.server.opener.open = open_request
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.request('gpt-test')
        self.assertEqual(cm.exception.code, 502)
        text = cm.exception.read().decode()
        cm.exception.close()
        self.assertIn('connection_refused', text)
        self.assertNotIn('private', text)
        self.assertEqual(len(attempts), 2)

    def test_tls_failures_are_classified_without_retry_or_secret_exposure(self):
        attempts = []
        self.server.audit_path = str(pathlib.Path(self.context_temp.name) / 'audit.jsonl')
        def open_request(req, **_):
            attempts.append(req)
            raise urllib.error.URLError(ssl.SSLEOFError(8, 'private fake-relay-secret URL'))
        self.server.opener.open = open_request
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.request('ds')
        text = cm.exception.read().decode()
        cm.exception.close()
        self.assertIn('tls_eof', text)
        audit = pathlib.Path(self.server.audit_path).read_text()
        self.assertIn('SSLEOFError', audit)
        self.assertNotIn('fake-relay-secret', text + audit)
        self.assertNotIn('private', text + audit)
        self.assertEqual(len(attempts), 1)

    def test_resets_and_timeouts_with_uncertain_delivery_are_not_retried(self):
        for error in (ConnectionResetError(54, 'private'), TimeoutError('private')):
            attempts = []
            def open_request(req, **_):
                attempts.append(req)
                raise urllib.error.URLError(error)
            self.server.opener.open = open_request
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.request('ds')
            cm.exception.close()
            self.assertEqual(len(attempts), 1)

    def test_a_started_response_is_never_replayed(self):
        attempts = []
        class BrokenRead(FakeUpstream):
            def __iter__(self):
                raise urllib.error.URLError(ConnectionResetError(54, 'private'))
        def open_request(req, **_):
            attempts.append(req)
            return BrokenRead(b'')
        self.server.opener.open = open_request
        with self.request('ds', body={'stream': True}) as response:
            events = [json.loads(line[6:]) for line in response if line.startswith(b'data: ')]
        self.assertEqual(events[-1]['type'], 'response.failed')
        self.assertIn('connection_reset', events[-1]['response']['error']['message'])
        self.assertEqual(len(attempts), 1)


if __name__ == '__main__':
    unittest.main()
