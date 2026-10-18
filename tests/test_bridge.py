"""Protocol and credential-boundary tests; no live API requests."""
import io
import json
import pathlib
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


if __name__ == '__main__':
    unittest.main()
