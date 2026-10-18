"""Chat tool name limits, collision isolation, and native-call restoration."""
import io
import json
import unittest

from bridge import ProtocolError, relay_events, to_chat, tool_definitions, tool_wire_names


class ToolNameTests(unittest.TestCase):
    namespace = 'mcp__codex_apps__codex_document_control'
    name = '_execute_document_command'

    def definition(self, name=None, namespace=None, kind='function'):
        tool = {'type': kind, 'name': name or self.name}
        return {'type': 'namespace', 'name': namespace or self.namespace, 'tools': [tool]}

    def call_stream(self, wire, arguments='{}'):
        chunks = [
            {'choices': [{'delta': {'tool_calls': [
                {'index': 0, 'id': 'call_test', 'function': {'name': wire, 'arguments': arguments[:2]}}]}}]},
            {'choices': [{'delta': {'tool_calls': [
                {'index': 0, 'function': {'arguments': arguments[2:]}}]}, 'finish_reason': 'tool_calls'}]},
        ]
        data = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
        return io.BytesIO(data.encode())

    def test_actual_overlength_connector_names_are_advertised_with_unique_aliases(self):
        cases = [
            (self.namespace, self.name),
            (self.namespace, '_get_document_tool_schemas'),
            ('mcp__codex_apps__github', '_search_installed_repositories_streaming'),
            ('mcp__codex_apps__safety_settings', '_prepare_parental_control_update'),
            ('mcp__codex_apps__sites', '_create_source_repository_write_credential'),
        ]
        definitions = [self.definition(name, namespace) for namespace, name in cases]
        chat, mapping = to_chat({'input': 'test', 'tools': definitions}, 'ds', 8192)
        self.assertEqual(len(mapping), 5)
        for tool, (namespace, name) in zip(chat['tools'], cases):
            wire = tool['function']['name']
            self.assertRegex(wire, r'^[A-Za-z0-9_-]{1,64}$')
            self.assertEqual(mapping[wire], {'namespace': namespace, 'name': name, 'type': 'function'})
        _, reordered = to_chat({'input': 'test', 'tools': list(reversed(definitions)) +
                               [{'type': 'function', 'name': 'exec_command'}]}, 'ds', 8192)
        for wire, meta in mapping.items():
            self.assertEqual(reordered[wire], meta)

    def test_alias_is_used_for_history_tool_choice_and_stream_restoration(self):
        history = {'type': 'function_call', 'namespace': self.namespace, 'name': self.name,
                   'call_id': 'call_old', 'arguments': '{"value":17}'}
        body = {'tools': [self.definition()], 'tool_choice': {'type': 'function',
                'namespace': self.namespace, 'name': self.name}, 'input': [history,
                {'type': 'function_call_output', 'call_id': 'call_old', 'output': '42'}]}
        chat, mapping = to_chat(body, 'ds', 8192)
        wire = chat['tools'][0]['function']['name']
        self.assertEqual(chat['tool_choice']['function']['name'], wire)
        self.assertEqual(chat['messages'][0]['tool_calls'][0]['function']['name'], wire)
        self.assertEqual(chat['messages'][1]['tool_call_id'], 'call_old')
        events = list(relay_events(self.call_stream(wire, '{"value":25}'), 'ds', mapping))
        item = events[-1]['response']['output'][0]
        self.assertEqual(item['namespace'], self.namespace)
        self.assertEqual(item['name'], self.name)
        self.assertEqual(item['arguments'], '{"value":25}')
        self.assertEqual(item['call_id'], 'call_test')
        # Replay Codex's restored identity, rather than the upstream alias.
        replay, _ = to_chat({'tools': [self.definition()], 'input': [
            item, {'type': 'function_call_output', 'call_id': 'call_test', 'output': '42'}]}, 'ds', 8192)
        self.assertEqual(replay['messages'][0]['tool_calls'][0]['function']['name'], wire)
        self.assertEqual(replay['messages'][1]['tool_call_id'], 'call_test')

    def test_history_without_tool_definitions_still_uses_valid_alias(self):
        call = {'type': 'function_call', 'namespace': self.namespace, 'name': self.name,
                'call_id': 'call_old', 'arguments': '{}'}
        with_tools, _ = to_chat({'tools': [self.definition()], 'input': [call]}, 'ds', 8192)
        without_tools, mapping = to_chat({'tools': [], 'input': [call]}, 'ds', 8192)
        expected = with_tools['tools'][0]['function']['name']
        self.assertEqual(without_tools['messages'][0]['tool_calls'][0]['function']['name'], expected)
        self.assertNotIn('tools', without_tools)
        self.assertEqual(mapping, {})

    def test_custom_tool_alias_restores_input(self):
        chat, mapping = to_chat({'input': 'test', 'tools': [self.definition(kind='custom')]}, 'ds', 8192)
        wire = chat['tools'][0]['function']['name']
        events = list(relay_events(self.call_stream(wire, '{"input":"patch text"}'), 'ds', mapping))
        item = events[-1]['response']['output'][0]
        self.assertEqual(item['type'], 'custom_tool_call')
        self.assertEqual(item['namespace'], self.namespace)
        self.assertEqual(item['name'], self.name)
        self.assertEqual(item['input'], 'patch text')
        replay, _ = to_chat({'tools': [self.definition(kind='custom')], 'input': [item]}, 'ds', 8192)
        self.assertEqual(replay['messages'][0]['tool_calls'][0]['function']['name'], wire)

    def test_ambiguous_flattened_names_get_distinct_order_independent_aliases(self):
        tools = [{'type': 'function', 'name': 'a__b'}, self.definition('b', 'a')]
        chat, mapping = to_chat({'input': 'test', 'tools': tools}, 'ds', 8192)
        self.assertEqual(len(mapping), 2)
        self.assertTrue(all(wire.startswith('dsbridge_') for wire in mapping))
        _, reversed_mapping = to_chat({'input': 'test', 'tools': list(reversed(tools))}, 'ds', 8192)
        self.assertEqual(mapping, reversed_mapping)
        for wire, meta in mapping.items():
            item = list(relay_events(self.call_stream(wire), 'ds', mapping))[-1]['response']['output'][0]
            self.assertEqual(item.get('namespace'), meta.get('namespace'))
            self.assertEqual(item['name'], meta['name'])

    def test_literal_tool_cannot_shadow_generated_alias(self):
        identity = (self.namespace, self.name)
        alias = tool_wire_names([identity])[identity]
        tools = [self.definition(), {'type': 'function', 'name': alias}]
        _, mapping = to_chat({'input': 'test', 'tools': tools}, 'ds', 8192)
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[alias]['name'], self.name)
        self.assertEqual(sum(meta['name'] == alias for meta in mapping.values()), 1)

    def test_short_names_and_boundary_are_preserved(self):
        for name in ('exec_command', 'a' * 64):
            _, mapping = tool_definitions([{'type': 'function', 'name': name}])
            self.assertIn(name, mapping)
        _, mapping = tool_definitions([{'type': 'function', 'name': 'a' * 65}])
        self.assertEqual(len(mapping), 1)
        self.assertLessEqual(len(next(iter(mapping))), 64)
        self.assertEqual(tool_wire_names([('collaboration', 'send_message')]),
                         {('collaboration', 'send_message'): 'collaboration__send_message'})

    def test_long_shared_prefix_and_illegal_characters_never_truncate_to_same_name(self):
        identities = [(None, 'x' * 80 + suffix) for suffix in ('one', 'two')]
        identities += [(None, 'tool.with spaces'), ('文档', '读取')]
        names = tool_wire_names(identities)
        self.assertEqual(len(set(names.values())), len(identities))
        for wire in names.values():
            self.assertRegex(wire, r'^[A-Za-z0-9_-]{1,64}$')

    def test_true_duplicate_definition_has_separate_error(self):
        for second_kind in ('function', 'custom'):
            with self.subTest(second_kind=second_kind), self.assertRaisesRegex(
                    ProtocolError, 'Duplicate Responses tool definition'):
                tool_definitions([{'type': 'function', 'name': 'duplicate'},
                                  {'type': second_kind, 'name': 'duplicate'}])

    def test_undeclared_alias_is_still_rejected(self):
        _, mapping = to_chat({'input': 'test', 'tools': [self.definition()]}, 'ds', 8192)
        with self.assertRaisesRegex(ProtocolError, 'undeclared tool'):
            list(relay_events(self.call_stream('dsbridge_unregistered'), 'ds', mapping))


if __name__ == '__main__':
    unittest.main()
