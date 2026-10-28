#!/usr/bin/env python3
"""Loopback-only Codex Responses -> Chat Completions bridge (stdlib only)."""
import argparse
import ctypes
import hashlib
import json
import hmac
import os
import pathlib
import re
import secrets
import socket
import ssl
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
       'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length',
       'content-encoding', 'accept-encoding'}


class ProtocolError(ValueError):
    pass


class CompactionError(ProtocolError):
    """A failed summary must leave the original history intact."""


def network_error_details(error):
    """Fixed categories and numeric diagnostics only; never log reason text."""
    if isinstance(error, urllib.error.HTTPError):
        return None
    if not isinstance(error, (urllib.error.URLError, ConnectionError, TimeoutError,
                              socket.gaierror, ssl.SSLError)):
        return None
    reason = error
    for _ in range(4):
        if not isinstance(reason, urllib.error.URLError):
            break
        reason = reason.reason
    category = 'network_error'
    if isinstance(reason, ssl.SSLCertVerificationError):
        category = 'tls_certificate_verification'
    elif isinstance(reason, ssl.SSLEOFError):
        category = 'tls_eof'
    elif isinstance(reason, ssl.SSLError):
        category = 'tls_error'
    elif isinstance(reason, socket.gaierror):
        category = 'dns_temporary' if reason.errno == socket.EAI_AGAIN else 'dns_resolution'
    elif isinstance(reason, ConnectionRefusedError):
        category = 'connection_refused'
    elif isinstance(reason, TimeoutError):
        category = 'timeout'
    elif isinstance(reason, ConnectionResetError):
        category = 'connection_reset'
    elif isinstance(reason, str) and 'tunnel connection failed' in reason.lower():
        category = 'proxy_tunnel_failure'
    details = {'category': category, 'reason_type': type(reason).__name__}
    for attribute in ('errno', 'verify_code'):
        value = getattr(reason, attribute, None)
        if isinstance(value, int):
            details[attribute] = value
    tls_reason = getattr(reason, 'reason', None)
    if isinstance(tls_reason, str) and re.fullmatch(r'[A-Z0-9_]{1,80}', tls_reason):
        details['tls_reason'] = tls_reason
    return details


COMPACTION_MAX_OUTPUT = 16384
COMPACTION_INSTRUCTIONS = (
    'Perform context compaction only. Do not continue the task or call tools. '
    'Write a concise, self-contained handoff, targeting 2000 tokens and no more than 3000 tokens. '
    'Prioritize the latest user objective and constraints, current file ownership, exact working paths '
    'and revisions, verified changes and test outcomes, unfinished work, blockers, and the next actions. '
    'Preserve unique identifiers and facts needed to continue, but combine repeated facts and summarize '
    'completed phases briefly. Omit raw logs, source code, full diffs, repeated commands, and obsolete plans. '
    'Distinguish verified facts from guesses. Return only the visible handoff summary; do not reveal private reasoning.'
)
COMPACTION_RETRY_INSTRUCTIONS = (
    'The previous handoff did not finish within its output budget or returned no visible summary. '
    'Start a new, shorter handoff from the original history above. Target 1200 tokens, at most 2000 tokens. '
    'Keep the active objective, binding constraints, current paths/revisions, verified state and next actions. '
    'Compress older completed work to brief references. Do not repeat the previous draft or continue the task. '
    'Return a complete visible summary only, without tools, raw logs, code, or private reasoning.'
)


class ContextStore:
    """Private local summaries addressed by random, model-bound opaque tokens."""
    prefix = 'dsctx_v1_'

    def __init__(self, directory):
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink() or self.directory.stat().st_mode & 0o077:
            raise ProtocolError('Compaction context directory must be private (0700)')

    def put(self, model, summary):
        token = self.prefix + secrets.token_urlsafe(32)
        path = self.directory / (hashlib.sha256(token.encode()).hexdigest() + '.json')
        value = json.dumps({'model': model, 'summary': summary, 'created_at': int(time.time())}).encode()
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as f:
            f.write(value)
        return token

    def get(self, token, model):
        if not isinstance(token, str) or not re.fullmatch(r'dsctx_v1_[A-Za-z0-9_-]{43}', token):
            raise ProtocolError('Foreign or invalid compaction state; use a fresh DeepSeek agent')
        path = self.directory / (hashlib.sha256(token.encode()).hexdigest() + '.json')
        try:
            if path.is_symlink() or path.stat().st_mode & 0o077 or path.stat().st_size > 1024 * 1024:
                raise ProtocolError('Compaction context file is not private or is invalid')
            value = json.loads(path.read_text())
        except FileNotFoundError:
            raise ProtocolError('Local DeepSeek compaction state is missing; restore contexts or start a fresh agent') from None
        except (ValueError, OSError):
            raise ProtocolError('Local DeepSeek compaction state is unreadable') from None
        if value.get('model') != model or not isinstance(value.get('summary'), str) or not value['summary'].strip():
            raise ProtocolError('Compaction state model mismatch or empty summary')
        return value['summary']


def prepare_context(body, store, model):
    """Resolve provider-owned summaries and detect a native V2 compact request."""
    if isinstance(body.get('input'), str):
        return body, False
    result = dict(body)
    items, compact = [], False
    for item in body.get('input', []):
        kind = item.get('type', 'message')
        if kind == 'compaction_trigger':
            if compact:
                raise ProtocolError('Multiple compaction triggers are unsupported')
            compact = True
        elif kind in ('compaction', 'context_compaction'):
            if store is None:
                raise ProtocolError('A persistent context store is required for DeepSeek compaction')
            summary = store.get(item.get('encrypted_content'), model)
            items.append({'type': 'message', 'role': 'user',
                          'content': 'Context summary retained by the local DeepSeek bridge:\n' + summary})
        else:
            items.append(item)
    result['input'] = items
    if compact:
        if store is None:
            raise ProtocolError('A context store is required for DeepSeek compaction')
        result['tools'] = []
        result['tool_choice'] = 'none'
        items.append({'type': 'message', 'role': 'user', 'content': COMPACTION_INSTRUCTIONS})
    return result, compact


def compact_events(upstream, model, mapping, store, retry=None, audit=None):
    """Adapt Chat summarization to native Responses V2 compaction events."""
    identity = 'resp_' + uuid.uuid4().hex
    response = {'id': identity, 'object': 'response', 'created_at': int(time.time()),
                'status': 'in_progress', 'model': model, 'output': [],
                'usage': None, 'error': None, 'incomplete_details': None}
    yield {'type': 'response.created', 'sequence_number': 0, 'response': dict(response)}
    yield {'type': 'response.in_progress', 'sequence_number': 1, 'response': dict(response)}
    last_heartbeat = time.monotonic()

    def collect(stream, attempt):
        nonlocal last_heartbeat
        final = None
        for event in relay_events(stream, model, mapping):
            if time.monotonic() - last_heartbeat >= 10:
                yield {'type': 'keepalive'}
                last_heartbeat = time.monotonic()
            if event['type'] in ('response.completed', 'response.incomplete'):
                final = event['response']
        if final and audit:
            audit(route='deepseek_compaction_attempt', model=model, attempt=attempt,
                  status=final['status'], usage=final.get('usage'),
                  incomplete_reason=(final.get('incomplete_details') or {}).get('reason'),
                  summary_characters=sum(len(text_content(i.get('content')))
                                         for i in final['output'] if i['type'] == 'message'))
        return final

    final = yield from collect(upstream, 1)
    empty = final and not any(
        text_content(i.get('content')).strip() for i in final['output'] if i['type'] == 'message')
    if retry and final and (final['status'] == 'incomplete' or empty):
        # No partial draft is persisted or sent to Codex. Retry once against
        # the same original history, with a shorter handoff instruction.
        yield {'type': 'keepalive'}
        with retry() as repeated:
            final = yield from collect(repeated, 2)
    if not final or final['status'] != 'completed':
        raise CompactionError('DeepSeek compaction exceeded its summary output limit; original context was not compacted')
    if any(i['type'] not in ('message', 'reasoning') for i in final['output']):
        raise CompactionError('DeepSeek compaction attempted a tool call; original context was not compacted')
    summary = '\n'.join(text_content(i.get('content')) for i in final['output'] if i['type'] == 'message').strip()
    if not summary:
        raise CompactionError('DeepSeek compaction returned no visible summary; original context was not compacted')
    token = store.put(model, summary)
    item = {'type': 'compaction', 'id': 'cmp_' + uuid.uuid4().hex, 'encrypted_content': token}
    yield {'type': 'response.output_item.added', 'sequence_number': 2, 'output_index': 0, 'item': dict(item)}
    yield {'type': 'response.output_item.done', 'sequence_number': 3, 'output_index': 0, 'item': dict(item)}
    response.update(status='completed', output=[item], usage=final.get('usage'))
    yield {'type': 'response.completed', 'sequence_number': 4, 'response': response}


def text_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    parts = []
    for part in content:
        if part.get('type') in ('input_text', 'output_text', 'text', 'summary_text', 'reasoning_text'):
            parts.append(part.get('text', ''))
        else:
            raise ProtocolError('This bridge currently accepts text tool results and text input only')
    return '\n'.join(parts)


def shape(value, depth=0):
    """Diagnostic schema only; excludes prompt, tool argument and credential values."""
    if depth > 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {k: (v if k == 'type' and isinstance(v, str) else shape(v, depth + 1)) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(v, depth + 1) for v in value[:4]]
    return type(value).__name__


def tool_wire_names(identities):
    """Keep ordinary names; deterministically alias long or ambiguous names."""
    prefix = 'dsbridge_'
    candidates, groups = {}, {}
    for namespace, name in identities:
        if not isinstance(name, str) or not name:
            raise ProtocolError('Responses tool name must be a non-empty string')
        if namespace is not None and (not isinstance(namespace, str) or not namespace):
            raise ProtocolError('Responses tool namespace must be a non-empty string')
        identity = (namespace, name)
        wire = ((namespace + '__') if namespace else '') + name
        candidates[identity] = wire
        groups.setdefault(wire, set()).add(identity)
    names, allocated = {}, {}
    for identity, candidate in candidates.items():
        wire = candidate
        if (not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', wire)
                or wire.startswith(prefix) or len(groups[wire]) > 1):
            # Hash the structured identity, not the flattened name: two
            # namespaces can flatten to the same string. Reserve the prefix
            # so a real tool named like an alias cannot shadow another tool.
            digest = hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).hexdigest()[:32]
            readable = re.sub(r'[^A-Za-z0-9_-]', '_', candidate)[:22]
            wire = prefix + readable + '_' + digest
        if wire in allocated and allocated[wire] != identity:
            raise ProtocolError('Chat tool alias collision')
        names[identity] = wire
        allocated[wire] = identity
    return names


def tool_definitions(tools, extra_identities=()):
    result, mapping, definitions, seen = [], {}, [], set()

    def add(tool, namespace=None):
        kind = tool.get('type')
        # Codex can inherit hosted-search definitions from the parent even when
        # this model's catalog declares supports_search_tool=false. They are not
        # executable by a Chat relay, so do not advertise them to this model.
        if kind in ('web_search', 'web_search_preview'):
            return
        if kind == 'namespace':
            for child in tool.get('tools', []):
                add(child, tool['name'])
            return
        if kind not in ('function', 'custom'):
            raise ProtocolError('Unsupported Responses tool type: ' + str(kind))
        name = tool['name']
        identity = (namespace, name)
        # Validate before using the identity in a set.
        tool_wire_names([identity])
        if identity in seen:
            raise ProtocolError('Duplicate Responses tool definition: ' +
                                ((namespace + '.') if namespace else '') + name)
        seen.add(identity)
        definitions.append((tool, namespace))

    for tool in tools:
        add(tool)
    names = tool_wire_names(list(seen) + list(extra_identities))
    for tool, namespace in definitions:
        name, kind = tool['name'], tool['type']
        wire = names[(namespace, name)]
        meta = {'name': name, 'type': kind}
        if namespace:
            meta['namespace'] = namespace
        mapping[wire] = meta
        params = tool.get('parameters', {'type': 'object', 'properties': {}})
        if kind == 'custom':
            params = {'type': 'object', 'properties': {'input': {'type': 'string'}},
                      'required': ['input'], 'additionalProperties': False}
        result.append({'type': 'function', 'function': {
            'name': wire, 'description': tool.get('description', ''), 'parameters': params}})

    return result, mapping


def to_chat(body, upstream_model, max_output):
    if body.get('previous_response_id'):
        raise ProtocolError('Stateful continuation is unsupported; use full HTTP input replay')
    items = body.get('input', [])
    if isinstance(items, str):
        items = [{'type': 'message', 'role': 'user', 'content': items}]
    # Historical calls still need valid names when a later request omits their
    # definitions (including a compaction request with tools disabled).
    identities = [(i.get('namespace'), i['name']) for i in items
                  if i.get('type') in ('function_call', 'custom_tool_call')]
    choice = body.get('tool_choice', 'auto')
    if isinstance(choice, dict):
        identities.append((choice.get('namespace'), choice['name']))
    tools, mapping = tool_definitions(body.get('tools', []), identities)
    reverse = tool_wire_names([(v.get('namespace'), v['name']) for v in mapping.values()] + identities)
    messages, pending_reasoning = [], ''

    def assistant_message(content='', reasoning=''):
        # A Responses turn can contain reasoning, commentary and several calls
        # as separate items. Chat requires them in one assistant message,
        # including reasoning_content when returning thinking-mode tool calls.
        if not messages or messages[-1]['role'] != 'assistant':
            messages.append({'role': 'assistant', 'content': None})
        message = messages[-1]
        if content:
            message['content'] = ((message.get('content') or '') + '\n' + content).lstrip('\n')
        if reasoning:
            message['reasoning_content'] = message.get('reasoning_content', '') + reasoning
        return message

    if body.get('instructions'):
        messages.append({'role': 'system', 'content': body['instructions']})
    for item in items:
        kind = item.get('type', 'message')
        if kind == 'message':
            role = item['role']
            if role == 'assistant':
                assistant_message(text_content(item.get('content')), pending_reasoning)
                pending_reasoning = ''
                continue
            message = {'role': 'system' if role == 'developer' else role,
                       'content': text_content(item.get('content'))}
            messages.append(message)
        elif kind == 'agent_message':
            parts = item.get('content', [])
            # In v2 the visible text may be only an empty "Payload:" envelope;
            # the real task can live entirely in the encrypted portion. Never
            # silently turn that envelope into a task for another provider.
            if any(p.get('type') == 'encrypted_content' for p in parts):
                raise ProtocolError('Encrypted native v2 agent messages require local v1 orchestration')
            readable = [p for p in parts if p.get('type') in ('input_text', 'output_text', 'text')]
            if not readable:
                raise ProtocolError('Native agent message has no portable text')
            unknown = [p.get('type') for p in parts if p.get('type') not in ('input_text', 'output_text', 'text')]
            if unknown:
                raise ProtocolError('Unsupported native agent message content')
            messages.append({'role': 'user', 'content': '[Agent message from ' + str(item.get('author', 'parent')) + ']\n' + text_content(readable)})
        elif kind == 'reasoning':
            pending_reasoning += text_content(item.get('content') or item.get('summary') or [])
        elif kind in ('function_call', 'custom_tool_call'):
            wire = reverse.get((item.get('namespace'), item['name']), item['name'])
            args = item.get('arguments', '{}') if kind == 'function_call' else json.dumps({'input': item.get('input', '')})
            message = assistant_message(reasoning=pending_reasoning)
            pending_reasoning = ''
            message.setdefault('tool_calls', []).append(
                {'id': item['call_id'], 'type': 'function',
                 'function': {'name': wire, 'arguments': args}})
        elif kind in ('function_call_output', 'custom_tool_call_output'):
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'],
                             'content': text_content(item.get('output'))})
        else:
            raise ProtocolError('Unsupported Responses input item: ' + str(kind))
    payload = {'model': upstream_model, 'messages': messages, 'stream': True,
               'stream_options': {'include_usage': True},
               'max_tokens': min(body.get('max_output_tokens') or max_output, max_output)}
    reasoning = body.get('reasoning') or {}
    if not isinstance(reasoning, dict):
        raise ProtocolError('reasoning must be an object')
    effort = reasoning.get('effort') or 'high'
    if effort not in ('low', 'high', 'max'):
        raise ProtocolError('DeepSeek reasoning effort must be low, high or max')
    payload['thinking'] = {'type': 'enabled'}
    payload['reasoning_effort'] = effort
    if tools:
        payload['tools'] = tools
        choice = body.get('tool_choice', 'auto')
        if isinstance(choice, dict):
            name = reverse.get((choice.get('namespace'), choice['name']), choice['name'])
            choice = {'type': 'function', 'function': {'name': name}}
        payload['tool_choice'] = choice
    for key in ('temperature', 'top_p'):
        if key in body:
            payload[key] = body[key]
    # GPT-specific tier and Responses-Lite parameters are never forwarded.
    return payload, mapping


def convert_usage(usage):
    converted = {
        'input_tokens': usage.get('prompt_tokens', 0),
        'output_tokens': usage.get('completion_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0),
    }
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    if cached is None:
        cached = usage.get('prompt_cache_hit_tokens')
    if cached is not None:
        converted['input_tokens_details'] = {'cached_tokens': cached}
    reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    if reasoning is not None:
        converted['output_tokens_details'] = {'reasoning_tokens': reasoning}
    return converted


def sse_records(upstream):
    data = []
    for raw in upstream:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                payload = '\n'.join(data)
                data = []
                if payload == '[DONE]':
                    return
                yield json.loads(payload)
        elif line.startswith('data:'):
            data.append(line[5:].lstrip())
    if data and '\n'.join(data) != '[DONE]':
        yield json.loads('\n'.join(data))


def relay_events(upstream, model, mapping):
    response = {'id': 'resp_' + uuid.uuid4().hex, 'object': 'response',
                'created_at': int(time.time()), 'status': 'in_progress', 'model': model,
                'output': [], 'usage': None, 'error': None, 'incomplete_details': None}
    seq, slots, output = 0, {}, []

    def event(kind, **kwargs):
        nonlocal seq
        d = {'type': kind, 'sequence_number': seq, **kwargs}
        seq += 1
        return d

    yield event('response.created', response=dict(response))
    yield event('response.in_progress', response=dict(response))
    finish = None
    for chunk in sse_records(upstream):
        if chunk.get('error'):
            raise ProtocolError('The upstream reported a streaming error')
        if chunk.get('usage'):
            response['usage'] = convert_usage(chunk['usage'])
        for choice in chunk.get('choices', []):
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
            delta = choice.get('delta', {})
            for field, kind in [('reasoning_content', 'reasoning'), ('content', 'message')]:
                value = delta.get(field)
                if not value:
                    continue
                if field not in slots:
                    index = len(output)
                    item = ({'id': 'rs_' + uuid.uuid4().hex, 'type': 'reasoning', 'summary': [],
                             'content': [{'type': 'reasoning_text', 'text': ''}]} if kind == 'reasoning' else
                            {'id': 'msg_' + uuid.uuid4().hex, 'type': 'message', 'role': 'assistant',
                             'status': 'in_progress', 'content': [{'type': 'output_text', 'text': '', 'annotations': []}]})
                    slots[field] = index
                    output.append(item)
                    yield event('response.output_item.added', output_index=index, item=json.loads(json.dumps(item)))
                    if kind == 'message':
                        yield event('response.content_part.added', item_id=item['id'], output_index=index,
                                    content_index=0, part={'type': 'output_text', 'text': '', 'annotations': []})
                index = slots[field]
                item = output[index]
                item['content'][0]['text'] += value
                yield event('response.reasoning_text.delta' if kind == 'reasoning' else 'response.output_text.delta',
                            item_id=item['id'], output_index=index, content_index=0, delta=value)
            for tc in delta.get('tool_calls', []):
                key = 'tool_' + str(tc['index'])
                fn = tc.get('function', {})
                if key not in slots:
                    if not fn.get('name'):
                        raise ProtocolError('Tool stream has no initial name')
                    meta = mapping.get(fn['name'])
                    if not meta:
                        raise ProtocolError('The upstream selected an undeclared tool')
                    index = len(output)
                    item = {'id': 'fc_' + uuid.uuid4().hex, 'type': 'custom_tool_call' if meta['type'] == 'custom' else 'function_call',
                            'call_id': tc.get('id') or 'call_' + uuid.uuid4().hex,
                            'name': meta['name'], 'status': 'in_progress'}
                    if meta.get('namespace'):
                        item['namespace'] = meta['namespace']
                    item['input' if meta['type'] == 'custom' else 'arguments'] = ''
                    slots[key] = {'index': index, 'raw': '', 'custom': meta['type'] == 'custom'}
                    output.append(item)
                    yield event('response.output_item.added', output_index=index, item=dict(item))
                slot = slots[key]
                index, arg = slot['index'], fn.get('arguments') or ''
                item = output[index]
                slot['raw'] += arg
                if arg and not slot['custom']:
                    item['arguments'] += arg
                    yield event('response.function_call_arguments.delta', item_id=item['id'], output_index=index, delta=arg)
    if finish not in ('stop', 'tool_calls', 'length'):
        raise ProtocolError('Upstream stream ended without a successful finish reason')
    for key, slot in slots.items():
        if key.startswith('tool_') and slot['custom']:
            item = output[slot['index']]
            value = json.loads(slot['raw']).get('input')
            if not isinstance(value, str):
                raise ProtocolError('Custom tool input is not a string')
            item['input'] = value
            yield event('response.custom_tool_call_input.delta', item_id=item['id'], output_index=slot['index'], delta=value)
    for index, item in enumerate(output):
        kind = item['type']
        if kind == 'message':
            item['phase'] = 'commentary' if finish == 'tool_calls' else 'final_answer'
            yield event('response.output_text.done', item_id=item['id'], output_index=index, content_index=0, text=item['content'][0]['text'])
            yield event('response.content_part.done', item_id=item['id'], output_index=index, content_index=0, part=item['content'][0])
            item['status'] = 'completed'
        elif kind == 'reasoning':
            yield event('response.reasoning_text.done', item_id=item['id'], output_index=index, content_index=0, text=item['content'][0]['text'])
        else:
            item['status'] = 'completed'
            field = 'input' if kind == 'custom_tool_call' else 'arguments'
            yield event('response.custom_tool_call_input.done' if field == 'input' else 'response.function_call_arguments.done',
                        item_id=item['id'], output_index=index, **{field: item[field]})
        yield event('response.output_item.done', output_index=index, item=item)
    response['output'], response['status'] = output, 'incomplete' if finish == 'length' else 'completed'
    if finish == 'length':
        response['incomplete_details'] = {'reason': 'max_output_tokens'}
    yield event('response.incomplete' if finish == 'length' else 'response.completed', response=response)


class Gateway(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        self.activity_lock = threading.Lock()
        self.active_connections = 0
        self.last_activity = time.monotonic()
        super().__init__(*args, **kwargs)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def handle(self):
        with self.server.activity_lock:
            self.server.active_connections += 1
            self.server.last_activity = time.monotonic()
        try:
            super().handle()
        finally:
            with self.server.activity_lock:
                self.server.active_connections -= 1
                self.server.last_activity = time.monotonic()

    def log_message(self, *_):
        pass

    def audit(self, **fields):
        line = json.dumps({'time': time.time(), **fields}, ensure_ascii=False)
        with self.server.audit_lock:
            if self.server.audit_path:
                with open(self.server.audit_path, 'a') as f:
                    f.write(line + '\n')

    def json_response(self, code, value):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)

    def open_upstream(self, request, timeout):
        for attempt in range(2):
            try:
                return self.server.opener.open(request, timeout=timeout)
            except Exception as exc:
                details = network_error_details(exc)
                # These two failures establish no connection, so no request
                # body can have reached the provider. Never retry TLS errors,
                # write/reset/timeouts or an already-started response here.
                if attempt or not details or details['category'] not in (
                        'dns_temporary', 'connection_refused'):
                    raise
                self.audit(route='connect_retry', attempt=1, network_error=details)
                time.sleep(0.25)

    def do_GET(self):
        if self.path == '/health':
            return self.json_response(200, {'status': 'ok', 'deepseek_model': self.server.model,
                                           'deepseek_inline_compaction': True})
        if self.path.split('?')[0] == '/models':
            if not self.authorized():
                return
            return self.json_response(200, {'models': self.server.catalog['models']})
        self.json_response(404, {'error': {'message': 'Unknown gateway endpoint'}})

    def do_POST(self):
        if not self.authorized():
            return
        begun = time.monotonic()
        streamed = False
        stream_response, stream_sequence = None, -1
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 32 * 1024 * 1024:
                raise ProtocolError('Invalid request size')
            raw = self.rfile.read(size)
            body = json.loads(raw)
            model = body.get('model')
            if self.path.split('?')[0] not in ('/responses', '/responses/compact'):
                return self.json_response(404, {'error': {'message': 'Unknown gateway endpoint'}})
            if model == self.server.model:
                if self.path.split('?')[0] != '/responses':
                    raise ProtocolError('DeepSeek server-side compaction is unsupported')
                prepared, compact = prepare_context(body, getattr(self.server, 'context_store', None), self.server.model)
                chat, mapping = to_chat(prepared, self.server.model, self.server.max_output)
                if compact:
                    # Summary generation has its own budget; changing it must
                    # not enlarge ordinary task replies or their user limits.
                    chat['max_tokens'] = getattr(self.server, 'max_compaction_output', COMPACTION_MAX_OUTPUT)
                # Construct relay headers from scratch. ChatGPT auth/identity never crosses this branch.
                headers = {'Authorization': 'Bearer ' + self.server.api_key,
                           'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                           'User-Agent': 'Codex-DeepSeek-Bridge/0.1'}
                req = urllib.request.Request(self.server.relay + '/chat/completions',
                                             data=json.dumps(chat).encode(), headers=headers)
                self.audit(route='deepseek', model=model, input_items=len(body.get('input', [])),
                           tool_count=len(mapping), tool_names=list(mapping),
                           reasoning_effort_forwarded=chat['reasoning_effort'],
                           thinking_type=chat['thinking']['type'],
                           compaction_request=compact,
                           max_output_tokens=chat['max_tokens'],
                           excluded_hosted_tools=[t['type'] for t in body.get('tools', []) if t.get('type') in ('web_search', 'web_search_preview')],
                           upstream='chat/completions', chatgpt_headers_forwarded=False)

                def retry_summary():
                    retry_chat = dict(chat)
                    retry_chat['messages'] = chat['messages'][:-1] + [
                        {'role': 'user', 'content': COMPACTION_RETRY_INSTRUCTIONS}]
                    repeated = urllib.request.Request(self.server.relay + '/chat/completions',
                        data=json.dumps(retry_chat).encode(), headers=headers)
                    return self.open_upstream(repeated, timeout=60)

                with self.open_upstream(req, timeout=60) as upstream:
                    events = (compact_events(upstream, model, mapping, self.server.context_store,
                                             retry=retry_summary, audit=self.audit) if compact
                              else relay_events(upstream, model, mapping))
                    if body.get('stream'):
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/event-stream')
                        self.send_header('Cache-Control', 'no-cache')
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        self.close_connection, streamed = True, True
                        for event in events:
                            if event.get('response'):
                                stream_response = event['response']
                            stream_sequence = max(stream_sequence, event.get('sequence_number', -1))
                            data = ('event: ' + event['type'] + '\ndata: ' + json.dumps(event, ensure_ascii=False) + '\n\n').encode()
                            self.wfile.write(data)
                            self.wfile.flush()
                            if event['type'] in ('response.completed', 'response.incomplete'):
                                self.audit(route='deepseek_completed', model=model,
                                           output_types=[i['type'] for i in event['response']['output']],
                                           usage=event['response'].get('usage'),
                                           seconds=round(time.monotonic()-begun, 3))
                    else:
                        final = None
                        for event in events:
                            if event['type'] in ('response.completed', 'response.incomplete'):
                                final = event['response']
                        self.json_response(200, final)
            elif model in self.server.gpt_models:
                if isinstance(body.get('input'), list) and any(
                    isinstance(i.get('encrypted_content'), str) and i['encrypted_content'].startswith(ContextStore.prefix)
                    for i in body['input']):
                    raise ProtocolError('DeepSeek compaction state cannot be sent to GPT; use a fresh agent')
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and not k.lower().startswith('x-bridge-')}
                headers['Accept-Encoding'] = 'identity'
                req = urllib.request.Request('https://chatgpt.com/backend-api/codex' + self.path, data=raw, headers=headers)
                self.audit(route='openai', model=model, path=self.path,
                           tool_types=[t.get('type') for t in body.get('tools', [])],
                           tool_names=[t.get('name') for t in body.get('tools', [])],
                           reasoning_effort=(body.get('reasoning') or {}).get('effort'),
                           multi_agent_schema={k: shape(v) for k, v in body.items() if 'agent' in k},
                           request_field_names=list(body))
                with self.open_upstream(req, timeout=180) as upstream:
                    self.send_response(upstream.status)
                    for k, v in upstream.headers.items():
                        if k.lower() not in HOP:
                            self.send_header(k, v)
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection, streamed = True, True
                    while data := upstream.read1(16384):
                        self.wfile.write(data)
                        self.wfile.flush()
                self.audit(route='openai_completed', model=model, seconds=round(time.monotonic()-begun, 3))
            else:
                self.json_response(400, {'error': {'message': 'Unknown model route', 'type': 'invalid_request_error'}})
        except (BrokenPipeError, ConnectionResetError):
            self.audit(route='client_disconnected')
        except Exception as exc:
            code = (exc.code if isinstance(exc, urllib.error.HTTPError) else
                    502 if isinstance(exc, CompactionError) else
                    400 if isinstance(exc, (ProtocolError, ValueError)) else 502)
            # Never log upstream response bodies, prompts, headers, or exception URLs/tokens.
            message = str(exc) if isinstance(exc, ProtocolError) else 'Upstream HTTP ' + str(code) if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
            network = network_error_details(exc)
            if network:
                message = 'Upstream network failure: ' + network['category'] + ' (' + network['reason_type'] + ')'
            upstream_code = None
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    error = json.loads(exc.read(8192)).get('error', {})
                    # Preserve useful fixed error codes, not arbitrary upstream bodies.
                    candidate = error.get('code')
                    if isinstance(candidate, str) and candidate.replace('_', '').isalnum() and len(candidate) <= 80:
                        upstream_code = candidate
                        message += ' (' + candidate + ')'
                except (ValueError, AttributeError):
                    pass
                finally:
                    exc.close()
            self.audit(route='error', status=code, error=message, upstream_code=upstream_code,
                       network_error=network)
            if isinstance(exc, ProtocolError) and 'Unsupported Responses input item' in message:
                self.audit(route='unsupported_input_schema', items=[shape(i) for i in body.get('input', []) if i.get('type') not in ('message', 'reasoning', 'function_call', 'function_call_output', 'custom_tool_call', 'custom_tool_call_output')])
            if streamed:
                event = {'type': 'error', 'message': message, 'code': 'bridge_error'}
                if stream_response is not None:
                    # Codex only interprets a few special top-level "error"
                    # events. A native failure terminal preserves the actual
                    # cause instead of becoming "stream closed before ...".
                    failed = dict(stream_response)
                    failed.update(status='failed', output=[],
                                  error={'code': 'bridge_error', 'message': message})
                    event = {'type': 'response.failed', 'sequence_number': stream_sequence + 1,
                             'response': failed}
                try:
                    self.wfile.write(('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n').encode())
                    self.wfile.flush()
                except OSError:
                    pass
            else:
                self.json_response(code, {'error': {'message': message, 'type': 'bridge_error'}})

    def authorized(self):
        if hmac.compare_digest(self.headers.get('X-Bridge-Key', ''), self.server.token):
            return True
        self.close_connection = True
        self.json_response(401, {'error': {'message': 'Invalid local gateway capability'}})
        return False


def serve(args):
    key = os.environ.pop('DEEPSEEK_API_KEY', '')
    if not key:
        raise SystemExit('DEEPSEEK_API_KEY must be supplied to the gateway process')
    parts = urllib.parse.urlsplit(args.relay)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise SystemExit('Relay must be an HTTPS base URL without credentials or query parameters')
    with open(args.catalog) as catalog_file:
        catalog = json.load(catalog_file)
    if getattr(args, 'systemd_socket', False):
        if os.environ.pop('LISTEN_PID', '') != str(os.getpid()) or os.environ.pop('LISTEN_FDS', '') != '1':
            raise SystemExit('Expected one socket from systemd for this process')
        server = Gateway(('127.0.0.1', 0), Handler, bind_and_activate=False)
        server.socket.close()
        server.socket = socket.socket(fileno=3)
        server.server_address = server.socket.getsockname()
        if server.server_address[0] != '127.0.0.1':
            server.server_close()
            raise SystemExit('systemd socket must listen on IPv4 loopback')
        server.server_name, server.server_port = server.server_address
    elif args.launchd_socket:
        library = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        activate = library.launch_activate_socket
        activate.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), ctypes.POINTER(ctypes.c_size_t)]
        activate.restype = ctypes.c_int
        fds = ctypes.POINTER(ctypes.c_int)()
        count = ctypes.c_size_t()
        status = activate(args.launchd_socket.encode(), ctypes.byref(fds), ctypes.byref(count))
        if status or count.value != 1:
            raise SystemExit('Could not activate the single launchd loopback socket: ' + str(status))
        fd = fds[0]
        library.free.argtypes = [ctypes.c_void_p]
        library.free(fds)
        server = Gateway(('127.0.0.1', 0), Handler, bind_and_activate=False)
        server.socket.close()
        server.socket = socket.socket(fileno=fd)
        server.server_address = server.socket.getsockname()
        server.server_name, server.server_port = server.server_address
    else:
        server = Gateway(('127.0.0.1', args.port), Handler)
    server.api_key, server.relay, server.model = key, args.relay.rstrip('/'), args.model
    server.token = os.environ.pop('CODEX_BRIDGE_TOKEN', '')
    if not server.token:
        server.server_close()
        raise SystemExit('CODEX_BRIDGE_TOKEN must be supplied to the gateway')
    server.catalog, server.max_output = catalog, args.max_output
    server.max_compaction_output = getattr(args, 'max_compaction_output', COMPACTION_MAX_OUTPUT)
    server.gpt_models = {m['slug'] for m in catalog['models'] if m['slug'].startswith('gpt-') or m['slug'] == 'codex-auto-review'}
    server.audit_path, server.audit_lock = args.audit, threading.Lock()
    temporary_contexts = None
    context_path = getattr(args, 'context_store', None)
    if context_path is None:
        temporary_contexts = tempfile.TemporaryDirectory(prefix='codex-deepseek-contexts-')
        context_path = temporary_contexts.name
    server.context_store = ContextStore(context_path)
    # No redirects: a gateway must never forward credentials to a redirected host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *unused):
            return None
    server.opener = urllib.request.build_opener(NoRedirect())
    if args.follow_pid:
        def follow_parent():
            while os.getppid() == args.follow_pid:
                time.sleep(1)
            server.shutdown()
        threading.Thread(target=follow_parent, daemon=True).start()
    if args.idle_timeout:
        def retire_idle_worker():
            while True:
                time.sleep(min(5, args.idle_timeout))
                with server.activity_lock:
                    idle = server.active_connections == 0 and time.monotonic() - server.last_activity >= args.idle_timeout
                if idle:
                    server.shutdown()
                    return
        threading.Thread(target=retire_idle_worker, daemon=True).start()
    print(json.dumps({'status': 'ready', 'host': '127.0.0.1', 'port': server.server_port}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if temporary_contexts is not None:
            temporary_contexts.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=24589)
    parser.add_argument('--relay', required=True, help='HTTPS Chat Completions base URL, normally ending in /v1')
    parser.add_argument('--model', default='deepseek-v4.1-flash')
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--audit')
    parser.add_argument('--context-store', help='Private persistent directory for opaque DeepSeek compaction summaries')
    parser.add_argument('--max-output', type=int, default=8192)
    parser.add_argument('--max-compaction-output', type=int, default=COMPACTION_MAX_OUTPUT,
                        help='Independent output budget for summaries (ordinary task replies are unchanged)')
    parser.add_argument('--follow-pid', type=int, help='Exit when the launcher process disappears')
    parser.add_argument('--launchd-socket', help='macOS launchd socket entry name')
    parser.add_argument('--systemd-socket', action='store_true', help='Linux systemd socket activation')
    parser.add_argument('--idle-timeout', type=int, default=0, help='Exit after this many idle seconds')
    serve(parser.parse_args())
