"""Model-bound local summaries behind the native Responses compaction protocol."""
import io
import json
import pathlib
import tempfile
import unittest
from bridge import ContextStore, ProtocolError, compact_events, prepare_context, to_chat


def completion(text, reason='stop'):
    value={'choices':[{'delta':{'content':text},'finish_reason':reason}],
           'usage':{'prompt_tokens':200,'completion_tokens':25,'total_tokens':225}}
    return io.BytesIO(('data: '+json.dumps(value)+'\n\ndata: [DONE]\n\n').encode())


class CompactionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=ContextStore(pathlib.Path(self.temp.name)/'contexts')

    def tearDown(self):
        self.temp.cleanup()

    def test_native_compaction_roundtrip_keeps_summary_out_of_client_token(self):
        summary='Remember the verified marker violet-8427 and the remaining file task.'
        events=list(compact_events(completion(summary),'deepseek',{},self.store))
        final=events[-1]['response']
        self.assertEqual(events[-1]['type'],'response.completed')
        self.assertEqual(final['status'],'completed')
        item=final['output'][0]
        self.assertEqual(item['type'],'compaction')
        self.assertTrue(item['encrypted_content'].startswith('dsctx_v1_'))
        self.assertNotIn('violet-8427',json.dumps(final))
        replay,requested=prepare_context({'input':[item,{'role':'user','content':'Continue'}]},self.store,'deepseek')
        self.assertFalse(requested)
        chat,_=to_chat(replay,'deepseek',8192)
        self.assertIn(summary,chat['messages'][0]['content'])
        self.assertNotIn(item['encrypted_content'],json.dumps(chat))
        self.assertEqual(final['usage']['input_tokens'],200)

    def test_compaction_request_does_not_continue_task_or_advertise_tools(self):
        body={'input':[{'role':'user','content':'Task'},
                       {'type':'compaction_trigger'}],
              'tools':[{'type':'function','name':'exec_command'}]}
        prepared,compact=prepare_context(body,self.store,'ds')
        self.assertTrue(compact)
        chat,_=to_chat(prepared,'ds',8192)
        self.assertNotIn('tools',chat)
        self.assertIn('Do not continue the task',chat['messages'][-1]['content'])
        self.assertFalse(any(i.get('type')=='compaction_trigger' for i in prepared['input']))

    def test_private_store_survives_worker_restart_and_binds_model(self):
        token=self.store.put('ds','Persistent summary')
        reopened=ContextStore(self.store.directory)
        self.assertEqual(reopened.get(token,'ds'),'Persistent summary')
        self.assertEqual(self.store.directory.stat().st_mode & 0o777,0o700)
        self.assertEqual(next(self.store.directory.iterdir()).stat().st_mode & 0o777,0o600)
        with self.assertRaises(ProtocolError):
            reopened.get(token,'different-model')

    def test_missing_foreign_and_corrupt_contexts_fail_closed(self):
        for token in ['foreign-openai-state','../../auth.json','dsctx_v1_'+'a'*43,None]:
            with self.subTest(token=token),self.assertRaises(ProtocolError):
                self.store.get(token,'ds')
        token=self.store.put('ds','Summary')
        next(self.store.directory.iterdir()).write_text('not JSON')
        with self.assertRaises(ProtocolError):
            self.store.get(token,'ds')

    def test_truncated_or_empty_summary_does_not_replace_context(self):
        for text,reason in [('partial','length'),('','stop')]:
            with self.subTest(text=text),self.assertRaises(ProtocolError):
                list(compact_events(completion(text,reason),'ds',{},self.store))
        self.assertEqual(list(self.store.directory.iterdir()),[])

    def test_multiple_triggers_and_missing_store_are_rejected(self):
        for body,store in [
            ({'input':[{'type':'compaction_trigger'}]},None),
            ({'input':[{'type':'compaction_trigger'},{'type':'compaction_trigger'}]},self.store)]:
            with self.assertRaises(ProtocolError):
                prepare_context(body,store,'ds')


if __name__=='__main__':
    unittest.main()
