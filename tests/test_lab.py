import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import agents
import research
import tools
from check_citations import check


def fixture():
    sources = [
        {'n':1,'id':'1','url':'https://arxiv.org/abs/1','title':'A','date':'2025-01-01','source':'arxiv'},
        {'n':2,'id':'2','url':'https://huggingface.co/papers/2','title':'B','date':'2025-01-02','source':'hf-search'},
        {'n':3,'id':'3','url':'https://example.org/c','title':'C','date':'2025-01-03','source':'web'}]
    report = '# Survey\n## TL;DR\nA [1]\n## Background\nB [2]\n## Theme A\nC [3]\n## Theme B\nA [1]\n## Theme C\nB [2]\n## Trends and open problems\nC [3]\n## References\n'
    report += '\n'.join(f"[{s['n']}] {s['title']}. {s['source']}. {s['url']} ({s['date']})" for s in sources)
    return report, sources

class Citations(unittest.TestCase):
    def test_valid_and_grouped(self):
        report,sources = fixture()
        self.assertEqual(check(report,sources),[])
        self.assertEqual(check(report.replace('A [1]','A [1-3]'),sources),[])
        self.assertEqual(check(report.replace('A [1]','A [1, 2]'),sources),[])

    def test_invalid(self):
        report,sources = fixture()
        for changed in [report.split('## References')[0],report.replace('[1]','[9]'),report.replace('A [1]','A'),
                        report+'\n[1] extra https://example.com',report.replace('[1] A.','[1] https://extra.org A.')]:
            with self.subTest(report=changed):
                self.assertTrue(check(changed,sources))
        self.assertTrue(check(report,sources+[sources[0]]))
        self.assertTrue(check(report,[]))

    def test_code_and_links_not_citations(self):
        report,sources = fixture()
        report = report.replace('A [1]','A').replace('## Background','```python\n[1]\n```\n[1](https://example.org)\n## Background')
        self.assertIn('source [1] never cited',check(report,sources))

class FinalizerIntegration(unittest.TestCase):
    def test_provided_finalizer_matches_validator(self):
        from finalize_citations import finalize
        report, sources = fixture()
        body = report.split('## References')[0].replace('A [1]', 'A [1, 2]')
        sources.append({**sources[0], 'n':4})
        normalized, normalized_sources, problems = finalize(body, sources)
        self.assertEqual(problems, [])
        self.assertEqual(check(normalized, normalized_sources), [])
        second, second_sources, problems = finalize(normalized, normalized_sources)
        self.assertEqual((second, second_sources), (normalized, normalized_sources))

class Retry(unittest.TestCase):
    @patch('tools.time.sleep')
    def test_success_and_retry_after(self,sleep):
        fn = unittest.mock.Mock(side_effect=[tools.RetryableError('busy',7),'ok'])
        self.assertEqual(tools.with_retry(fn),'ok')
        sleep.assert_called_once_with(7)

    @patch('tools.time.sleep')
    def test_exhaustion_and_nonretry(self,sleep):
        fn = unittest.mock.Mock(side_effect=tools.RetryableError('busy'))
        with self.assertRaises(tools.RetryableError):
            tools.with_retry(fn,attempts=3,cap=2)
        self.assertEqual(sleep.call_count,2)
        self.assertTrue(all(c.args[0]<=2 for c in sleep.call_args_list))
        sleep.reset_mock()
        with self.assertRaises(ValueError):
            tools.with_retry(unittest.mock.Mock(side_effect=ValueError('bad')))
        sleep.assert_not_called()

    @patch('tools.httpx.request')
    def test_http_errors(self,request):
        for code in (429,500,502,503,504):
            request.return_value=httpx.Response(code,headers={'Retry-After':'4'},request=httpx.Request('GET','https://example.org'))
            with self.assertRaises(tools.RetryableError) as ctx:
                tools._request('GET','https://example.org')
            self.assertEqual(ctx.exception.retry_after,4)
        request.side_effect=httpx.ConnectError('offline')
        with self.assertRaises(tools.RetryableError):
            tools._request('GET','https://example.org')

class Sources(unittest.TestCase):
    @patch('tools._request')
    def test_arxiv(self,request):
        request.return_value=SimpleNamespace(text='<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id><title> A  title </title><summary> Summary </summary><published>2025-01-01T00:00:00Z</published></entry></feed>')
        with patch('tools._arxiv_last',None):
            result=json.loads(tools.arxiv_search.invoke({'query':'world: "model" AND'}))
        self.assertEqual(result[0]['url'],'https://arxiv.org/abs/2501.00001')
        self.assertEqual(request.call_args.kwargs['params']['search_query'],'all:world AND all:model')
        request.reset_mock()
        self.assertEqual(tools.arxiv_search.invoke({'query':':: AND'}),'NO RESULTS')
        request.assert_not_called()

    @patch('tools.time.sleep')
    @patch('tools._request')
    def test_arxiv_spacing(self,request,sleep):
        request.return_value=SimpleNamespace(text='<feed/>')
        with patch('tools._arxiv_last',None),patch('tools.time.monotonic',side_effect=[10,11,13]):
            tools.arxiv_search.invoke({'query':'a'})
            tools.arxiv_search.invoke({'query':'b'})
        sleep.assert_called_once_with(2)

    @patch('tools._request')
    def test_hf(self,request):
        request.return_value.json.return_value=[{'paper':{'id':'1','title':'World model','summary':'long','ai_summary':'short','upvotes':3}}, {'paper':{}}]
        self.assertEqual(json.loads(tools.hf_search_papers.invoke({'query':'world'}))[0]['summary'],'short')
        self.assertEqual(tools.hf_daily_papers.invoke({'keyword':'missing'}),'NO RESULTS')
        request.side_effect=ValueError('bad data')
        self.assertTrue(tools.hf_search_papers.invoke({'query':'x'}).startswith('ERROR:'))

    @patch.dict('os.environ', {'EXA_API_KEY': 'test-key'})
    @patch('tools.time.sleep')
    @patch('tools._request')
    def test_exa_native_requests_and_retry(self, request, sleep):
        request.side_effect = [tools.RetryableError('busy', 2),
                               SimpleNamespace(json=lambda: {'results': [{'url': 'https://example.org', 'highlights': ['evidence']}]})]
        result = json.loads(tools.web_search.invoke({'query': 'coding agents', 'objective': 'Find implementation docs'}))
        self.assertEqual(result[0]['highlights'], ['evidence'])
        sleep.assert_called_once_with(2)
        call = request.call_args
        self.assertEqual(call.args, ('POST', 'https://api.exa.ai/search'))
        self.assertEqual(call.kwargs['headers']['x-api-key'], 'test-key')
        self.assertEqual(call.kwargs['json'], {'query': 'coding agents', 'objective': 'Find implementation docs', 'numResults': 5, 'contents': {'highlights': True}})

    @patch.dict('os.environ', {'EXA_API_KEY': 'test-key'})
    @patch('tools._request')
    def test_contents_status_and_text_budget(self, request):
        request.return_value.json.return_value = {'results': [{'url': 'https://example.org', 'text': 'x'*13000}], 'statuses': [{'status': 'success'}]}
        result = json.loads(tools.web_fetch.invoke({'url': 'https://example.org'}))
        self.assertEqual(len(result[0]['text']), 12000)
        self.assertEqual(request.call_args.kwargs['json'], {'urls': ['https://example.org'], 'text': {'maxCharacters': 12000}})
        request.return_value.json.return_value = {'results': [], 'statuses': [{'status': 'error', 'id': 'https://example.org'}]}
        self.assertIn('extraction failed', tools.web_fetch.invoke({'url': 'https://example.org'}))

    def test_missing_key_and_redaction(self):
        with patch.dict('os.environ', {'EXA_API_KEY': ''}), patch('tools._request') as request:
            self.assertIn('Set EXA_API_KEY', tools.web_search.invoke({'query': 'test'}))
            request.assert_not_called()
        with patch.dict('os.environ', {'EXA_API_KEY': 'private-example-key'}), patch('tools._exa', side_effect=ValueError('private-example-key')):
            self.assertNotIn('private-example-key', tools.web_fetch.invoke({'url': 'https://example.org'}))

class Runner(unittest.TestCase):
    def messages(self):
        return [{'tool_calls':[{'name':'task','args':{'subagent_type':'researcher'}}]*3,'usage_metadata':{'input_tokens':12,'output_tokens':4}},{}]

    def test_slug_and_summary(self):
        self.assertEqual(research.slugify('../../x'),'x')
        self.assertEqual(research.slugify(''),'topic')
        self.assertEqual(len(research.slugify('a'*100)),60)
        self.assertEqual(research.summarize(self.messages(),1.23,'test')['tokens'],{'input':12,'output':4})

    def test_save_exact_bytes_and_reject(self):
        report,sources=fixture()
        data={research.REPORT_PATH:report.encode(),research.SOURCES_PATH:json.dumps(sources).encode()}
        with tempfile.TemporaryDirectory(dir=research.ROOT) as folder,patch('research.download',return_value=data):
            result=research.save_outputs(None,'topic',self.messages(),1,'test',folder)
            self.assertEqual(result.read_bytes(),data[research.REPORT_PATH])
            self.assertEqual(result.with_suffix('.sources.json').read_bytes(),data[research.SOURCES_PATH])
        for bad in (None,b'',b'bad json'):
            with tempfile.TemporaryDirectory(dir=research.ROOT) as folder,patch('research.download',return_value={**data,research.SOURCES_PATH:bad}):
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None,'topic',self.messages(),1,'test',folder)
                self.assertEqual(list(Path(folder).iterdir()),[])

    def test_save_rollback(self):
        report,sources=fixture()
        data={research.REPORT_PATH:report.encode(),research.SOURCES_PATH:json.dumps(sources).encode()}
        real_replace=research.os.replace
        count=0
        def fail_second(src,dst):
            nonlocal count
            count+=1
            if count==2:
                raise OSError('simulated disk failure')
            return real_replace(src,dst)
        with tempfile.TemporaryDirectory(dir=research.ROOT) as folder,patch('research.download',return_value=data),patch('research.os.replace',side_effect=fail_second):
            prior=Path(folder)/'topic.md'
            prior.write_bytes(b'old report')
            with self.assertRaises(RuntimeError):
                research.save_outputs(None,'topic',self.messages(),1,'test',folder)
            self.assertEqual(prior.read_bytes(),b'old report')
            self.assertEqual(list(Path(folder).iterdir()),[prior])

    def test_main_failure_and_usage(self):
        self.assertEqual(research.main(''),2)
        with patch('research.make_model',side_effect=RuntimeError('not configured')):
            self.assertEqual(research.main('topic'),1)

    def test_main_sandbox_cleanup_on_failure(self):
        from contextlib import contextmanager
        events = []
        @contextmanager
        def sandbox():
            events.append('opened')
            try:
                yield SimpleNamespace(execute=lambda cmd: SimpleNamespace(exit_code=0))
            finally:
                events.append('closed')
        with patch('research.make_model'), patch('research.open_sandbox',sandbox), patch('research.upload',side_effect=RuntimeError('upload failed')):
            self.assertEqual(research.main('topic'),1)
        self.assertEqual(events,['opened','closed'])

    def test_agent_configuration(self):
        specs=agents.build_subagents()
        self.assertIsNot(specs[0]['middleware'][0],specs[1]['middleware'][0])
        with patch('agents.create_deep_agent') as create:
            agents.build_lead_agent('backend','model')
            self.assertEqual(create.call_args.kwargs['backend'],'backend')
            self.assertEqual(len(create.call_args.kwargs['middleware']),3)

if __name__=='__main__':
    unittest.main()
