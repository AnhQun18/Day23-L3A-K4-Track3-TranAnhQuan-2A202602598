"""Host-only source tools. No credentials are passed to the sandbox."""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
ARXIV_URL = 'https://export.arxiv.org/api/query'
HF_DAILY_URL = 'https://huggingface.co/api/daily_papers'
HF_SEARCH_URL = 'https://huggingface.co/api/papers/search'
EXA_URL = 'https://api.exa.ai'
_arxiv_lock = threading.Lock()
_arxiv_last = None

class RetryableError(Exception):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after

def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    if attempts < 1 or base < 0 or cap < 0:
        raise ValueError('invalid retry configuration')
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            delay = exc.retry_after
            if delay is None:
                delay = base * 2**attempt + random.uniform(0, base)
            time.sleep(min(cap, max(0, float(delay))))

def _retry_after(value):
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError):
            return None

def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, timeout=60, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(str(exc)) from exc
    if response.status_code in {429, 500, 502, 503, 504}:
        raise RetryableError(f'HTTP {response.status_code}', _retry_after(response.headers.get('Retry-After')))
    response.raise_for_status()
    return response

def _error(exc):
    text = f'{type(exc).__name__}: {exc}'
    for name, value in os.environ.items():
        if value and any(word in name for word in ('API_KEY', 'SECRET', 'TOKEN', 'OPENAI_KEY')):
            text = text.replace(value, '[REDACTED]').replace(quote(value, safe=''), '[REDACTED]')
    text = re.sub(r'(exaApiKey=)[^\s&\"\']+', r'\1[REDACTED]', text, flags=re.I)
    return 'ERROR: ' + text

def _compact(value):
    return ' '.join(str(value or '').split())

def _records(records):
    return json.dumps(records, ensure_ascii=False) if records else 'NO RESULTS'

@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv keywords newest first; return JSON {id,url,published,title,summary}, NO RESULTS or ERROR."""
    try:
        terms = [s for s in re.findall(r'[^\W_]+(?:-[^\W_]+)*', query) if s.upper() not in {'AND','OR','NOT','ALL','TI','AU','ABS'}]
        if not terms:
            return 'NO RESULTS'
        def fetch():
            global _arxiv_last
            with _arxiv_lock:
                if _arxiv_last is not None:
                    time.sleep(max(0, 3 - (time.monotonic() - _arxiv_last)))
                _arxiv_last = time.monotonic()
                return _request('GET', ARXIV_URL, params={'search_query':' AND '.join('all:'+t for t in terms), 'sortBy':'submittedDate','sortOrder':'descending','max_results':max(1,min(30,max_results)),'start':0})
        root = ET.fromstring(with_retry(fetch, attempts=7, cap=60).text)
        records = []
        ns = {'a':'http://www.w3.org/2005/Atom'}
        for entry in root.findall('a:entry', ns):
            get = lambda key: entry.findtext('a:'+key, default='', namespaces=ns)
            ident = re.sub(r'v\d+$','',get('id').split('/abs/')[-1])
            if not ident:
                continue
            records.append({'id':ident,'url':'https://arxiv.org/abs/'+ident,'published':get('published')[:10],'title':_compact(get('title')),'summary':_compact(get('summary'))[:600]})
        return _records(records)
    except Exception as exc:
        return _error(exc)

def _hf(items, prefer_ai=False):
    records = []
    for item in items:
        paper = item.get('paper') or {}
        if not paper.get('id'):
            continue
        def get(key, default=''):
            return paper.get(key) or item.get(key) or default
        summary = get('ai_summary') if prefer_ai else ''
        records.append({'id':paper['id'],'url':'https://huggingface.co/papers/'+paper['id'],'published':get('publishedAt')[:10], 'title':_compact(get('title')), 'summary':_compact(summary or get('summary'))[:600], 'upvotes':get('upvotes',0),'github':get('githubRepo'),'stars':get('githubStars',0)})
    return records

@tool
def hf_daily_papers(limit: int = 30, date: str = '', keyword: str = '') -> str:
    """Get trending Hugging Face papers, optionally by date and keyword; JSON sorted by upvotes. Not topic search."""
    try:
        params = {'limit':max(1,min(100,limit))}
        if date:
            params['date'] = date
        records = _hf(with_retry(lambda: _request('GET',HF_DAILY_URL,params=params)).json())
        records = [r for r in records if keyword.lower() in (r['title']+' '+r['summary']).lower()]
        return _records(sorted(records,key=lambda r:r['upvotes'],reverse=True))
    except Exception as exc:
        return _error(exc)

@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic; return JSON {id,url,published,title,summary,upvotes,github,stars}."""
    try:
        if not query.strip():
            return 'NO RESULTS'
        return _records(_hf(with_retry(lambda: _request('GET',HF_SEARCH_URL,params={'q':query,'limit':max(1,min(50,limit))})).json(),True))
    except Exception as exc:
        return _error(exc)

def _exa(endpoint, arguments):
    key = os.getenv('EXA_API_KEY', '').strip()
    if not key:
        raise ValueError('Set EXA_API_KEY in the host environment or local .env file')
    def fetch():
        payload = _request(
            'POST', EXA_URL + endpoint,
            headers={'x-api-key': key, 'Content-Type': 'application/json'},
            json=arguments,
        ).json()
        # Extraction may fail for individual URLs even with HTTP 200.
        for status in payload.get('statuses', []):
            if status.get('status') != 'success':
                raise ValueError('Exa content extraction failed: ' + json.dumps(status))
        return _records(payload.get('results', []))
    return with_retry(fetch, attempts=7, base=2, cap=60)

@tool
def web_search(query: str, objective: str = '', num_results: int = 5) -> str:
    """Search web with Exa; describe desired sources in objective. Returns text with source URLs."""
    try:
        if not query.strip():
            return 'NO RESULTS'
        body = {'query': query, 'contents': {'highlights': True},
                'numResults': max(1, min(10, num_results))}
        if objective.strip():
            body['objective'] = objective
        return _exa('/search', body)
    except Exception as exc:
        return _error(exc)

@tool
def web_fetch(url: str) -> str:
    """Fetch one source URL as text for evidence verification; page text limited to 12000 characters, with source metadata."""
    try:
        # Full text is needed to verify citations; preserve valid JSON when limiting it.
        payload = _exa('/contents', {'urls': [url], 'text': {'maxCharacters': 12000}})
        if payload == 'NO RESULTS':
            return payload
        records = json.loads(payload)
        for record in records:
            if isinstance(record.get('text'), str):
                record['text'] = record['text'][:12000]
        return _records(records)
    except Exception as exc:
        return _error(exc)

SOURCE_TOOLS = [arxiv_search,hf_daily_papers,hf_search_papers,web_search,web_fetch]

if __name__ == '__main__':
    for fn,args in [(arxiv_search,{'query':'world model','max_results':3}),(hf_daily_papers,{'limit':20}),(hf_search_papers,{'query':'world model','limit':3}),(web_search,{'query':'survey paper on world models','num_results':2}),(web_fetch,{'url':'https://arxiv.org/abs/1803.10122'})]:
        print(fn.name+'\n'+fn.invoke(args)[:400]+'\n')
