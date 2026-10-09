"""Run a survey and download validated, unchanged sandbox artifacts."""
import json
import os
import re
import sys
import uuid
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / 'reports'
VALIDATOR_SOURCE = ROOT / 'check_citations.py'
FINALIZER_SOURCE = ROOT / 'finalize_citations.py'
FAMILIES = {'arxiv','hf-daily','hf-search','web'}

def slugify(topic):
    return re.sub(r'[^\w]+','-',str(topic).lower(),flags=re.UNICODE).strip('-')[:60].rstrip('-') or 'topic'

def build_prompt(topic):
    return (f'Research topic: {topic}\nCurrent date: {datetime.now(timezone(timedelta(hours=7))).date().isoformat()}. '
            'Produce an English survey following REPORT_TEMPLATE.md structure described in your instructions. '
            'Delegate at least 3 independent research questions; use at least 3 source labels. '
            'Verify evidence and citations inside the sandbox before completing.')

def _field(message, name, default=None):
    return message.get(name,default) if isinstance(message,dict) else getattr(message,name,default)

def summarize(messages, elapsed, model_name):
    calls = Counter()
    tokens = {'input':0,'output':0}
    for message in messages:
        for call in _field(message,'tool_calls',[]) or []:
            calls[call['name']] += 1
        usage = _field(message,'usage_metadata',{}) or {}
        tokens['input'] += usage.get('input_tokens',0) or 0
        tokens['output'] += usage.get('output_tokens',0) or 0
    return {'model':model_name,'elapsed_s':round(elapsed,1),'subagent_calls':calls['task'],
            'tool_calls':dict(calls),'tokens':tokens}

def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    files = download(backend,[REPORT_PATH,SOURCES_PATH])
    report, raw_sources = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    try:
        if not report or not report.strip() or not raw_sources:
            raise ValueError('missing or empty report/sources')
        text, sources = report.decode('utf-8-sig'), json.loads(raw_sources)
        problems = check(text,sources)
        if problems:
            raise ValueError('; '.join(problems))
        labels = {source.get('source') for source in sources}
        if not labels <= FAMILIES or len(labels) < 3:
            raise ValueError('need at least three valid source labels')
        for source in sources:
            label, url, ident = source['source'],source['url'],source.get('id')
            if label == 'arxiv' and (not ident or url != f'https://arxiv.org/abs/{ident}' or re.search(r'v\d+$',str(ident))):
                raise ValueError('invalid arxiv URL/id')
            if label.startswith('hf-') and (not ident or url != f'https://huggingface.co/papers/{ident}'):
                raise ValueError('invalid Hugging Face URL/id')
        for heading in ('## TL;DR','## Background','## Trends and open problems','## References'):
            if not re.search(r'^'+re.escape(heading)+r'\s*$',text,re.M):
                raise ValueError('missing heading '+heading)
        sections = re.findall(r'^## (.+?)\s*$', text, re.M)
        themes = [s for s in sections if s not in {'TL;DR','Background','Trends and open problems','References'}]
        if not 3 <= len(themes) <= 6:
            raise ValueError('need 3-6 thematic sections')
        summary = summarize(messages,elapsed,model_name)
        researcher_calls = sum(call.get('args',{}).get('subagent_type') == 'researcher'
                               for message in messages for call in (_field(message,'tool_calls',[]) or []) if call.get('name')=='task')
        if summary['subagent_calls'] < 3 or researcher_calls < 3:
            raise ValueError('need at least three researcher delegations')
        meta = {'topic':topic,**summary,'n_sources':len(sources),'source_families':sorted(labels),
                'token_scope':'lead messages only; excludes subagent usage'}
    except (ValueError,TypeError,KeyError,AttributeError,UnicodeError) as exc:
        raise RuntimeError(f'invalid sandbox outputs: {exc}') from exc
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True,exist_ok=True)
    slug = slugify(topic)
    payloads = {reports_dir/f'{slug}.md':report,reports_dir/f'{slug}.sources.json':raw_sources,
                reports_dir/f'{slug}.meta.json':json.dumps(meta,ensure_ascii=False,indent=2).encode('utf-8')}
    previous = {path:path.read_bytes() if path.exists() else None for path in payloads}
    staged, replaced = {}, []
    try:
        for path, data in payloads.items():
            temp = reports_dir / ('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
            staged[path] = temp
            with temp.open('xb') as handle:
                handle.write(data)
        for path, temp in staged.items():
            os.replace(temp,path)
            replaced.append(path)
    except OSError as exc:
        for path in replaced:
            if previous[path] is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous[path])
        raise RuntimeError(f'could not save outputs: {exc}') from exc
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)
    return reports_dir/f'{slug}.md'

def main(topic):
    if not topic.strip():
        print('Usage: python research.py "<topic>"',file=sys.stderr); return 2
    started = time.monotonic()
    try:
        model = make_model()
        with open_sandbox() as backend:
            created = backend.execute(f'mkdir -p {WORKDIR}/research/notes {WORKDIR}/report')
            if created.exit_code != 0:
                raise RuntimeError('cannot create sandbox workspace')
            upload(backend,{VALIDATOR_PATH:VALIDATOR_SOURCE.read_bytes(),FINALIZER_PATH:FINALIZER_SOURCE.read_bytes()})
            # Verify uploads because the provided upload helper discards per-file errors.
            seeded = download(backend,[VALIDATOR_PATH,FINALIZER_PATH])
            if seeded.get(VALIDATOR_PATH) != VALIDATOR_SOURCE.read_bytes() or seeded.get(FINALIZER_PATH) != FINALIZER_SOURCE.read_bytes():
                raise RuntimeError('sandbox script upload failed')
            result = build_lead_agent(backend,model).invoke({'messages':[{'role':'user','content':build_prompt(topic)}]},config={'recursion_limit':1000})
            validated = backend.execute(f'python3 {VALIDATOR_PATH}')
            if validated.exit_code != 0 or not validated.output.startswith('OK:'):
                raise RuntimeError('sandbox citation validation failed')
            path = save_outputs(backend,topic,result['messages'],time.monotonic()-started,
                                os.getenv('LAB_MODEL') or os.getenv('OPENAI_DEPLOYMENT_MODEL','unknown'))
        print(f'Saved report: {path}'); return 0
    except Exception as exc:
        # Provider exceptions can include credentials; use the same host redaction as tools.
        from tools import _error
        print('FAILED: '+_error(exc),file=sys.stderr); return 1

if __name__ == '__main__':
    sys.exit(main(' '.join(sys.argv[1:])))
