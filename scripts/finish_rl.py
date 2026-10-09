"""Finish the existing RL report with Daytona file operations, without LLM/Exa calls."""
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents import REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, FINALIZER_PATH
from check_citations import check
from sandbox import open_sandbox, upload, download


def main():
    stem = ROOT / 'reports' / 'survey-about-reinforcement-learning-for-llm-reasoning'
    report_path = stem.with_suffix('.md')
    sources_path = stem.with_suffix('.sources.json')
    meta_path = stem.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_bytes())
    seed = {REPORT_PATH: report_path.read_bytes(), SOURCES_PATH: sources_path.read_bytes(),
            VALIDATOR_PATH: (ROOT/'check_citations.py').read_bytes(),
            FINALIZER_PATH: (ROOT/'finalize_citations.py').read_bytes(),
            '/tmp/work/repair_rl_citations.py': (ROOT/'scripts/repair_rl_citations.py').read_bytes()}
    with open_sandbox() as backend:
        created = backend.execute('mkdir -p /tmp/work/report')
        if created.exit_code:
            raise RuntimeError('Cannot initialize sandbox')
        upload(backend, seed)
        if download(backend,list(seed)) != seed:
            raise RuntimeError('Sandbox upload verification failed')
        for command in ('python3 /tmp/work/repair_rl_citations.py',
                        f'python3 {FINALIZER_PATH}', f'python3 {VALIDATOR_PATH}'):
            result = backend.execute(command)
            print(result.output, flush=True)
            if result.exit_code:
                raise RuntimeError('Sandbox correction failed')
        files = download(backend,[REPORT_PATH,SOURCES_PATH])
        report, raw_sources = files[REPORT_PATH], files[SOURCES_PATH]
        sources = json.loads(raw_sources)
        errors = check(report.decode('utf-8'),sources)
        labels = {s['source'] for s in sources}
        if errors or len(labels) < 3 or not labels <= {'arxiv','hf-daily','hf-search','web'}:
            raise RuntimeError(f'Rejected corrected artifact: {errors}')
    # Preserve original measured model statistics; this correction uses no model.
    meta.update(n_sources=len(sources), source_families=sorted(labels))
    meta['postprocessing'] = {
        'method':'deterministic citation mapping in Daytona; provided finalizer and validator',
        'llm_calls':0, 'exa_calls':0,
        'review':'named-paper attribution reviewed; unidentifiable claims removed',
        'numeric_spotchecks':['DeepSeekMath GSM8K/MATH', 'DAPO AIME 2024', 'ReTool AIME 2024'],
        'statistics':'original research run retained; correction did not invoke researchers'}
    payloads = {report_path:report, sources_path:raw_sources,
                meta_path:json.dumps(meta,ensure_ascii=False,indent=2).encode('utf-8')}
    previous = {path:path.read_bytes() for path in payloads}
    staged, replaced = {}, []
    try:
        for path,data in payloads.items():
            temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
            staged[path] = temporary
            with temporary.open('xb') as handle:
                handle.write(data)
        for path,temporary in staged.items():
            os.replace(temporary,path)
            replaced.append(path)
    except OSError:
        for path in replaced:
            path.write_bytes(previous[path])
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    print('Saved exact sandbox artifacts; original run statistics preserved')


if __name__ == '__main__':
    main()
