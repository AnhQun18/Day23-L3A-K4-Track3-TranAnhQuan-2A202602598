"""Validate report citations using only the Python standard library."""
import json
import re
import sys
from collections import Counter

REPORT = '/tmp/work/report/report.md'
SOURCES = '/tmp/work/research/sources.json'

def check(report_text, sources):
    problems = []
    if not isinstance(sources, list) or not sources:
        return ['no sources in sources.json']
    indexed, urls = {}, set()
    for item in sources:
        if not isinstance(item, dict):
            problems.append('source must be an object'); continue
        n, url = item.get('n'), item.get('url')
        if type(n) is not int or n < 1:
            problems.append('source n must be a positive integer'); continue
        if n in indexed:
            problems.append(f'duplicate source number [{n}]')
        indexed[n] = item
        if not isinstance(url, str) or not re.match(r'^https?://\S+$', url):
            problems.append(f'invalid URL for [{n}]')
        elif url in urls:
            problems.append(f'duplicate URL for [{n}]')
        else:
            urls.add(url)
    # Fences may use backticks or tildes. Their contents are not prose citations.
    clean = re.sub(r'(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$', '', report_text)
    parts = re.split(r'(?m)^## References\s*$', clean, maxsplit=1)
    if len(parts) != 2:
        return problems + ['missing ## References']
    body, references = parts
    body = re.sub(r'`+[^`\n]*`+', '', body)
    cited = set()
    for match in re.finditer(r'\[(\d+(?:\s*(?:,|-)\s*\d+)*)\](?!\s*\()', body):
        for part in match[1].split(','):
            ends = [int(x) for x in part.split('-')]
            if len(ends) == 1:
                cited.add(ends[0])
            elif ends[0] <= ends[1] and ends[1] - ends[0] <= 10000:
                cited.update(range(ends[0], ends[1] + 1))
            else:
                problems.append('invalid citation range')
    for n in sorted(cited - indexed.keys()):
        problems.append(f'[{n}] cited but missing from sources.json')
    for n in sorted(indexed.keys() - cited):
        problems.append(f'source [{n}] never cited')
    counts = Counter()
    for match in re.finditer(r'(?m)^\[(\d+)\]\s*(.*)$', references):
        n, text = int(match[1]), match[2]
        counts[n] += 1
        links = re.findall(r'https?://[^\s<>]+', text)
        if n not in indexed:
            problems.append(f'unknown reference [{n}]')
        elif len(links) != 1 or links[0] != indexed[n].get('url'):
            problems.append(f'reference [{n}] must contain exactly its source URL')
    for n in indexed:
        if counts[n] != 1:
            problems.append(f'reference [{n}] occurs {counts[n]} times (expected 1)')
    return problems

def main(argv):
    try:
        with open(argv[1] if len(argv)>1 else REPORT, encoding='utf-8') as f:
            report = f.read()
        with open(argv[2] if len(argv)>2 else SOURCES, encoding='utf-8') as f:
            sources = json.load(f)
        problems = check(report, sources)
    except (OSError, ValueError) as exc:
        print(f'cannot read inputs: {exc}'); return 1
    if problems:
        print('\n'.join(problems)); return 1
    print(f'OK: {len(sources)} sources, all citations resolve'); return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
