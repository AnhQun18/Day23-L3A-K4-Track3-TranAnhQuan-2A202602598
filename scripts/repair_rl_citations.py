"""One-time deterministic repair of the original generated RL artifact, inside sandbox.

No model or network calls. Source discovery labels and original run statistics are
preserved. The input hash prevents applying the old numbering map to a new report.
"""
import hashlib
import json
import re
from pathlib import Path

REPORT = Path('/tmp/work/report/report.md')
SOURCES = Path('/tmp/work/research/sources.json')
EXPECTED = '00189695ae22df7547d4c6b07f2ff20a0b6344149f6a9f713178dfae5b3f7d85'

# Original body number -> source record number, established from named papers.
MAPPING = {1:20,2:8,3:9,4:24,5:25,6:26,7:8,8:27,9:28,10:29,11:30,
           12:10,13:11,14:16,15:31,16:3,17:32,18:18,19:19,20:33,21:21,
           22:36,23:23,25:14,26:37,27:37,28:38,29:15,30:39,31:40,32:41,
           33:8,34:5,35:4,36:6,37:44,39:34,40:35,41:17,42:1,43:22,
           1007:2,1012:12,1013:13,1018:42,1019:7,1023:43}


def repair():
    raw = REPORT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Unexpected report revision; refusing stale citation mapping')
    sources = json.loads(SOURCES.read_bytes())
    assert next(s for s in sources if s['n']==20)['id'] == '2402.03300'
    paragraphs = re.split(r'\n\s*\n', raw.decode('utf-8').split('## References')[0])
    # Resolve numbers reused for different papers before simultaneous remapping.
    paragraphs[2] = paragraphs[2].replace('average accuracy [12]', 'average accuracy [1012]')
    paragraphs[14] = paragraphs[14].replace('[13]', '[1013]')
    paragraphs[25] = paragraphs[25].replace('[18]', '[1018]').replace('[19]', '[1019]')
    paragraphs[28] = paragraphs[28].replace('[12]', '[1012]').replace('[23]', '[1023]')
    paragraphs[31] = paragraphs[31].replace('[7]', '[1007]')
    for i in (35,37):
        paragraphs[i] = paragraphs[i].replace('[12]', '[1012]')
    # Omit evidence whose intended source is absent or cannot be identified.
    paragraphs[16] = paragraphs[16].split(' The behavioural consequence is documented too:')[0]
    paragraphs[29] = paragraphs[29].split(', and a proposed Entropulse schedule')[0] + '.'
    paragraphs[34] = paragraphs[34].replace('systematic abandonment of rule induction [24], ', '')
    start = paragraphs[36].index('The pass@1/pass@k trade-off is also')
    tail = paragraphs[36].index('pass@k optimization can degrade pass@1')
    paragraphs[36] = paragraphs[36][:start] + 'The ' + paragraphs[36][tail:]
    start = paragraphs[37].index('Numeric discrepancies appear even within single papers:')
    end = paragraphs[37].index('Gaps this survey could not close include:')
    paragraphs[37] = paragraphs[37][:start] + paragraphs[37][end:]
    paragraphs[26] = paragraphs[26].replace(
        '; that source is the origin of the widely circulated estimate that the RL phase '
        "accounted for the large majority of R1's headline training cost", '')
    paragraphs[35] = paragraphs[35].replace('[37][38]', '[37]')
    body = '\n\n'.join(paragraphs).strip() + '\n'
    def replace(match):
        n = int(match.group(1))
        if n not in MAPPING:
            raise ValueError(f'Unresolved original citation [{n}]')
        return f'[{MAPPING[n]}]'
    body = re.sub(r'\[(\d+)\]', replace, body)
    REPORT.write_text(body, encoding='utf-8')
    print('Repaired citation mapping; removed claims without identifiable sources')


if __name__ == '__main__':
    repair()
