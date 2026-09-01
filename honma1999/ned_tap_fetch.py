"""Bulk NED queries via the TAP service -- the sanctioned way.

NED runs a Table Access Protocol endpoint at https://ned.ipac.caltech.edu/tap.
One ADQL query replaces thousands of per-object page fetches, so this cannot
hammer their web servers the way HTML scraping does. Big queries must use the
ASYNCHRONOUS mode (UWS): submit a job, poll its phase at a gentle cadence,
fetch the result once COMPLETE. The synchronous endpoint kills anything over
60 seconds.

Usage:
    python ned_tap_fetch.py "SELECT ... FROM NEDTAP.objdir WHERE ..." out.csv

Useful columns of NEDTAP.objdir: prefname, ra, dec, gallon, gallat,
prefphytype ('G' for galaxies), z, zunc, zflag, objid.
"""
import sys
import time
import urllib.parse
import urllib.request

BASE = 'https://ned.ipac.caltech.edu/tap'
POLL_S = 10          # polling cadence -- keep this gentle
TIMEOUT_S = 1800

def _get(url):
    req = urllib.request.Request(url, headers={'User-Agent':
        'binary-galaxy-research (astropy-style TAP client; wclar001@odu.edu)'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode('utf-8', 'replace')

def fetch(adql, out_csv):
    params = urllib.parse.urlencode({
        'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
        'PHASE': 'RUN', 'QUERY': adql})
    req = urllib.request.Request(BASE + '/async', data=params.encode(),
        headers={'User-Agent':
        'binary-galaxy-research (astropy-style TAP client; wclar001@odu.edu)'})
    with urllib.request.urlopen(req, timeout=120) as r:
        job_url = r.url  # redirected to the job resource
    print('job:', job_url)

    t0 = time.time()
    while True:
        phase = _get(job_url + '/phase').strip()
        print('  phase: %s   (%.0f s)' % (phase, time.time() - t0))
        if phase == 'COMPLETED':
            break
        if phase in ('ERROR', 'ABORTED'):
            print(_get(job_url + '/error')[:2000])
            raise SystemExit('TAP job failed')
        if time.time() - t0 > TIMEOUT_S:
            raise SystemExit('TAP job timed out')
        time.sleep(POLL_S)

    data = _get(job_url + '/results/result')
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write(data)
    print('wrote %s (%d lines)' % (out_csv, data.count('\n')))

if __name__ == '__main__':
    fetch(sys.argv[1], sys.argv[2])
