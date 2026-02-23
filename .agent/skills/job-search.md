# Job Search Skill

## Luis's constraints
- No government, federal, or clearance-required roles
- Remote only
- Skills: Python, backend, cloud, distributed systems, full-stack

## Working endpoints (bot-friendly)

### RemoteOK - JSON API (best)
COMMAND: curl -sL -H "User-Agent: Mozilla/5.0" "https://remoteok.com/api?tags=python" | python3 -c "import sys,json; raw=sys.stdin.read().strip(); print('no data from RemoteOK - switch to SEARCH') if not raw else [print(j.get('company','?'), '|', j.get('position','?'), '|', j.get('url','?')) for j in json.loads(raw)[1:6] if isinstance(j,dict)]"

### We Work Remotely - RSS (reliable)
COMMAND: curl -sL "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss" | python3 -c "import sys,xml.etree.ElementTree as ET; data=sys.stdin.read().strip(); root=ET.fromstring(data) if data else None; titles=[(i.findtext('title') or '').strip() for i in (root.findall('.//item') if root is not None else [])]; [print(t) for t in titles[:8] if t]"

### Hacker News Who's Hiring - monthly thread (signal-rich)
SEARCH: site:news.ycombinator.com "who is hiring" python backend remote 2026

### Wellfound (AngelList) - use FETCH not curl
FETCH: https://wellfound.com/role/r/software-engineer

### LinkedIn - use SEARCH not curl (curl gets blocked)
SEARCH: LinkedIn remote Python backend engineer job 2026

## What NOT to do
- Never curl Indeed or Glassdoor - Cloudflare blocks immediately
- Never use grep with lookbehind on HTML - variable length, always fails
- Never curl LinkedIn - always blocked
- If curl returns "Attention Required" or empty, switch to SEARCH immediately

## Workflow for job leads task
1. Hit RemoteOK API for JSON listings
2. Hit WWR RSS for backend roles
3. SEARCH for HN hiring thread
4. Cross-reference results against /home/mantis/resume.md skills
5. Filter out: government, federal, clearance, defense, contractor
6. Post top 3 to Discord with: title | company | direct URL

## Resume location
Always read from: /home/mantis/resume.md
Key skills to match against: Python, backend, cloud optimization,
distributed systems, React, AWS, data pipelines, full-stack
