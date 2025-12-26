import re
import requests
from bs4 import BeautifulSoup

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
    "Referer": "https://matchcenter.fvrz.ch/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_team(s: str) -> str:
    s = _clean(s).lower()
    s = re.sub(r"(fc|sc|sv|1\.?|erste|mannschaft|team|club)", " ", s)
    s = re.sub(r"[^a-z0-9äöü\s\-\.\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fetch_table_from_matchcenter(v: int) -> str:
    url = f"https://matchcenter.fvrz.ch/default.aspx?lng=1&&cxxlnus=1&v={v}&bn=0"
    s = requests.Session()
    r = s.get(url, timeout=20, headers=UA)

    r.raise_for_status()
    return r.text

def parse_rank_goals_matches(html_text: str, team_query: str):
    soup = BeautifulSoup(html_text, "lxml")

    qn = _norm_team(team_query)
    if not qn:
        return None

    best = None
    best_score = 0.0

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for tr in rows:
            tds = tr.find_all(["td", "th"])
            cols = [_clean(td.get_text(" ", strip=True)) for td in tds]
            if len(cols) < 5:
                continue

            row_text = " ".join(cols)

            # try detect patterns like:
            # rank | team | matches | ... | goals "45 : 30"
            m_goals = re.search(r"(\d+)\s*:\s*(\d+)", row_text)
            if not m_goals:
                continue

            gf = int(m_goals.group(1))
            ga = int(m_goals.group(2))

            # rank = first integer in row
            m_rank = re.match(r"^\s*(\d+)\b", cols[0])
            if not m_rank:
                continue
            rank = int(m_rank.group(1))

            # matches = first integer after team cell (heuristic)
            # find any cell that is only digits and looks like match count
            matches = None
            for c in cols:
                if re.fullmatch(r"\d{1,2}", c):
                    n = int(c)
                    if 1 <= n <= 60:
                        matches = n
                        break
            if matches is None:
                continue

            # team name guess: pick the longest non-numeric cell
            team_name = max(
                (c for c in cols if not re.fullmatch(r"[\d\W]+", c) and len(c) >= 3),
                key=len,
                default=""
            )
            if not team_name:
                continue

            tn = _norm_team(team_name)

            # similarity score (simple token overlap)
            q_tokens = set(qn.split())
            t_tokens = set(tn.split())
            if not q_tokens or not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
            score = overlap

            if score > best_score:
                best_score = score
                best = {
                    "name": team_name,
                    "rank": rank,
                    "matches": matches,
                    "goals_for": gf,
                    "goals_against": ga,
                }

    # require at least some overlap
    return best if best and best_score >= 0.5 else None
