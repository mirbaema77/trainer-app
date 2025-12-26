import requests

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
    "Referer": "https://matchcenter.fvrz.ch/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

url = "https://matchcenter.fvrz.ch/default.aspx?lng=1&&cxxlnus=1&v=1514&bn=0"
r = requests.get(url, headers=UA, timeout=20)
print("STATUS:", r.status_code)
print("HEAD:", r.text[:500])
