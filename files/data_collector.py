"""
LLM Gateway - Free Data Collector v2
=======================================
Collects real-time data from FREE Python APIs BEFORE any LLM call.
Only escalates to Gemini 3 Flash (cheap_plus + web_search) when ALL APIs fail.

Free Tools:
  - Open-Meteo:       FREE weather + geocoding, no API key, no limit
  - yfinance:         FREE stocks, no API key (Yahoo Finance)
  - NewsAPI.org:      FREE tier = 100 req/day (email registration)
  - DuckDuckGo:       FREE web search, no API key, no limit
  - TextBlob:         FREE sentiment analysis (local Python lib)

Pipeline:
  1. Detect if query needs real-time data (weather/stocks/news/web)
  2. Try free APIs -> collect raw data
  3. If SUCCESS -> feed data to CHEAP model for formatting
  4. If FAILURE -> escalate to CHEAP_PLUS (Gemini 3 Flash + paid web search)
"""

import os
import re
import time
import logging
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import httpx

log = logging.getLogger("gateway.data_collector")


# DATA TYPES

class DataCategory(str, Enum):
    WEATHER = "weather"
    STOCKS = "stocks"
    NEWS = "news"
    WEB_SEARCH = "web_search"
    MIXED = "mixed"
    NONE = "none"


@dataclass
class CollectedData:
    category: DataCategory = DataCategory.NONE
    success: bool = False
    data_text: str = ""
    raw_data: dict = field(default_factory=dict)
    sources_tried: list = field(default_factory=list)
    sources_succeeded: list = field(default_factory=list)
    sources_failed: list = field(default_factory=list)
    collection_time_ms: float = 0.0
    error_summary: str = ""

    @property
    def should_escalate(self):
        if self.category == DataCategory.NONE:
            return False
        return len(self.sources_succeeded) == 0 and len(self.sources_failed) > 0

    @property
    def has_data(self):
        return bool(self.data_text.strip())


# QUERY CLASSIFIER

WEATHER_PATTERNS = [
    re.compile(r'\b(wetter|weather|temperatur|temperature|regen|rain|schnee|snow|'
               r'frost|glatteis|ice|nebel|fog|wind|sturm|storm|unwetter|gewitter|'
               r'thunder|forecast|vorhersage|sonnig|sunny|niederschlag|precipitation|'
               r'humidity|feuchtigkeit)\b', re.I),
]

STOCK_PATTERNS = [
    re.compile(r'\b(aktie|stock|aktienkurs|share.?price|kurs|dax|nasdaq|s&p|'
               r'dow.?jones|msci|etf|bitcoin|btc|ethereum|eth|crypto|krypto|'
               r'wechselkurs|exchange.?rate|rendite|yield|dividende|dividend|'
               r'portfolio|depot)\b', re.I),
    re.compile(r'\b\w*aktie[n]?\b', re.I),
    re.compile(r'\$[A-Z]{1,5}\b'),
    re.compile(r'\b[A-Z]{2,5}\.(DE|F|L|PA|SW|AS|MI|MC|HK|T)\b'),
]

NEWS_PATTERNS = [
    re.compile(r'\b(nachrichten|news|schlagzeile|headline|meldung|'
               r'breaking|neuigkeiten|pressemitteilung)\b', re.I),
    re.compile(r'\b(regulierung|regulation|gesetz|law|reform|politik|'
               r'sanktion|sanction|embargo|zoll|tariff|streik|strike|'
               r'insolvenz|bankruptcy|acquisition|merger|ipo)\b', re.I),
]

REALTIME_PATTERNS = [
    re.compile(r'\b(aktuell|currently|right.?now|gerade|heute|today|'
               r'live|jetzt|derzeit|momentan|latest)\b', re.I),
    re.compile(r'\b(spielstand|score|ergebnis|result|tabelle|standings|'
               r'fahrplan|schedule|preis.?von|price.?of)\b', re.I),
]


def classify_realtime_query(query):
    categories = set()
    for p in WEATHER_PATTERNS:
        if p.search(query):
            categories.add(DataCategory.WEATHER)
            break
    for p in STOCK_PATTERNS:
        if p.search(query):
            categories.add(DataCategory.STOCKS)
            break
    for p in NEWS_PATTERNS:
        if p.search(query):
            categories.add(DataCategory.NEWS)
            break
    if not categories:
        for p in REALTIME_PATTERNS:
            if p.search(query):
                return DataCategory.WEB_SEARCH
        return DataCategory.NONE
    return DataCategory.MIXED if len(categories) > 1 else categories.pop()


# GEOCODING - Open-Meteo (FREE, any city worldwide)

_geocode_cache = {}

_KNOWN_CITIES = {
    "berlin": ("Berlin", 52.52, 13.405),
    "hamburg": ("Hamburg", 53.551, 9.994),
    "frankfurt": ("Frankfurt", 50.110, 8.682),
    "stuttgart": ("Stuttgart", 48.776, 9.183),
    "konstanz": ("Konstanz", 47.660, 9.175),
    "london": ("London", 51.507, -0.128),
    "paris": ("Paris", 48.857, 2.352),
    "new york": ("New York", 40.713, -74.006),
    "tokyo": ("Tokyo", 35.682, 139.759),
    "sydney": ("Sydney", -33.869, 151.209),
    "amsterdam": ("Amsterdam", 52.370, 4.895),
    "barcelona": ("Barcelona", 41.389, 2.159),
    "madrid": ("Madrid", 40.417, -3.704),
    "istanbul": ("Istanbul", 41.009, 28.978),
    "dubai": ("Dubai", 25.205, 55.271),
    "seoul": ("Seoul", 37.567, 126.978),
    "mumbai": ("Mumbai", 19.076, 72.878),
    "bangkok": ("Bangkok", 13.756, 100.502),
    "bonn": ("Bonn", 50.737, 7.099),
    "bremen": ("Bremen", 53.079, 8.802),
    "dresden": ("Dresden", 51.051, 13.738),
    "leipzig": ("Leipzig", 51.340, 12.375),
    "hannover": ("Hannover", 52.376, 9.739),
    "freiburg": ("Freiburg", 47.999, 7.842),
    "mannheim": ("Mannheim", 49.489, 8.467),
    "karlsruhe": ("Karlsruhe", 49.007, 8.404),
    "heidelberg": ("Heidelberg", 49.399, 8.673),
    "bern": ("Bern", 46.948, 7.448),
    "basel": ("Basel", 47.559, 7.589),
    "graz": ("Graz", 47.070, 15.439),
    "salzburg": ("Salzburg", 47.811, 13.055),
}

# Also match with umlauts
for _orig, _repl in [("muenchen", "München"), ("koeln", "Köln"), ("duesseldorf", "Düsseldorf"),
                     ("zuerich", "Zürich"), ("nuernberg", "Nürnberg"), ("tuebingen", "Tübingen")]:
    pass  # handled by _normalize below

_UMLAUT_CITIES = {
    "münchen": ("München", 48.137, 11.576), "munich": ("München", 48.137, 11.576),
    "köln": ("Köln", 50.938, 6.960), "cologne": ("Köln", 50.938, 6.960),
    "düsseldorf": ("Düsseldorf", 51.228, 6.774),
    "zürich": ("Zürich", 47.377, 8.540), "zurich": ("Zürich", 47.377, 8.540),
    "wien": ("Wien", 48.208, 16.374), "vienna": ("Wien", 48.208, 16.374),
    "nürnberg": ("Nürnberg", 49.454, 11.078),
    "tübingen": ("Tübingen", 48.521, 9.058),
    "würzburg": ("Würzburg", 49.794, 9.929),
    "genf": ("Genf", 46.205, 6.144), "geneva": ("Genf", 46.205, 6.144),
    "rom": ("Rom", 41.903, 12.496), "rome": ("Rom", 41.903, 12.496),
    "athen": ("Athen", 37.984, 23.728), "athens": ("Athen", 37.984, 23.728),
    "mailand": ("Mailand", 45.464, 9.190), "milan": ("Mailand", 45.464, 9.190),
    "prag": ("Prag", 50.076, 14.438), "prague": ("Prag", 50.076, 14.438),
    "tokio": ("Tokyo", 35.682, 139.759),
    "peking": ("Peking", 39.904, 116.407), "beijing": ("Peking", 39.904, 116.407),
    "kapstadt": ("Kapstadt", -33.925, 18.424),
    "kairo": ("Kairo", 30.044, 31.236),
    "singapur": ("Singapur", 1.352, 103.820), "singapore": ("Singapur", 1.352, 103.820),
    "moskau": ("Moskau", 55.756, 37.617), "moscow": ("Moskau", 55.756, 37.617),
    "warschau": ("Warschau", 52.230, 21.012), "warsaw": ("Warschau", 52.230, 21.012),
    "lissabon": ("Lissabon", 38.722, -9.139), "lisbon": ("Lissabon", 38.722, -9.139),
    "los angeles": ("Los Angeles", 34.052, -118.244),
    "san francisco": ("San Francisco", 37.775, -122.419),
    "rio de janeiro": ("Rio de Janeiro", -22.907, -43.173),
    "buenos aires": ("Buenos Aires", -34.604, -58.382),
    "innsbruck": ("Innsbruck", 47.260, 11.394),
}
_KNOWN_CITIES.update(_UMLAUT_CITIES)


async def geocode_city(city_name):
    """Resolve any city to (display_name, lat, lon). Uses Open-Meteo Geocoding (FREE)."""
    low = city_name.lower().strip()

    # 1. Known cities (instant)
    if low in _KNOWN_CITIES:
        return _KNOWN_CITIES[low]

    # 2. Cache (instant)
    if low in _geocode_cache:
        return _geocode_cache[low]

    # 3. Open-Meteo Geocoding API (FREE, no key, any city worldwide)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city_name, "count": 1, "language": "de", "format": "json"},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", [])
        if results:
            r = results[0]
            name = r.get("name", city_name)
            country = r.get("country", "")
            display = f"{name}, {country}" if country else name
            result = (display, r["latitude"], r["longitude"])
            _geocode_cache[low] = result
            log.debug(f"Geocoded '{city_name}' -> {display}")
            return result
    except Exception as e:
        log.warning(f"Geocoding failed for '{city_name}': {e}")
    return None


def extract_city_name(query):
    """Extract a city/location name from a natural language query."""
    q = query.strip()
    # Regex patterns for city extraction
    patterns = [
        re.compile(r'(?:wetter|weather|temperatur|temperature|forecast|vorhersage|'
                   r'regen|rain|schnee|snow|wind|klima)\s+'
                   r'(?:in|f.r|for|at|near|bei)?\s*'
                   r'([\w\u00c0-\u024f][\w\u00c0-\u024f\s\-\.]{1,40})', re.I),
        re.compile(r'(?:wie\s+ist\s+(?:das\s+)?wetter|how.s\s+the\s+weather)\s+'
                   r'(?:in|f.r|at)?\s*'
                   r'([\w\u00c0-\u024f][\w\u00c0-\u024f\s\-\.]{1,40})', re.I),
        re.compile(r'\bin\s+([\w\u00c0-\u024f][\w\u00c0-\u024f\s\-\.]{1,40}?)\s+'
                   r'(?:wetter|regnet|schneit|rain|snow|temperatur|weather)', re.I),
        re.compile(r'^([\w\u00c0-\u024f][\w\u00c0-\u024f\s\-\.]{1,30}?)\s+'
                   r'(?:wetter|weather|temperatur|forecast)', re.I),
    ]
    bad = {"der", "die", "das", "dem", "den", "the", "morgen", "heute",
           "tomorrow", "today", "gerade", "jetzt", "aktuell", "currently",
           "schlecht", "gut", "morgens", "abends", "mittags", "nachmittags",
           "diese", "woche", "naechste", "next", "this", "week"}
    for pattern in patterns:
        m = pattern.search(q)
        if m:
            city = m.group(1).strip().rstrip("?.!,;:")
            # Strip trailing common words (e.g. "Konstanz morgen" -> "Konstanz")
            words = city.split()
            while words and words[-1].lower() in bad:
                words.pop()
            city = " ".join(words)
            if city.lower() not in bad and len(city) >= 2:
                return city
    # Fallback: check known cities in query
    q_lower = query.lower()
    for key in sorted(_KNOWN_CITIES.keys(), key=len, reverse=True):
        if key in q_lower:
            return _KNOWN_CITIES[key][0]
    return None


# TICKER & KEYWORD EXTRACTION

COMPANY_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA", "meta": "META",
    "netflix": "NFLX",
    "dhl": "DHL.DE", "deutsche post": "DHL.DE",
    "siemens": "SIE.DE", "sap": "SAP.DE", "basf": "BAS.DE",
    "bmw": "BMW.DE", "volkswagen": "VOW3.DE", "vw": "VOW3.DE",
    "mercedes": "MBG.DE", "allianz": "ALV.DE", "deutsche bank": "DBK.DE",
    "bayer": "BAYN.DE", "adidas": "ADS.DE", "telekom": "DTE.DE",
    "infineon": "IFX.DE", "rheinmetall": "RHM.DE", "porsche": "P911.DE",
    "airbus": "AIR.PA", "nestle": "NESN.SW", "roche": "ROG.SW",
    "dax": "^GDAXI", "s&p 500": "^GSPC", "s&p": "^GSPC",
    "nasdaq": "^IXIC", "dow jones": "^DJI",
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD", "solana": "SOL-USD",
}


def extract_tickers(query):
    q = query.lower()
    tickers = [t for n, t in COMPANY_TICKERS.items() if n in q and t]
    tickers += re.findall(r'\$([A-Z]{1,5})\b', query)
    tickers += re.findall(r'\b([A-Z]{2,5}\.(?:DE|F|L|PA|SW|AS))\b', query)
    seen = set()
    return [t for t in tickers if not (t in seen or seen.add(t))][:5]


def extract_news_keywords(query):
    stop = {"was", "wie", "ist", "sind", "der", "die", "das", "ein", "eine",
            "und", "oder", "mit", "von", "zu", "in", "auf", "what", "how",
            "is", "are", "the", "a", "an", "and", "or", "for", "with",
            "aktuell", "current", "latest", "heute", "today", "news", "nachrichten"}
    words = re.findall(r'\b[\w\u00c0-\u024f]{3,}\b', query.lower())
    return [w for w in words if w not in stop][:5]


def extract_search_query(query):
    q = re.sub(r'\b(suche|search|finde|find|zeige|show)\b', '', query, flags=re.I)
    q = q.strip().strip("?.!,;:")
    return f"{q} {datetime.now().strftime('%B %Y')}" if len(q) < 100 else q


# FREE API COLLECTORS

class WeatherCollector:
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    CODES = {
        0: "Klar", 1: "Meist klar", 2: "Teilw. bewoelkt", 3: "Bewoelkt",
        45: "Nebel", 48: "Raureif", 51: "Niesel", 53: "Nieselregen",
        56: "Gefr. Niesel", 61: "Leichter Regen", 63: "Regen", 65: "Starkregen",
        66: "Gefr. Regen", 71: "Leichter Schnee", 73: "Schnee", 75: "Starker Schnee",
        80: "Regenschauer", 81: "Starke Schauer", 85: "Schneeschauer",
        95: "Gewitter", 96: "Gewitter+Hagel",
    }

    async def collect(self, city, lat, lon):
        try:
            params = {
                "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                           "precipitation,weather_code,wind_speed_10m,wind_gusts_10m",
                "daily": "temperature_2m_max,temperature_2m_min,"
                         "precipitation_probability_max,weather_code",
            }
            async with httpx.AsyncClient() as client:
                r = await client.get(self.BASE_URL, params=params, timeout=10.0)
                r.raise_for_status()
                data = r.json()
            c = data.get("current", {})
            d = data.get("daily", {})
            desc = self.CODES.get(c.get("weather_code", -1), "?")
            lines = [
                f"=== WETTER {city.upper()} ({datetime.now():%d.%m.%Y %H:%M}) ===",
                f"Quelle: Open-Meteo | Koord: {lat:.2f}, {lon:.2f}",
                f"Aktuell: {desc}",
                f"Temp: {c.get('temperature_2m','?')}C (gefuehlt {c.get('apparent_temperature','?')}C)",
                f"Feuchte: {c.get('relative_humidity_2m','?')}%",
                f"Wind: {c.get('wind_speed_10m','?')} km/h (Boeen {c.get('wind_gusts_10m','?')})",
                f"Niederschlag: {c.get('precipitation',0)} mm",
            ]
            t = c.get("temperature_2m", 99)
            p = c.get("precipitation", 0)
            if isinstance(t,(int,float)) and t<=2 and isinstance(p,(int,float)) and p>0:
                lines.append("!! GLATTEISRISIKO !!")
            elif isinstance(t,(int,float)) and t<=0:
                lines.append("!! FROST !!")
            if d.get("time"):
                lines.append("\nPrognose:")
                for i, ds in enumerate(d["time"][:3]):
                    dc = d.get("weather_code",[0])[i] if i<len(d.get("weather_code",[])) else 0
                    tmax = d.get("temperature_2m_max",["?"])[i] if i<len(d.get("temperature_2m_max",[])) else "?"
                    tmin = d.get("temperature_2m_min",["?"])[i] if i<len(d.get("temperature_2m_min",[])) else "?"
                    rp = d.get("precipitation_probability_max",[0])[i] if i<len(d.get("precipitation_probability_max",[])) else 0
                    lines.append(f"  {ds}: {self.CODES.get(dc,'?')} | {tmin}-{tmax}C | Regen {rp}%")
            return "\n".join(lines)
        except Exception as e:
            log.warning(f"Open-Meteo failed for {city}: {e}")
            return None


class StockCollector:
    async def collect(self, tickers):
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self._fetch, tickers)
        except Exception as e:
            log.warning(f"yfinance: {e}")
            return None

    def _fetch(self, tickers):
        try:
            import yfinance as yf
        except ImportError:
            return None
        lines = [f"=== BOERSE ({datetime.now():%d.%m.%Y %H:%M}) === Quelle: Yahoo Finance"]
        ok = False
        for ts in tickers:
            try:
                t = yf.Ticker(ts)
                fi = t.fast_info if hasattr(t,'fast_info') else {}
                price = getattr(fi,'last_price',None)
                prev = getattr(fi,'previous_close',None)
                cur = getattr(fi,'currency','USD')
                if price is None:
                    h = t.history(period="2d")
                    if not h.empty:
                        price = h['Close'].iloc[-1]
                        if len(h)>1: prev = h['Close'].iloc[-2]
                if price is not None:
                    ok = True
                    l = f"\n{ts}: {price:.2f} {cur}"
                    if prev and prev>0:
                        ch = ((price-prev)/prev)*100
                        l += f" | {ch:+.2f}%"
                    lines.append(l)
                else:
                    lines.append(f"\n{ts}: Keine Daten")
            except Exception as e:
                lines.append(f"\n{ts}: Fehler")
        return "\n".join(lines) if ok else None


class NewsCollector:
    def __init__(self, api_key=""):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY", "")

    async def collect(self, keywords, language="de", max_articles=5):
        if not self.api_key:
            return None
        try:
            q = " OR ".join(keywords[:3])
            async with httpx.AsyncClient() as client:
                r = await client.get("https://newsapi.org/v2/everything",
                    params={"q":q,"language":language,"sortBy":"publishedAt",
                            "pageSize":max_articles,"apiKey":self.api_key}, timeout=10.0)
                r.raise_for_status()
                data = r.json()
            arts = data.get("articles",[])
            if not arts: return None
            lines = [f"=== NEWS ({datetime.now():%d.%m.%Y %H:%M}) === Quelle: NewsAPI | {q}\n"]
            for i,a in enumerate(arts[:max_articles],1):
                lines.append(f"{i}. [{a.get('source',{}).get('name','?')}] {a.get('title','?')}")
                d = (a.get("description") or "")[:200]
                if d: lines.append(f"   {d}")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            log.warning(f"NewsAPI: {e}")
            return None


class WebSearchCollector:
    """DuckDuckGo: 100% free, no API key, no limit."""

    async def collect(self, query, max_results=5, time_filter="w"):
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._search, query, max_results, time_filter)
        except Exception as e:
            log.warning(f"DuckDuckGo: {e}")
            return None

    def _search(self, query, max_results, time_filter):
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, timelimit=time_filter))
            if not results: return None
            lines = [f"=== WEBSUCHE ({datetime.now():%d.%m.%Y %H:%M}) === Quelle: DuckDuckGo | {query}\n"]
            for i,r in enumerate(results[:max_results],1):
                lines.append(f"{i}. {r.get('title','?')}")
                b = r.get("body","")[:250]
                if b: lines.append(f"   {b}")
                h = r.get("href","")
                if h: lines.append(f"   -> {h}")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            log.warning(f"DDG error: {e}")
            return None


# ORCHESTRATOR

class FreeDataCollector:
    def __init__(self, newsapi_key=""):
        self.weather = WeatherCollector()
        self.stocks = StockCollector()
        self.news = NewsCollector(api_key=newsapi_key)
        self.web_search = WebSearchCollector()

    async def collect(self, query):
        start = time.time()
        cat = classify_realtime_query(query)
        if cat == DataCategory.NONE:
            return CollectedData(category=DataCategory.NONE)

        result = CollectedData(category=cat)
        tasks, names = [], []

        if cat in (DataCategory.WEATHER, DataCategory.MIXED):
            cn = extract_city_name(query)
            if cn:
                loc = await geocode_city(cn)
                if loc:
                    tasks.append(self.weather.collect(loc[0], loc[1], loc[2]))
                    names.append("open-meteo")
                else:
                    result.sources_failed.append("geocode")

        if cat in (DataCategory.STOCKS, DataCategory.MIXED):
            tks = extract_tickers(query)
            if tks:
                tasks.append(self.stocks.collect(tks))
                names.append("yfinance")

        if cat in (DataCategory.NEWS, DataCategory.MIXED):
            kw = extract_news_keywords(query)
            if kw:
                tasks.append(self.news.collect(kw))
                names.append("newsapi")

        if cat == DataCategory.WEB_SEARCH:
            tasks.append(self.web_search.collect(extract_search_query(query), 5, "w"))
            names.append("duckduckgo")

        # DuckDuckGo backup for specialized categories
        if cat in (DataCategory.WEATHER, DataCategory.STOCKS, DataCategory.NEWS, DataCategory.MIXED):
            tasks.append(self.web_search.collect(extract_search_query(query), 3, "d"))
            names.append("duckduckgo-backup")

        if not tasks:
            return CollectedData(category=DataCategory.NONE)

        api_results = await asyncio.gather(*tasks, return_exceptions=True)
        parts, ddg_backup = [], None

        for name, ar in zip(names, api_results):
            result.sources_tried.append(name)
            if isinstance(ar, Exception) or ar is None:
                result.sources_failed.append(name)
            else:
                result.sources_succeeded.append(name)
                if name == "duckduckgo-backup":
                    ddg_backup = ar
                else:
                    parts.append(ar)

        if not parts and ddg_backup:
            parts.append(ddg_backup)

        if parts:
            result.success = True
            result.data_text = "\n\n".join(parts)

        result.collection_time_ms = (time.time()-start)*1000
        log.info(f"DataCollector: {cat.value} ok={result.sources_succeeded} "
                 f"fail={result.sources_failed} escalate={result.should_escalate} "
                 f"{result.collection_time_ms:.0f}ms")
        return result

    def build_enriched_prompt(self, original_query, collected):
        src = ", ".join(collected.sources_succeeded)
        return (f"ECHTZEITDATEN ({datetime.now():%d.%m.%Y %H:%M}, Quellen: {src}):\n\n"
                f"{collected.data_text}\n\n---\n\n"
                f"NUTZER-FRAGE: {original_query}\n\n"
                f"Beantworte basierend auf den Daten. Nenne Quellen. "
                f"Wenn unvollstaendig, sage das ehrlich.")


# SINGLETON
_collector = None

def get_collector(newsapi_key=""):
    global _collector
    if _collector is None:
        _collector = FreeDataCollector(newsapi_key=newsapi_key)
    return _collector
