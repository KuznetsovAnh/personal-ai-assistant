"""Single source of truth for gacha game routing and answer profiles.

The registry deliberately keeps game-specific vocabulary in one place so the
search, extraction, validation, and render layers do not drift apart.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import urlparse


_VN_TZ = timezone(timedelta(hours=7))


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, and collapse whitespace for matching."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class GameProfile:
    key: str
    display: str
    aliases: tuple[str, ...]
    banner_label: str
    unit_label: str
    event_label: str = "Event"
    equipment_label: str | None = None
    character_banner_label: str | None = None
    equipment_banner_label: str | None = None
    support_banner_label: str | None = None
    rerun_banner_label: str | None = None
    reward_code_label: str | None = None
    server_policy: str = "global_default"
    default_server: str = "Global"
    trusted_domains: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    priority_domains: tuple[str, ...] = ()
    priority_urls: tuple[str, ...] = ()
    current_source_patterns: tuple[str, ...] = ()
    history_source_patterns: tuple[str, ...] = ()
    hub_source_patterns: tuple[str, ...] = ()
    official_source_patterns: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    query_terms: tuple[str, ...] = ()
    render_sections: tuple[str, ...] = ("current_banners", "current_events", "upcoming")
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def normalized_aliases(self) -> tuple[str, ...]:
        return tuple(normalize_text(alias) for alias in self.aliases)


GAME_PROFILES: dict[str, GameProfile] = {
    "arknights_endfield": GameProfile(
        key="arknights_endfield",
        display="Arknights: Endfield",
        aliases=("arknights endfield", "endfield", "ak endfield"),
        banner_label="Headhunting",
        character_banner_label="Featured Character Banner",
        equipment_banner_label="Equipment Headhunting",
        unit_label="Character",
        equipment_label="Equipment / Weapon",
        event_label="Event / Drill / Resource Track",
        server_policy="strict_spin_off",
        trusted_domains=("gryphline.com", "game8.co", "lootbar.com", "prydwen.gg", "wiki.gg", "ldshop.gg", "buffhub.com"),
        priority_domains=("game8.co", "lootbar.com", "prydwen.gg"),
        current_source_patterns=(
            "gryphline.com/en-us/arknights-endfield/news",
            "game8.co/games/Arknights-Endfield/archives",
            "prydwen.gg/arknights-endfield/guides/current",
            "buffhub.com/arknights-endfield",
        ),
        history_source_patterns=("banner-history", "banner_history"),
        hub_source_patterns=("gryphline.com/en-us/arknights-endfield", "game8.co/games/Arknights-Endfield", "prydwen.gg/arknights-endfield", "buffhub.com/arknights-endfield"),
        official_source_patterns=("gryphline.com",),
        forbidden_terms=("arknights.wiki.gg", "arknights.global"),
        query_terms=("featured character banner", "equipment headhunting", "version", "event", "resource track"),
        render_sections=("version", "current_banners", "equipment_banners", "current_events", "upcoming"),
    ),
    "arknights": GameProfile(
        key="arknights",
        display="Arknights",
        aliases=("arknights",),
        banner_label="Headhunting",
        character_banner_label="Event Headhunting / Standard Pool / Kernel Headhunting",
        unit_label="Operator",
        event_label="Event",
        trusted_domains=("arknights.wiki.gg", "arknights.global", "oldwell.info", "gamepress.gg", "game8.co", "lootbar.com", "prydwen.gg", "fandom.com"),
        priority_domains=("arknights.wiki.gg", "oldwell.info", "game8.co", "lootbar.com", "prydwen.gg"),
        priority_urls=(
            "https://arknights.wiki.gg/wiki/Headhunting/Banners",
            "https://arknights.wiki.gg/wiki/Event",
            "https://arknights.wiki.gg/wiki/Event/Upcoming",
            "https://oldwell.info/",
        ),
        current_source_patterns=(
            "arknights.wiki.gg/wiki/headhunting/banners",
            "arknights.wiki.gg/wiki/event",
            "oldwell.info",
            "arknights.global/news",
        ),
        history_source_patterns=("banner-history", "history"),
        hub_source_patterns=("arknights.wiki.gg/wiki/main_page",),
        official_source_patterns=("arknights.global",),
        forbidden_terms=("endfield", "arknights endfield"),
        query_terms=("current headhunting", "standard pool", "kernel headhunting", "event"),
        render_sections=("current_banners", "standard_kernel", "current_events", "upcoming"),
    ),
    "hsr": GameProfile(
        key="hsr",
        display="Honkai Star Rail",
        aliases=("honkai star rail", "star rail", "hsr", "honkai"),
        banner_label="Warp",
        character_banner_label="Character Event Warp",
        equipment_banner_label="Light Cone Event Warp",
        unit_label="Character",
        equipment_label="Light Cone",
        event_label="Event",
        trusted_domains=("hoyolab.com", "hoyoverse.com", "hsr.hoyoverse.com", "game8.co", "lootbar.com", "prydwen.gg", "fandom.com"),
        priority_domains=("game8.co", "lootbar.com", "prydwen.gg"),
        current_source_patterns=("hsr.hoyoverse.com/en-us/news", "game8.co/games/Honkai-Star-Rail/archives", "prydwen.gg/star-rail/guides/current"),
        history_source_patterns=("prydwen.gg/star-rail/guides/banner-history", "banner-history", "warp-history"),
        hub_source_patterns=("hsr.hoyoverse.com/en-us", "game8.co/games/Honkai-Star-Rail", "prydwen.gg/star-rail"),
        official_source_patterns=("hsr.hoyoverse.com", "hoyolab.com"),
        query_terms=("character event warp", "light cone event warp", "version", "phase", "events"),
        render_sections=("version", "current_banners", "equipment_banners", "current_events", "upcoming"),
    ),
    "genshin": GameProfile(
        key="genshin",
        display="Genshin Impact",
        aliases=("genshin impact", "genshin"),
        banner_label="Wish",
        character_banner_label="Character Event Wish",
        equipment_banner_label="Epitome Invocation",
        unit_label="Character",
        equipment_label="Weapon",
        event_label="Event",
        trusted_domains=("hoyolab.com", "hoyoverse.com", "genshin.hoyoverse.com", "game8.co", "lootbar.com", "prydwen.gg", "ign.com", "eurogamer.net", "fandom.com"),
        priority_domains=("game8.co", "lootbar.com", "prydwen.gg"),
        current_source_patterns=("genshin.hoyoverse.com/en/news", "game8.co/games/Genshin-Impact/archives", "prydwen.gg/genshin/guides/current"),
        history_source_patterns=("banner-history", "wish-history", "wish_history"),
        hub_source_patterns=("genshin.hoyoverse.com/en", "game8.co/games/Genshin-Impact", "prydwen.gg/genshin"),
        official_source_patterns=("genshin.hoyoverse.com", "hoyolab.com"),
        query_terms=("character event wish", "epitome invocation", "version", "phase", "events"),
        render_sections=("version", "current_banners", "equipment_banners", "current_events", "upcoming"),
    ),
    "wuthering_waves": GameProfile(
        key="wuthering_waves",
        display="Wuthering Waves",
        aliases=("wuthering waves", "wuthering", "wuwa"),
        banner_label="Convene",
        character_banner_label="Featured Resonator Convene",
        equipment_banner_label="Weapon Convene",
        unit_label="Resonator",
        equipment_label="Weapon",
        event_label="Event",
        trusted_domains=("wutheringwaves.kurogames.com", "kurogames.com", "prydwen.gg", "game8.co", "lootbar.com", "wutheringwaves.wiki.gg", "lootbar.gg", "ldshop.gg"),
        priority_domains=("game8.co", "lootbar.com", "prydwen.gg"),
        current_source_patterns=("wutheringwaves.kurogames.com/en/main/news", "wutheringwaves.wiki.gg/wiki/convene", "game8.co/games/Wuthering-Waves/archives", "prydwen.gg/wuthering-waves/guides/current"),
        history_source_patterns=("prydwen.gg/wuthering-waves/guides/banner-history", "banner-history", "convene-history", "convene/history"),
        hub_source_patterns=("wutheringwaves.kurogames.com", "wutheringwaves.wiki.gg", "game8.co/games/Wuthering-Waves", "prydwen.gg/wuthering-waves"),
        official_source_patterns=("wutheringwaves.kurogames.com", "kurogames.com"),
        query_terms=("convene", "resonator banner", "weapon banner", "collab", "events"),
        render_sections=("version", "current_banners", "equipment_banners", "current_events", "upcoming"),
    ),
    "zzz": GameProfile(
        key="zzz",
        display="Zenless Zone Zero",
        aliases=("zenless zone zero", "zenless", "zzz"),
        banner_label="Signal Search",
        character_banner_label="Exclusive Channel",
        rerun_banner_label="Rerun Channel",
        equipment_banner_label="W-Engine Channel",
        unit_label="Agent",
        equipment_label="W-Engine",
        event_label="Event",
        trusted_domains=("zenless.hoyoverse.com", "hoyoverse.com", "game8.co", "lootbar.com", "prydwen.gg", "fandom.com", "lootbar.gg", "ldshop.gg"),
        priority_domains=("game8.co", "lootbar.com", "prydwen.gg"),
        current_source_patterns=("zenless.hoyoverse.com/en-us/news", "game8.co/games/Zenless-Zone-Zero/archives", "prydwen.gg/zenless/guides/current"),
        history_source_patterns=("banner-history", "signal-search-history", "signal_search_history"),
        hub_source_patterns=("zenless.hoyoverse.com/en-us", "game8.co/games/Zenless-Zone-Zero", "prydwen.gg/zenless"),
        official_source_patterns=("zenless.hoyoverse.com", "hoyolab.com"),
        query_terms=("exclusive channel", "rerun channel", "w-engine", "agent", "events"),
        render_sections=("version", "current_banners", "equipment_banners", "story_unlocks", "current_events", "upcoming"),
    ),
    "fgo": GameProfile(
        key="fgo",
        display="Fate/Grand Order",
        aliases=("fate grand order", "fate/grand order", "fgo"),
        banner_label="Summoning Campaign",
        character_banner_label="Pickup Summon",
        unit_label="Servant",
        equipment_label="Craft Essence",
        event_label="Campaign / Event",
        server_policy="split_na_jp_if_unspecified",
        default_server="NA",
        trusted_domains=("fate-go.us", "fategrandorder.fandom.com", "grandorder.gamepress.gg", "gamepress.gg", "reddit.com", "game8.co"),
        priority_domains=("fategrandorder.fandom.com", "fate-go.us", "grandorder.gamepress.gg"),
        priority_urls=(
            "https://fategrandorder.fandom.com/wiki/Event_List_(US)/Upcoming_Events",
            "https://fate-go.us/",
            "https://grandorder.gamepress.gg/",
        ),
        current_source_patterns=("fate-go.us", "fategrandorder.fandom.com/wiki/event_list_(us)/upcoming_events", "grandorder.gamepress.gg"),
        history_source_patterns=("fategrandorder.fandom.com/wiki/summon_banner_list", "banner_history", "history"),
        hub_source_patterns=("fategrandorder.fandom.com/wiki/main_page", "grandorder.gamepress.gg"),
        official_source_patterns=("fate-go.us",),
        query_terms=("pickup summon", "summoning campaign", "event list", "anniversary", "gssr"),
        render_sections=("server_sections",),
    ),
    "blue_archive": GameProfile(
        key="blue_archive",
        display="Blue Archive",
        aliases=("blue archive", "bluearchive"),
        banner_label="Recruitment",
        character_banner_label="Pick-Up Recruitment",
        unit_label="Student",
        event_label="Campaign / Event",
        server_policy="global_default_jp_on_request",
        trusted_domains=("ba.joexyz.online", "bluearchive.wiki", "bluearchive.fandom.com", "bluearchive.nexon.com", "joeschmoe.io", "fandom.com", "twitter.com", "x.com", "facebook.com"),
        priority_domains=("ba.joexyz.online", "bluearchive.wiki", "bluearchive.nexon.com"),
        priority_urls=(
            "https://ba.joexyz.online/global/banners",
            "https://bluearchive.wiki/wiki/Events",
            "https://bluearchive.wiki/wiki/Recruitment_(Gacha)",
            "https://bluearchive.wiki/wiki/Main_Page",
        ),
        current_source_patterns=("ba.joexyz.online/global/banners", "bluearchive.wiki/wiki/events", "bluearchive.nexon.com/events", "bluearchive.nexon.com/news"),
        history_source_patterns=("history",),
        hub_source_patterns=("bluearchive.wiki/wiki/main_page",),
        official_source_patterns=("bluearchive.nexon.com",),
        query_terms=("recruitment", "banner", "student", "campaign", "global"),
        render_sections=("current_banners", "upcoming", "current_events", "real_world_events"),
    ),
    "nikke": GameProfile(
        key="nikke",
        display="Goddess of Victory: NIKKE",
        aliases=("goddess of victory nikke", "goddess of victory: nikke", "nikke"),
        banner_label="Recruit",
        character_banner_label="Special Recruit / Pick Up Recruit",
        rerun_banner_label="Limited Select Recruit",
        unit_label="Nikke",
        event_label="Event",
        reward_code_label="CD-Key",
        trusted_domains=("nikke-en.com", "nikke.gg", "dotgg.gg/nikke", "prydwen.gg", "nikke-goddess-of-victory-international.fandom.com", "fandom.com", "lootandwaifus.com", "twitter.com", "x.com"),
        priority_domains=("nikke.gg", "dotgg.gg", "nikke-en.com", "lootandwaifus.com", "nikke-goddess-of-victory-international.fandom.com"),
        priority_urls=(
            "https://nikke.gg/news/",
            "https://dotgg.gg/nikke",
            "https://nikke-goddess-of-victory-international.fandom.com/wiki/Event",
            "https://nikke-en.com/",
            "https://lootandwaifus.com/nikke-banner-history/",
        ),
        current_source_patterns=("nikke.gg/news/", "dotgg.gg/nikke", "nikke-en.com", "fandom.com/wiki/event"),
        history_source_patterns=("lootandwaifus.com/nikke-banner-history", "banner-history", "history", "archive"),
        hub_source_patterns=("nikke.gg/news/", "dotgg.gg/nikke"),
        official_source_patterns=("nikke-en.com",),
        query_terms=("special recruit", "pick up recruit", "event", "cd-key", "reward code"),
        render_sections=("current_banners", "current_events", "reward_codes", "upcoming"),
    ),
    "uma_musume": GameProfile(
        key="uma_musume",
        display="Uma Musume Pretty Derby",
        aliases=("uma musume pretty derby", "uma musume", "umamusume"),
        banner_label="Scout / Gacha",
        character_banner_label="Trainee Character Gacha",
        support_banner_label="Support Card Gacha",
        unit_label="Trainee",
        equipment_label="Support Card",
        event_label="Story Event / Campaign",
        server_policy="global_default_jp_on_request",
        trusted_domains=("gametora.com", "umamusume.com", "cygames.co.jp", "game8.co", "lootbar.com", "prydwen.gg", "lootbar.gg", "x.com", "twitter.com"),
        priority_domains=("game8.co", "lootbar.com", "prydwen.gg", "gametora.com"),
        current_source_patterns=("gametora.com/umamusume", "umamusume.com/news", "game8.co/games/Umamusume-Pretty-Derby/archives", "prydwen.gg/umamusume/guides/current"),
        history_source_patterns=("banner-history", "gacha-history", "scout-history"),
        hub_source_patterns=("gametora.com/umamusume", "umamusume.com", "game8.co/games/Umamusume-Pretty-Derby", "prydwen.gg/umamusume"),
        official_source_patterns=("umamusume.com", "cygames.co.jp"),
        query_terms=("trainee character gacha", "support card gacha", "story event", "roadmap"),
        render_sections=("current_banners", "support_card_banners", "current_events", "upcoming"),
    ),
}


GACHA_INTENT_TERMS = (
    "banner", "banners", "gacha", "schedule", "current", "upcoming", "next",
    "event", "events", "warp", "wish", "convene", "summon", "summoning",
    "recruit", "recruitment", "headhunting", "pickup", "pick up", "rate up",
    "light cone", "w-engine", "w engine", "weapon", "support card", "scout",
    "code", "reward", "cd-key", "phase", "version", "patch", "update",
    "lich banner", "su kien", "hien tai", "dang dien ra", "dang chay", "sap toi",
    "ma code", "phan thuong", "moi nhat", "tin moi", "phien ban", "cap nhat", "ban cap nhat",
)

BLOCKED_GACHA_DOMAINS = (
    "gamek.vn", "gamelade.vn", "genk.vn", "kenh14.vn", "thanhnien.vn",
    "tuoitre.vn", "vnexpress.net", "dantri.com.vn", "vietnamnet.vn",
    "gamehub.vn", "wattpad.com",
)


def _iter_alias_matches(normalized: str) -> Iterable[tuple[int, GameProfile, str]]:
    for profile in GAME_PROFILES.values():
        for alias in profile.normalized_aliases:
            if alias and alias in normalized:
                yield (len(alias), profile, alias)


def detect_game(text: str) -> GameProfile | None:
    normalized = normalize_text(text)
    if not normalized:
        return None

    # Strict spin-off handling: Endfield must win before plain Arknights.
    if any(alias in normalized for alias in GAME_PROFILES["arknights_endfield"].normalized_aliases):
        excluded_endfield = re.search(
            r"\b(khong lay|khong phai|khong|not|exclude|without|no)\s+(arknights\s+)?endfield\b",
            normalized,
        )
        if excluded_endfield and "arknights" in normalized:
            return GAME_PROFILES["arknights"]
        return GAME_PROFILES["arknights_endfield"]

    matches = sorted(_iter_alias_matches(normalized), key=lambda item: item[0], reverse=True)
    if not matches:
        return None
    return matches[0][1]


def detect_server(text: str, profile: GameProfile) -> str:
    normalized = normalize_text(text)
    if re.search(r"\b(jp|japan|japanese|nhat ban|server nhat)\b", normalized):
        return "JP"
    if re.search(r"\b(na|north america|us|usa|en server)\b", normalized):
        return "NA"
    if re.search(r"\b(cn|china|trung quoc|server trung)\b", normalized):
        return "CN"
    if re.search(r"\b(global|en|english|global/en)\b", normalized):
        return "Global"
    return profile.default_server


def resolve_server_for_query(text: str, profile: GameProfile) -> str:
    normalized = normalize_text(text)
    explicit_server = re.search(
        r"\b(jp|japan|japanese|nhat ban|server nhat|na|north america|us|usa|en server|cn|china|trung quoc|server trung|global|en|english)\b",
        normalized,
    )
    if profile.server_policy == "split_na_jp_if_unspecified" and not explicit_server:
        return "NA+JP"
    return detect_server(text, profile)


def detect_intents(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    intents: list[str] = []
    if any(term in normalized for term in ("banner", "gacha", "warp", "wish", "convene", "summon", "recruit", "headhunting", "scout")):
        intents.append("current_banner")
    if any(term in normalized for term in ("event", "su kien", "campaign")):
        intents.append("current_event")
    if any(term in normalized for term in ("upcoming", "next", "sap toi", "roadmap", "phase 2", "coming")):
        intents.append("upcoming")
    if any(term in normalized for term in ("code", "cd-key", "reward", "phan thuong", "qua")):
        intents.append("reward")
    if any(term in normalized for term in ("support card", "light cone", "w-engine", "w engine", "weapon", "equipment")):
        intents.append("equipment")
    if not intents and any(term in normalized for term in ("version", "patch", "update", "phien ban", "cap nhat", "ban cap nhat", "tin moi", "moi nhat")):
        intents.append("current_event")
    if not intents and any(term in normalized for term in ("current", "hien tai", "dang dien ra", "dang chay")):
        intents.extend(["current_banner", "current_event"])
    return tuple(dict.fromkeys(intents or ["current_banner", "current_event"]))


def build_search_queries(text: str, *, now: datetime | None = None) -> list[str]:
    profile = detect_game(text)
    if not profile:
        return []
    now = now or datetime.now(_VN_TZ)
    month_year = now.strftime("%B %Y")
    year = now.strftime("%Y")
    server = resolve_server_for_query(text, profile)
    intents = detect_intents(text)

    base = f"{profile.display} {server}".strip()
    queries: list[str] = []
    priority_terms: list[str] = []
    if "current_banner" in intents:
        priority_terms.extend(("current", profile.banner_label, "banner", "schedule"))
    if "current_event" in intents:
        priority_terms.extend(("current", "events", "schedule"))
    if "upcoming" in intents:
        priority_terms.extend(("upcoming", "roadmap"))
    if "reward" in intents:
        priority_terms.extend(("active codes", "redeem", "gift", "coupon", "CD-Key", "reward", "code"))
    if not priority_terms:
        priority_terms.extend(("current", "banner", "events"))
    priority_text = " ".join(dict.fromkeys(priority_terms))

    for domain in profile.priority_domains[:4]:
        clean_domain = domain.strip().lower()
        if clean_domain:
            queries.append(f"site:{clean_domain} {base} {priority_text} {month_year}")

    if "current_banner" in intents:
        queries.append(f"{base} current {profile.banner_label} banner schedule {month_year}")
        if profile.character_banner_label:
            queries.append(f"{base} current {profile.character_banner_label} {year}")
    if "equipment" in intents or "current_banner" in intents:
        if profile.equipment_banner_label:
            queries.append(f"{base} current {profile.equipment_banner_label} {month_year}")
        if profile.support_banner_label:
            queries.append(f"{base} current {profile.support_banner_label} {month_year}")
    if "current_event" in intents:
        queries.append(f"{base} current events schedule {month_year}")
    if "upcoming" in intents:
        queries.append(f"{base} upcoming banners events roadmap {month_year}")
    if "reward" in intents:
        queries.append(f"{base} active redeem codes gift code coupon CD-Key {month_year}")
        queries.append(f"{base} current reward code event {month_year}")
        if profile.key == "nikke":
            queries.append(f"{base} NIKKE CD-Key redeem coupon active codes {month_year}")

    if profile.query_terms:
        queries.append(f"{base} {' '.join(profile.query_terms[:4])} {month_year}")

    # Add a broad but English normalized query as fallback.
    queries.append(f"{base} current banner and events {year}")

    deduped: list[str] = []
    seen = set()
    for query in queries:
        clean = re.sub(r"\s+", " ", query).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            deduped.append(clean)
    return deduped[:5]


def is_blocked_domain(url: str) -> bool:
    from urllib.parse import urlparse

    domain = urlparse(url or "").netloc.lower().replace("www.", "")
    if not domain:
        return False
    if domain.endswith(".vn"):
        return True
    return any(domain == blocked or domain.endswith("." + blocked) for blocked in BLOCKED_GACHA_DOMAINS)


def _source_pattern_matches(pattern: str, url: str, source: str = "") -> bool:
    pattern = (pattern or "").strip().lower().replace("www.", "")
    if not pattern:
        return False

    parsed_pattern = urlparse(pattern if "://" in pattern else f"https://{pattern}")
    pattern_host = (parsed_pattern.netloc or "").lower().replace("www.", "")
    pattern_path = (parsed_pattern.path or "").lower().rstrip("/")
    if not pattern_host:
        return False

    parsed_url = urlparse(url or "")
    url_host = (parsed_url.netloc or "").lower().replace("www.", "")
    url_path = (parsed_url.path or "").lower().rstrip("/")
    host_matches = bool(url_host and (url_host == pattern_host or url_host.endswith(f".{pattern_host}")))
    if host_matches and pattern_path:
        return url_path == pattern_path or url_path.startswith(f"{pattern_path}/")
    if host_matches:
        return True

    source_lower = (source or "").lower().replace("www.", "")
    return not pattern_path and pattern_host in source_lower


def source_tier(profile: GameProfile, url: str, source: str = "") -> int:
    """Return a small source tier score: 3 high, 2 medium, 1 low, 0 blocked."""
    if is_blocked_domain(url):
        return 0
    haystack = f"{url or ''} {source or ''}".lower()
    if any(_source_pattern_matches(domain, url, source) for domain in profile.trusted_domains):
        return 3
    if any(domain in haystack for domain in ("wiki", "fandom.com", "game8.co", "prydwen.gg", "official", "hoyoverse", "kurogames")):
        return 2
    if any(domain in haystack for domain in ("lootbar", "ldshop", "reddit", "youtube", "x.com", "twitter", "facebook")):
        return 1
    return 1


def source_kind(profile: GameProfile, url: str, source: str = "") -> str:
    """Classify a gacha source without creating another registry layer."""
    if any(_source_pattern_matches(pattern, url, source) for pattern in profile.history_source_patterns):
        return "history"
    if any(_source_pattern_matches(pattern, url, source) for pattern in profile.current_source_patterns):
        return "current"
    if any(_source_pattern_matches(pattern, url, source) for pattern in profile.official_source_patterns):
        return "official"
    if any(_source_pattern_matches(pattern, url, source) for pattern in profile.hub_source_patterns):
        return "hub"

    haystack = normalize_text(f"{url or ''} {source or ''}")
    if any(term in haystack for term in ("banner history", "banner-history", "history")):
        return "history"
    if any(term in haystack for term in ("current", "ongoing", "schedule")):
        return "current"
    return "generic"


def wrong_game_penalty(profile: GameProfile, text: str) -> bool:
    normalized = normalize_text(text)
    if profile.key == "arknights" and "endfield" in normalized:
        return True
    if profile.key == "arknights_endfield" and "arknights.wiki.gg/wiki/headhunting" in normalized:
        return True
    if profile.key == "nikke" and "dotgg.gg/" in normalized and "dotgg.gg/nikke" not in normalized:
        return True
    has_target_alias = any(alias and alias in normalized for alias in profile.normalized_aliases)
    if not has_target_alias:
        for other in GAME_PROFILES.values():
            if other.key == profile.key:
                continue
            if any(alias and len(alias) > 3 and alias in normalized for alias in other.normalized_aliases):
                return True
    return any(normalize_text(term) in normalized for term in profile.forbidden_terms)
