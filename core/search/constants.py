from typing import Dict, List, Set, Tuple


CACHE_TTL_SECONDS = 3600
MAX_CACHE_SIZE = 1000


SEARCH_TOPICS: Dict[str, Dict[str, List[str]]] = {
    "gaming": {
        "keywords": ['game', 'patch', 'banner', 'update', 'release date', 'roadmap', 'leak', 'speculation', 'gacha', 'reroll', 'tier list', 'build', 'nhân vật', 'honkai', 'hsr', 'star rail', 'genshin', 'zzz', 'zenless', 'wuwa', 'wuthering waves', 'arknights', 'fgo', 'phiên bản', 'sự kiện'],
        "suffixes": ["update", "release date", "patch notes", "roadmap", "leaks", "speculation", "official", "tin tức"]
    },
    "tech": {
        "keywords": ['tech', 'công nghệ', 'ai', 'ios', 'android', 'app', 'software', 'hardware', 'card màn hình', 'cpu', 'laptop', 'phone'],
        "suffixes": ["review", "release date", "news", "vs", "benchmark", "specs", "đánh giá", "tin tức"]
    },
    "science": {
        "keywords": ['science', 'khoa học', 'space', 'vũ trụ', 'nasa', 'discovery', 'research', 'nghiên cứu', 'y tế'],
        "suffixes": ["new discovery", "latest research", "breakthrough", "study finds", "công bố", "nghiên cứu mới"]
    },
    "finance": {
        "keywords": ['finance', 'tài chính', 'stock', 'cổ phiếu', 'market', 'thị trường', 'investment', 'đầu tư', 'economy', 'kinh tế', 'lãi suất', 'ngân hàng'],
        "suffixes": ["stock price", "market analysis", "forecast", "news", "earnings report", "phân tích", "dự báo"]
    },
    "movies_tv": {
        "keywords": ['movie', 'phim', 'tv show', 'series', 'netflix', 'disney+', 'trailer', 'actor', 'diễn viên', 'đạo diễn', 'lịch chiếu'],
        "suffixes": ["review", "release date", "trailer", "cast", "ending explained", "season 2", "lịch chiếu phim", "đánh giá"]
    },
    "anime_manga": {
        "keywords": ['anime', 'manga', 'light novel', 'manhwa', 'manhua', 'chapter', 'episode', 'season', 'ova', 'phần mới', 'tập mới'],
        "suffixes": ["release date", "new season", "chapter review", "discussion", "spoiler", "tin tức anime"]
    },
    "sports": {
        "keywords": ['sports', 'thể thao', 'bóng đá', 'football', 'basketball', 'tennis', 'cầu lông', 'f1', 'đội tuyển', 'cầu thủ', 'trận đấu'],
        "suffixes": ["match result", "highlights", "live score", "news", "transfer", "lịch thi đấu", "kết quả"]
    },
    "music": {
        "keywords": ['music', 'âm nhạc', 'bài hát', 'ca sĩ', 'album', 'mv', 'concert', 'lyrics', 'lời bài hát', 'spotify', 'apple music'],
        "suffixes": ["new song", "album review", "music video", "tour dates", "lyrics meaning", "bài hát mới"]
    },
    "celebrity_gossip": {
        "keywords": ['celebrity', 'người nổi tiếng', 'showbiz', 'tin đồn', 'scandal', 'drama', 'diễn viên', 'ca sĩ'],
        "suffixes": ["scandal", "news", "gossip", "drama", "phốt", "tin đồn"]
    },
    "general": {
        "keywords": [],
        "suffixes": ["news", "latest", "update", "information", "tin tức", "thông tin", "mới nhất"]
    }
}

SEARCH_CACHE_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "at", "is", "are", "be",
    "toi", "la", "va", "cua", "cho", "ve", "trong", "tai", "duoc", "khong", "nao", "gi", "bao", "khi",
    "news", "information", "thong", "tin", "xem", "hoi", "giup",
}

SEARCH_CACHE_PHRASE_ALIASES: List[Tuple[str, str]] = [
    ("moi nhat", "latest"),
    ("hien tai", "current"),
    ("cap nhat", "update"),
    ("khi nao", "when"),
    ("bao gio", "when"),
    ("ket thuc", "end"),
    ("thoi gian", "schedule"),
    ("lich", "schedule"),
]

SEARCH_CACHE_TOKEN_ALIASES: Dict[str, str] = {
    "hsr": "honkai_star_rail",
    "starrail": "honkai_star_rail",
    "banner": "banner",
    "latest": "latest",
    "current": "current",
    "update": "update",
    "patch": "patch",
    "schedule": "schedule",
}
