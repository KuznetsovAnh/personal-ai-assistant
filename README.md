# Personal AI Assistant

Personal AI Assistant is a Streamlit-based assistant that combines an LLM, tool calling, lightweight memory, and realtime web retrieval. It supports Vietnamese and English conversations, with current-data tools for news, web search, weather, stock prices, exchange rates, calculations, translation, and music playback control.

## Highlights

- **Realtime answers**: uses tools for time-sensitive questions instead of relying on stale model knowledge.
- **Unified web search core**: `core/search` handles realtime search with provider fallback and response formatting.
- **Clean tool orchestration**: the LLM interprets user intent, calls the right tool, then summarizes evidence into a natural answer.
- **Local conversation memory**: stores profile facts, action states, tool ledger, and conversation history in JSON.
- **Music control**: opens, switches, pauses, resumes, and stops music from natural chat commands.
- **Streamlit UI**: multi-conversation chat interface for local testing and daily use.
- **MCP server**: exposes the same tools through Model Context Protocol.

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- .env.example
`-- core/
    |-- agent.py             # Main LLM agent, routing, and response normalization
    |-- tools.py             # Tool implementations used by the agent and MCP server
    |-- memory.py            # Conversation and long-term memory manager
    |-- models.py            # Shared response models
    |-- settings.py          # Environment-driven runtime configuration
    |-- game_registry.py     # Game metadata and source rules for banner/event questions
    |-- mcp_server.py        # FastMCP tool server
    |-- search/              # Core realtime web search backend
    |-- ui/
    |   `-- streamlit_app.py # Streamlit chat UI
    `-- utils/
        `-- url_utils.py
```

## How It Works

1. The user sends a message in Streamlit.
2. `core.agent.PersonalAssistantAgent` decides whether the request can use a direct tool path or needs full LLM reasoning.
3. For realtime information, the agent calls tools from `core.tools`.
4. Web-backed answers use the unified search backend in `core/search`.
5. The final answer is cleaned, grounded in tool output, and returned with tool-event metadata for the UI.

## Main Capabilities

- **Web search and news**: realtime search, article filtering, source links, and concise summaries.
- **Stock prices**: resolves ticker/company names and fetches market quotes from Yahoo Finance.
- **Exchange rates**: converts currency questions into ISO 4217 currency pairs and fetches live rates.
- **Weather**: current weather and forecast by location.
- **Game banners/events**: current banner and in-game event lookup for supported gacha games.
- **Memory**: saves and reuses user facts, action states, and conversation history.
- **Utilities**: date/time, calculation, translation, and local knowledge lookup.
- **Music playback**: natural-language music commands routed to dedicated music tools, including switching songs.

## Music Tools

Music commands are handled directly by the assistant for faster responses. The tool layer supports:

- `play_music(url_or_query)`: opens a song, direct URL, or search query in a dedicated browser window.
- Song switching: calling `play_music(...)` while music is active closes the current tracked music window and opens the new song.
- `stop_music(song_hint=None)`: stops the current tracked music window, optionally matching a song hint.
- `pause_music()`: pauses the current tracked music playback.
- `resume_music()`: resumes paused music playback.

Example prompts:

```text
mo bai Nang Am Xa Dan
phat nhac lo-fi cho toi
chuyen sang bai See Tinh
tam dung nhac
tiep tuc phat nhac
dung nhac
```

The music state is synchronized with local memory so the assistant can track whether music is active or paused across conversation turns.

## Installation

```bash
pip install -r requirements.txt
```

If you use `uv`:

```bash
uv pip install -r requirements.txt
```

## Configuration

Set credentials through environment variables. Do not commit API keys to the repository.
Use `.env.example` as a safe template for the variables you need in your local environment.

| Variable | Purpose |
| --- | --- |
| `ASSISTANT_API_KEY` | API key for the OpenAI-compatible LLM endpoint |
| `ASSISTANT_BASE_URL` | LLM API base URL |
| `ASSISTANT_MODEL` | Model name |
| `ASSISTANT_RUNTIME_PROFILE` | Runtime profile: `demo_fast`, `balanced`, or `accurate` |
| `ASSISTANT_MAX_RESPONSE_CHARS` | Maximum final answer length |
| `ASSISTANT_LLM_TEMPERATURE` | LLM temperature |
| `ASSISTANT_LLM_MAX_COMPLETION_TOKENS` | Maximum LLM completion tokens |
| `ASSISTANT_MEMORY_FILE` | Path to the memory JSON file |
| `ASSISTANT_DATA_DIR` | Runtime data directory |
| `ASSISTANT_SEARCH_DDG_FAILURE_THRESHOLD` | Skip DuckDuckGo after this many blocking failures |
| `ASSISTANT_SEARCH_DDG_BLOCKED_SIGNATURE` | Substring in a DDG error that counts as a block |
| `TAVILY_API_KEY` | Optional Tavily key for search fallback |
| `EXA_API_KEY` | Optional Exa key for search fallback |
| `EXCHANGERATE_API_KEY` | ExchangeRate-API key |

PowerShell example:

```powershell
$env:ASSISTANT_API_KEY="your-key"
$env:ASSISTANT_BASE_URL="https://your-llm-endpoint/v1"
$env:ASSISTANT_MODEL="your-model"
$env:EXCHANGERATE_API_KEY="your-key"
```

## Run The App

```bash
uv run streamlit run app.py
```

Or, without `uv`:

```bash
streamlit run app.py
```

## Run MCP Server

```bash
python -m core.mcp_server
```

## Runtime Data

The assistant writes memory and conversation state under `data/` by default. This directory is ignored by Git because it may contain personal data.

## Notes For Contributors

- Keep `core/search` as the single source of truth for web search behavior.
- Prefer small changes, standard-library utilities, and deleting obsolete code over adding new layers.
- Do not hardcode API keys, exchange rates, stock tickers, URLs, or answer content.
- Keep user-facing answers grounded in tool output for realtime topics.
