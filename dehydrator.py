# ============================================================
# Module: Dehydration & Auto-tagging (dehydrator.py)
# 模块：数据脱水压缩 + 自动打标
#
# Capabilities:
# 能力：
#   1. Dehydrate: compress memory content into high-density summaries (save tokens)
#      脱水：将记忆桶的原始内容压缩为高密度摘要，省 token
#   2. Merge: blend old and new content, keeping bucket size constant
#      合并：揉合新旧内容，控制桶体积恒定
#   3. Analyze: auto-analyze content for domain/emotion/tags
#      打标：自动分析内容，输出主题域/情感坐标/标签
#
# Operating modes:
# 工作模式：
#   - API only: OpenAI-compatible API (DeepSeek/Ollama/LM Studio/vLLM/Gemini etc.)
#     仅 API：通过 OpenAI 兼容客户端调用 LLM API
#   - Dehydration cache: SQLite persistent cache to avoid redundant API calls
#     脱水缓存：SQLite 持久缓存，避免重复调用 API
#
# Depended on by: server.py
# 被谁依赖：server.py
# ============================================================


import os
import re
import json
import hashlib
import sqlite3
import logging

from openai import AsyncOpenAI

from utils import count_tokens_approx

logger = logging.getLogger("ombre_brain.dehydrator")


# --- Dehydration prompt: instructs cheap LLM to compress information ---
# --- 脱水提示词：指导廉价 LLM 压缩信息 ---
DEHYDRATE_PROMPT = """You are an information compression expert. Dehydrate the following content into a compact summary.

Compression rules:
1. Extract all core facts, remove redundant filler and repetition
2. Preserve the latest emotional state and attitudes
3. Preserve all todos / unfinished items
4. Key numbers, dates, and names must be retained
5. Target compression ratio > 70%

Output format (pure JSON, nothing else):
{
  "core_facts": ["fact 1", "fact 2"],
  "emotion_state": "current emotion keywords",
  "todos": ["todo 1", "todo 2"],
  "keywords": ["keyword 1", "keyword 2"],
  "summary": "Core summary in under 50 words"
}"""


# --- Diary digest prompt: split daily notes into independent memory entries ---
# --- 日记整理提示词：把一大段日常拆分成多个独立记忆条目 ---
DIGEST_PROMPT = """You are a diary organiser. The user will send a block of text about their day (possibly messy). Split it into multiple independent memory entries.

Organisation rules:
1. Each entry should cover one independent topic/event (do not mix topics)
2. Auto-analyse metadata for each entry
3. Remove meaningless filler and duplicate info; keep core content
4. Scattered info on the same topic should be merged into one entry
5. If there are todos, extract them as a separate entry
6. Each entry's content should be at least 50 words; merge overly short fragments into the most relevant entry
7. Keep total entries between 2-6 to avoid over-fragmentation
8. In content, mark proper nouns (people, places, products) with [[wikilinks]] (e.g. [[Tasha]], [[Obsidian]]); do not mark common words

Output format (pure JSON array, nothing else):
[
  {
    "name": "Entry title (under 10 words)",
    "content": "Organised content",
    "domain": ["domain1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["core word 1", "core word 2", "expanded word 1", "expanded word 2"],
    "importance": 5
  }
]

Tag generation rules: first extract 3-5 precise core words from the original text, then expand with 5-8 semantically related words (synonyms, hypernyms, related scenario words). Merge into one array.

Available domains (pick the most accurate 1-2, only pick truly relevant ones):
  Daily life: ["food", "fashion", "travel", "home", "shopping"]
  Relationships: ["family", "romance", "friendship", "social"]
  Growth: ["work", "study", "exams", "career"]
  Wellbeing: ["health", "psychology", "sleep", "exercise"]
  Interests: ["gaming", "film & TV", "music", "reading", "creative", "crafts"]
  Digital: ["coding", "AI", "hardware", "networking"]
  Affairs: ["finance", "planning", "todos"]
  Inner world: ["emotions", "memories", "dreams", "reflection"]
importance: 1-10, judge based on content significance
valence: 0~1 (0=negative, 0.5=neutral, 1=positive)
arousal: 0~1 (0=calm, 0.5=normal, 1=intense)"""


# --- Merge prompt: instruct LLM to blend old and new memories ---
# --- 合并提示词：指导 LLM 揉合新旧记忆 ---
MERGE_PROMPT = """You are an information merging expert. Merge the old memory with the new content into one unified, concise record.

Merging rules:
1. When new content conflicts with old memory, the new content takes precedence
2. Remove duplicate information
3. Preserve all important facts
4. Total length should not exceed 120% of the old memory
5. Mark proper nouns (people, places, products) with [[wikilinks]] (e.g. [[Tasha]], [[Obsidian]]); do not mark common words

Output the merged text directly, with no extra explanation."""


# --- Auto-tagging prompt: analyze content for domain and emotion coords ---
# --- 自动打标提示词：分析内容的主题域和情感坐标 ---
ANALYZE_PROMPT = """You are a content analyser. Analyse the following text and output structured metadata.

Analysis rules:
1. domain: pick the most accurate 1-2, only pick truly relevant ones
   Daily life: ["food", "fashion", "travel", "home", "shopping"]
   Relationships: ["family", "romance", "friendship", "social"]
   Growth: ["work", "study", "exams", "career"]
   Wellbeing: ["health", "psychology", "sleep", "exercise"]
   Interests: ["gaming", "film & TV", "music", "reading", "creative", "crafts"]
   Digital: ["coding", "AI", "hardware", "networking"]
   Affairs: ["finance", "planning", "todos"]
   Inner world: ["emotions", "memories", "dreams", "reflection"]
2. valence: 0.0~1.0, 0=extremely negative -> 0.5=neutral -> 1.0=extremely positive
3. arousal: 0.0~1.0, 0=very calm -> 0.5=normal -> 1.0=very intense
4. tags: generate in two steps, merge into one array:
   Step 1 — precise extraction: pull 3-5 true core words from the text, no generalising, no omissions
   Step 2 — expansion: add 8-10 semantically related words for the current context, including synonyms, hypernyms, related scenario words, words the user might search with different phrasing
   Merge both steps into one tags array, 10-15 total
5. suggested_name: a short title under 10 words
6. Do not use [[]] wikilink markup in tags or suggested_name

Output format (pure JSON, nothing else):
{
  "domain": ["domain1", "domain2"],
  "valence": 0.7,
  "arousal": 0.4,
  "tags": ["core word 1", "core word 2", "expanded word 1", "expanded word 2", "..."],
  "suggested_name": "Short title"
}"""


class Dehydrator:
    """
    Data dehydrator + content analyzer.
    Three capabilities: dehydration / merge / auto-tagging (domain + emotion).
    API-only: every public method requires a working LLM API.
    If the API is unavailable, methods raise RuntimeError so callers can
    surface the failure to the user instead of silently producing low-quality results.
    数据脱水器 + 内容分析器。
    三大能力：脱水压缩 / 新旧合并 / 自动打标。
    仅走 API：API 不可用时直接抛出 RuntimeError，调用方明确感知。
    （根据 BEHAVIOR_SPEC.md 三、降级行为表决策：无本地降级）
    """

    def __init__(self, config: dict):
        # --- Read dehydration API config / 读取脱水 API 配置 ---
        dehy_cfg = config.get("dehydration", {})
        self.api_key = dehy_cfg.get("api_key", "")
        self.model = dehy_cfg.get("model", "deepseek-chat")
        self.base_url = dehy_cfg.get("base_url", "https://api.deepseek.com/v1")
        self.max_tokens = dehy_cfg.get("max_tokens", 1024)
        self.temperature = dehy_cfg.get("temperature", 0.1)

        # --- API availability / 是否有可用的 API ---
        self.api_available = bool(self.api_key)

        # --- Initialize OpenAI-compatible client ---
        # --- 初始化 OpenAI 兼容客户端 ---
        if self.api_available:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=60.0,
            )
        else:
            self.client = None

        # --- SQLite dehydration cache ---
        # --- SQLite 脱水缓存：content hash → summary ---
        db_path = os.path.join(config["buckets_dir"], "dehydration_cache.db")
        self.cache_db_path = db_path
        self._init_cache_db()

    def _init_cache_db(self):
        """Create dehydration cache table if not exists."""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dehydration_cache (
                content_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

    def _get_cached_summary(self, content: str) -> str | None:
        """Look up cached dehydration result by content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        conn = sqlite3.connect(self.cache_db_path)
        row = conn.execute(
            "SELECT summary FROM dehydration_cache WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _set_cached_summary(self, content: str, summary: str):
        """Store dehydration result in cache."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO dehydration_cache (content_hash, summary, model) VALUES (?, ?, ?)",
            (content_hash, summary, self.model)
        )
        conn.commit()
        conn.close()

    def invalidate_cache(self, content: str):
        """Remove cached summary for specific content (call when bucket content changes)."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("DELETE FROM dehydration_cache WHERE content_hash = ?", (content_hash,))
        conn.commit()
        conn.close()

    # ---------------------------------------------------------
    # Dehydrate: compress raw content into concise summary
    # 脱水：将原始内容压缩为精简摘要
    # API only (no local fallback)
    # 仅通过 API 脱水（无本地回退）
    # ---------------------------------------------------------
    async def dehydrate(self, content: str, metadata: dict = None) -> str:
        """
        Dehydrate/compress memory content.
        Returns formatted summary string ready for Claude context injection.
        Uses SQLite cache to avoid redundant API calls.
        对记忆内容做脱水压缩。
        返回格式化的摘要字符串，可直接注入 Claude 上下文。
        使用 SQLite 缓存避免重复调用 API。
        """
        if not content or not content.strip():
            return "（空记忆 / empty memory）"

        # --- Content is short enough, no compression needed ---
        # --- 内容已经很短，不需要压缩 ---
        if count_tokens_approx(content) < 100:
            return self._format_output(content, metadata)

        # --- Check cache first ---
        # --- 先查缓存 ---
        cached = self._get_cached_summary(content)
        if cached:
            return self._format_output(cached, metadata)

        # --- API dehydration (no local fallback) ---
        # --- API 脱水（无本地降级）---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请配置 OMBRE_API_KEY")

        result = await self._api_dehydrate(content)
        # --- Cache the result ---
        self._set_cached_summary(content, result)
        return self._format_output(result, metadata)

    # ---------------------------------------------------------
    # Merge: blend new content into existing bucket
    # 合并：将新内容揉入已有桶，保持体积恒定
    # ---------------------------------------------------------
    async def merge(self, old_content: str, new_content: str) -> str:
        """
        Merge new content with old memory, preventing infinite bucket growth.
        将新内容与旧记忆合并，避免桶无限膨胀。
        """
        if not old_content and not new_content:
            return ""
        if not old_content:
            return new_content or ""
        if not new_content:
            return old_content

        # --- API merge (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_merge(old_content, new_content)
            if result:
                return result
            raise RuntimeError("API 合并返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 合并失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: dehydration
    # API 调用：脱水压缩
    # ---------------------------------------------------------
    async def _api_dehydrate(self, content: str) -> str:
        """
        Call LLM API for intelligent dehydration (via OpenAI-compatible client).
        调用 LLM API 执行智能脱水。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DEHYDRATE_PROMPT},
                {"role": "user", "content": content[:3000]},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    # ---------------------------------------------------------
    # API call: merge
    # API 调用：合并
    # ---------------------------------------------------------
    async def _api_merge(self, old_content: str, new_content: str) -> str:
        """
        Call LLM API for intelligent merge (via OpenAI-compatible client).
        调用 LLM API 执行智能合并。
        """
        user_msg = f"旧记忆：\n{old_content[:2000]}\n\n新内容：\n{new_content[:2000]}"
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MERGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""



    # ---------------------------------------------------------
    # Output formatting
    # 输出格式化
    # Wraps dehydrated result with bucket name, tags, emotion coords
    # 把脱水结果包装成带桶名、标签、情感坐标的可读文本
    # ---------------------------------------------------------
    def _format_output(self, content: str, metadata: dict = None) -> str:
        """
        Format dehydrated result into context-injectable text.
        将脱水结果格式化为可注入上下文的文本。
        """
        header = ""
        if metadata and isinstance(metadata, dict):
            name = metadata.get("name", "未命名")
            domains = ", ".join(metadata.get("domain", []))
            try:
                valence = float(metadata.get("valence", 0.5))
                arousal = float(metadata.get("arousal", 0.3))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3
            header = f"📌 Bucket: {name}"
            if domains:
                header += f" [Topic:{domains}]"
            header += f" [Emotion:V{valence:.1f}/A{arousal:.1f}]"
            # Show model's perspective if available (valence drift)
            model_v = metadata.get("model_valence")
            if model_v is not None:
                try:
                    header += f" [My perspective:V{float(model_v):.1f}]"
                except (ValueError, TypeError):
                    pass
            if metadata.get("digested"):
                header += " [digested]"
            header += "\n"
        
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
        return f"{header}{content}"

    # ---------------------------------------------------------
    # Auto-tagging: analyze content for domain + emotion + tags
    # 自动打标：分析内容，输出主题域 + 情感坐标 + 标签
    # Called by server.py when storing new memories
    # 存新记忆时由 server.py 调用
    # ---------------------------------------------------------
    async def analyze(self, content: str) -> dict:
        """
        Analyze content and return structured metadata.
        分析内容，返回结构化元数据。

        Returns: {"domain", "valence", "arousal", "tags", "suggested_name"}
        """
        if not content or not content.strip():
            return self._default_analysis()

        # --- API analyze (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_analyze(content)
            if result:
                return result
            raise RuntimeError("API 打标返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 打标失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: auto-tagging
    # API 调用：自动打标
    # ---------------------------------------------------------
    async def _api_analyze(self, content: str) -> dict:
        """
        Call LLM API for content analysis / tagging.
        调用 LLM API 执行内容分析打标。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYZE_PROMPT},
                {"role": "user", "content": content[:2000]},
            ],
            max_tokens=256,
            temperature=0.1,
        )
        if not response.choices:
            return self._default_analysis()
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return self._default_analysis()
        return self._parse_analysis(raw)

    # ---------------------------------------------------------
    # Parse API JSON response with safety checks
    # 解析 API 返回的 JSON，做安全校验
    # Ensure valence/arousal in 0~1, domain/tags valid
    # ---------------------------------------------------------
    def _parse_analysis(self, raw: str) -> dict:
        """
        Parse and validate API tagging result.
        解析并校验 API 返回的打标结果。
        """
        try:
            # Handle potential markdown code block wrapping
            # 处理可能的 markdown 代码块包裹
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"API tagging JSON parse failed / JSON 解析失败: {raw[:200]}")
            return self._default_analysis()

        if not isinstance(result, dict):
            return self._default_analysis()

        # --- Validate and clamp value ranges / 校验并钳制数值范围 ---
        try:
            valence = max(0.0, min(1.0, float(result.get("valence", 0.5))))
            arousal = max(0.0, min(1.0, float(result.get("arousal", 0.3))))
        except (ValueError, TypeError):
            valence, arousal = 0.5, 0.3

        return {
            "domain": result.get("domain", ["uncategorised"])[:3],
            "valence": valence,
            "arousal": arousal,
            "tags": result.get("tags", [])[:15],
            "suggested_name": str(result.get("suggested_name", ""))[:20],
        }

    # ---------------------------------------------------------
    # Default analysis result (empty content or total failure)
    # 默认分析结果（内容为空或完全失败时用）
    # ---------------------------------------------------------
    def _default_analysis(self) -> dict:
        """
        Return default neutral analysis result.
        返回默认的中性分析结果。
        """
        return {
            "domain": ["uncategorised"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "suggested_name": "",
        }

    # ---------------------------------------------------------
    # Diary digest: split daily notes into independent memory entries
    # 日记整理：把一大段日常拆分成多个独立记忆条目
    # For the "grow" tool — "dump a day's content and it gets organized"
    # 给 grow 工具用，"一天结束发一坨内容"靠这个
    # ---------------------------------------------------------
    async def digest(self, content: str) -> list[dict]:
        """
        Split a large chunk of daily content into independent memory entries.
        将一大段日常内容拆分成多个独立记忆条目。

        Returns: [{"name", "content", "domain", "valence", "arousal", "tags", "importance"}, ...]
        """
        if not content or not content.strip():
            return []

        # --- API digest (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_digest(content)
            if result:
                return result
            raise RuntimeError("API 日记整理返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 日记整理失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: diary digest
    # API 调用：日记整理
    # ---------------------------------------------------------
    async def _api_digest(self, content: str) -> list[dict]:
        """
        Call LLM API for diary organization.
        调用 LLM API 执行日记整理。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DIGEST_PROMPT},
                {"role": "user", "content": content[:5000]},
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        if not response.choices:
            return []
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return []
        return self._parse_digest(raw)

    # ---------------------------------------------------------
    # Parse diary digest result with safety checks
    # 解析日记整理结果，做安全校验
    # ---------------------------------------------------------
    def _parse_digest(self, raw: str) -> list[dict]:
        """
        Parse and validate API diary digest result.
        解析并校验 API 返回的日记整理结果。
        """
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            items = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"Diary digest JSON parse failed / JSON 解析失败: {raw[:200]}")
            return []

        if not isinstance(items, list):
            return []

        validated = []
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            try:
                importance = max(1, min(10, int(item.get("importance", 5))))
            except (ValueError, TypeError):
                importance = 5
            try:
                valence = max(0.0, min(1.0, float(item.get("valence", 0.5))))
                arousal = max(0.0, min(1.0, float(item.get("arousal", 0.3))))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3

            validated.append({
                "name": str(item.get("name", ""))[:20],
                "content": str(item.get("content", "")),
                "domain": item.get("domain", ["uncategorised"])[:3],
                "valence": valence,
                "arousal": arousal,
                "tags": item.get("tags", [])[:15],
                "importance": importance,
            })
        return validated
