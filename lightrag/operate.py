from __future__ import annotations
from functools import partial
from pathlib import Path

import asyncio
import os
import time
import json
import json_repair
from typing import TYPE_CHECKING, Any, AsyncIterator, overload, Literal
from collections import Counter, defaultdict

from lightrag.exceptions import (
    PipelineCancelledException,
    ChunkTokenLimitExceededError,
)
from lightrag.utils import (
    logger,
    compute_mdhash_id,
    Tokenizer,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    use_llm_func_with_cache,
    update_chunk_cache_list,
    remove_think_tags,
    pick_by_weighted_polling,
    pick_by_vector_similarity,
    process_chunks_unified,
    safe_vdb_operation_with_exception,
    create_prefixed_exception,
    convert_to_user_format,
    generate_reference_list_from_chunks,
    apply_source_ids_limit,
    merge_source_ids,
    make_relation_chunk_key,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    QueryResult,
    QueryContextResult,
)
from lightrag.extraction_parser import (
    process_extraction_result as _process_extraction_result,
    rebuild_from_extraction_result as _rebuild_from_extraction_result,
)
from lightrag.extraction_utils import should_retry_without_response_format
from lightrag.legal_document import (
    best_legal_status_metadata,
    build_structured_legal_continue_prompt,
    build_structured_legal_extraction_prompts,
    extract_legal_entities as run_legal_entity_extraction,
    legal_chunking_html,
    legal_chunking_markdown,
    legal_status_sort_priority,
)
from lightrag.prompt import PROMPTS
from lightrag.prompt_manager import PromptManager
from lightrag.language_detector import SupportedLanguage
from lightrag.constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_MAX_FILE_PATHS,
)
from lightrag.kg.shared_storage import get_storage_keyed_lock
from dotenv import load_dotenv

if TYPE_CHECKING:
    from lightrag.citation import CitationTracker


def _skipped_entity_chunk_failure(
    failed_chunks: int,
    total_chunks: int,
    max_failed_chunk_ratio: float,
    *,
    fail_on_all_chunks_failed: bool,
    fail_on_too_many_skipped_chunks: bool,
    first_errors: list[str] | None = None,
) -> tuple[bool, int, str]:
    if failed_chunks <= 0 or total_chunks <= 0:
        return False, 0, ""

    max_allowed_failed_chunks = max(1, int(total_chunks * max_failed_chunk_ratio))
    first_errors_text = ""
    if first_errors:
        first_errors_text = ". First errors: " + " | ".join(first_errors[:5])
    if fail_on_all_chunks_failed and failed_chunks >= total_chunks:
        return (
            True,
            max_allowed_failed_chunks,
            f"Entity extraction failed all chunks: {failed_chunks}/{total_chunks}"
            f"{first_errors_text}",
        )
    if fail_on_too_many_skipped_chunks and failed_chunks > max_allowed_failed_chunks:
        return (
            True,
            max_allowed_failed_chunks,
            f"Entity extraction skipped too many chunks: {failed_chunks}/{total_chunks}"
            f"{first_errors_text}",
        )
    return False, max_allowed_failed_chunks, ""


# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


def _detect_query_language(query: str) -> str:
    """Best-effort language detection for query response locking."""
    lower = query.lower()
    # Vietnamese diacritics and common Vietnamese-only letters.
    if any(
        ch in lower
        for ch in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ"
    ):
        return "Vietnamese"
    # Fallback default for Latin-script queries.
    return "English"


def _build_language_lock_instruction(query: str) -> str:
    """Build strict language lock instruction from current user query."""
    lang = _detect_query_language(query)
    return (
        "\n\nLanguage lock (highest priority): "
        f"Respond entirely in {lang}, matching the current user query language. "
        "Do not switch to another language because of source documents or prior turns."
    )


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    **kwargs,
) -> list[dict[str, Any]]:
    """Split content into chunks based on token size with optional character-based splitting.

    This function provides flexible text chunking with two modes:
    1. Token-based chunking: Splits content by token count with overlap
    2. Character-based chunking: Splits by a delimiter, then applies token limits

    Args:
        tokenizer: Tokenizer instance for encoding/decoding text
        content: Text content to be chunked
        split_by_character: Optional delimiter to split content before token chunking.
            If None, uses pure token-based chunking.
        split_by_character_only: If True, enforces strict character-based splitting
            without further token-based subdivision. Raises error if any chunk exceeds
            token limit. If False, oversized chunks are further split by tokens.
        chunk_overlap_token_size: Number of overlapping tokens between consecutive chunks.
            Default is 100 tokens.
        chunk_token_size: Maximum token size for each chunk. Default is 1200 tokens.

    Returns:
        List of chunk dictionaries, each containing:
            - tokens (int): Number of tokens in the chunk
            - content (str): The chunk text content (stripped)
            - chunk_order_index (int): Sequential index of the chunk

    Raises:
        ChunkTokenLimitExceededError: When split_by_character_only=True and a
            character-split chunk exceeds chunk_token_size.

    Examples:
        >>> tokenizer = TiktokenTokenizer()
        >>> content = "Long text content..."
        >>> chunks = chunking_by_token_size(tokenizer, content, chunk_token_size=500)
        >>> print(chunks[0])
        {'tokens': 500, 'content': '...', 'chunk_order_index': 0}
    """
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=chunk[:120],
                    )
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    for start in range(
                        0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + chunk_token_size]
                        )
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append(
                {
                    "tokens": min(chunk_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


def general_semantic_chunking_markdown(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    **kwargs,
) -> list[dict[str, Any]]:
    """Semantic chunking for general Markdown documents with breadcrumb inheritance.

    This function splits general documents by markdown headers (H1-H6) while maintaining
    a breadcrumb stack that tracks the header hierarchy. Each chunk includes a breadcrumb
    prefix showing its position in the document structure.

    Args:
        tokenizer: Tokenizer for counting tokens
        content: Markdown content of the document
        split_by_character: Not used (for signature compatibility)
        split_by_character_only: Not used (for signature compatibility)
        chunk_overlap_token_size: Token overlap between chunks (default: 100)
        chunk_token_size: Maximum tokens per chunk (default: 1200)

    Returns:
        List of chunk dictionaries with keys:
        - tokens: Number of tokens in the chunk
        - content: Chunk content with breadcrumb prefix
        - chunk_order_index: Sequential index of the chunk
        - context: Dict with 'breadcrumb' key containing list of header texts
    """
    import re

    # 1. Protect HTML tables by replacing them with placeholders
    table_pattern = re.compile(r"<table.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
    tables = {}

    def replace_table(match):
        placeholder = f"__TABLE_PLACEHOLDER_{len(tables)}__"
        tables[placeholder] = match.group(0)
        return placeholder

    content_with_placeholders = table_pattern.sub(replace_table, content)

    # 2. Split by double newlines to get blocks
    raw_blocks = content_with_placeholders.split("\n\n")

    blocks = []
    for raw_block in raw_blocks:
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        # Restore tables
        for placeholder, table_html in tables.items():
            if placeholder in raw_block:
                raw_block = raw_block.replace(placeholder, table_html)
        blocks.append(raw_block)

    results = []
    current_chunk_blocks = []
    current_chunk_tokens = 0
    chunk_index = 0

    # Breadcrumb tracking: stack of (level, text) tuples
    breadcrumb = []  # [(level, text), ...]

    def save_chunk(blocks_to_save):
        nonlocal chunk_index
        if not blocks_to_save:
            return

        # Build breadcrumb prefix
        breadcrumb_prefix = ""
        if breadcrumb:
            breadcrumb_path = " > ".join([h[1] for h in breadcrumb])
            breadcrumb_prefix = f"[{breadcrumb_path}]\n\n"

        chunk_content = breadcrumb_prefix + "\n\n".join(blocks_to_save)
        results.append(
            {
                "tokens": len(tokenizer.encode(chunk_content)),
                "content": chunk_content.strip(),
                "chunk_order_index": chunk_index,
                "context": {
                    "breadcrumb": [h[1] for h in breadcrumb],
                },
            }
        )
        chunk_index += 1

    for block in blocks:
        # Check if block is a header
        header_match = re.match(r"^(#{1,6})\s+(.*)", block)
        if header_match:
            level = len(header_match.group(1))
            text = header_match.group(2).strip()

            # Update breadcrumb stack
            # Remove headers at same or deeper level
            while breadcrumb and breadcrumb[-1][0] >= level:
                breadcrumb.pop()
            # Add current header
            breadcrumb.append((level, text))

        block_tokens = len(tokenizer.encode(block))

        # If adding this block exceeds limit AND we already have blocks
        if (
            current_chunk_tokens + block_tokens > chunk_token_size
            and current_chunk_blocks
        ):
            save_chunk(current_chunk_blocks)
            current_chunk_blocks = []
            current_chunk_tokens = 0

        # If the block itself is larger than the limit
        if block_tokens > chunk_token_size:
            # If it's a table, keep it intact
            if "<table" in block.lower():
                if current_chunk_blocks:
                    save_chunk(current_chunk_blocks)
                    current_chunk_blocks = []
                    current_chunk_tokens = 0
                save_chunk([block])
                continue
            else:
                # Fallback to token-based splitting
                if current_chunk_blocks:
                    save_chunk(current_chunk_blocks)
                    current_chunk_blocks = []
                    current_chunk_tokens = 0

                block_encoded = tokenizer.encode(block)
                for start in range(
                    0, len(block_encoded), chunk_token_size - chunk_overlap_token_size
                ):
                    chunk_content = tokenizer.decode(
                        block_encoded[start : start + chunk_token_size]
                    )
                    save_chunk([chunk_content])
                continue

        current_chunk_blocks.append(block)
        current_chunk_tokens += block_tokens

    if current_chunk_blocks:
        save_chunk(current_chunk_blocks)

    return results


def select_chunking_function(doc_type: str, doc_extension: str = "") -> callable:
    """Select appropriate chunking function based on document type.

    This function returns the appropriate chunking strategy based on the document type:
    - "legal": Returns legal_chunking_markdown for Vietnamese legal documents
    - "general": Returns general_semantic_chunking_markdown for general documents

    Args:
        doc_type: Document type classification ("legal" or "general")

    Returns:
        Callable chunking function appropriate for the document type

    Examples:
        >>> chunking_func = select_chunking_function("legal")
        >>> chunks = chunking_func(tokenizer, content)

        >>> chunking_func = select_chunking_function("general")
        >>> chunks = chunking_func(tokenizer, content)
    """
    normalized_extension = (doc_extension or "").lower().lstrip(".")

    if doc_type == "legal" and normalized_extension in {"html", "htm"}:
        return legal_chunking_html
    elif doc_type == "legal":
        return legal_chunking_markdown
    else:  # doc_type == "general" or any other value defaults to general
        return general_semantic_chunking_markdown


def semantic_chunking_markdown(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    **kwargs,
) -> list[dict[str, Any]]:
    """Semantic chunking for Markdown content, prioritizing headers and tables.

    This splits Markdown text logically by paragraphs and headers. It strictly avoids
    splitting HTML or Markdown tables across chunks. If a section is split, its parent
    header is prepended to subsequent chunks to preserve context.
    """
    import re

    # 1. Protect HTML tables by replacing them with placeholders
    table_pattern = re.compile(r"<table.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
    tables = {}

    def replace_table(match):
        placeholder = f"__TABLE_PLACEHOLDER_{len(tables)}__"
        tables[placeholder] = match.group(0)
        return placeholder

    content_with_placeholders = table_pattern.sub(replace_table, content)

    # 2. Split into blocks
    split_char = "\n\n" if not split_by_character else split_by_character
    raw_blocks = content_with_placeholders.split(split_char)

    blocks = []
    for raw_block in raw_blocks:
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        # Restore tables
        for placeholder, table_html in tables.items():
            if placeholder in raw_block:
                raw_block = raw_block.replace(placeholder, table_html)
        blocks.append(raw_block)

    results = []
    current_chunk_blocks = []
    current_chunk_tokens = 0
    chunk_index = 0
    current_context = ""

    def save_chunk(blocks_to_save):
        nonlocal chunk_index
        if not blocks_to_save:
            return
        chunk_content = "\n\n".join(blocks_to_save)
        results.append(
            {
                "tokens": len(tokenizer.encode(chunk_content)),
                "content": chunk_content.strip(),
                "chunk_order_index": chunk_index,
            }
        )
        chunk_index += 1

    for block in blocks:
        # Check if block is a header
        header_match = re.match(r"^(#{1,6})\s+(.*)", block)
        if header_match:
            current_context = block

        block_tokens = len(tokenizer.encode(block))

        # If adding this block exceeds limit AND we already have blocks
        if (
            current_chunk_tokens + block_tokens > chunk_token_size
            and current_chunk_blocks
        ):
            save_chunk(current_chunk_blocks)

            # Start new chunk, carrying over the current header context
            current_chunk_blocks = []
            current_chunk_tokens = 0
            if current_context and current_context != block:
                current_chunk_blocks.append(current_context)
                current_chunk_tokens += len(tokenizer.encode(current_context))

        # If the block itself is larger than the limit
        if block_tokens > chunk_token_size:
            # If it's a table, we keep it intact despite the size
            if "<table" in block.lower():
                if current_chunk_blocks:
                    save_chunk(current_chunk_blocks)
                    current_chunk_blocks = []
                    current_chunk_tokens = 0

                # Add context if any
                table_block = (
                    f"{current_context}\n\n{block}"
                    if current_context and not block.startswith(current_context)
                    else block
                )
                save_chunk([table_block])
                continue
            else:
                # Fallback to token-based splitting for long paragraphs
                if current_chunk_blocks:
                    save_chunk(current_chunk_blocks)
                    current_chunk_blocks = []
                    current_chunk_tokens = 0

                # Split block by tokens with overlap
                block_encoded = tokenizer.encode(block)
                for start in range(
                    0, len(block_encoded), chunk_token_size - chunk_overlap_token_size
                ):
                    chunk_content = tokenizer.decode(
                        block_encoded[start : start + chunk_token_size]
                    )
                    chunk_content = (
                        f"{current_context}\n\n{chunk_content}"
                        if current_context
                        else chunk_content
                    )
                    save_chunk([chunk_content])
                continue

        current_chunk_blocks.append(block)
        current_chunk_tokens += block_tokens

    if current_chunk_blocks:
        save_chunk(current_chunk_blocks)

    return results


async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    separator: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    if len(description_list) == 1:
        return description_list[0], False

    # Get configuration
    tokenizer: Tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list
        total_tokens = sum(len(tokenizer.encode(desc)) for desc in current_list)

        # If total length is within limits, perform final summarization
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                final_description = separator.join(current_list)
                return final_description if final_description else "", llm_was_used
            else:
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize description found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list
        for i, desc in enumerate(current_list):
            desc_tokens = len(tokenizer.encode(desc))

            # If adding current description would exceed limit, finalize current chunk
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize description found"
                    )
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # curren_chunk is ready for summary in reduce phase
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # Reduce phase: summarize each group from chunks
        new_summaries = []
        for chunk in chunks:
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    global_config,
                    llm_response_cache,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        current_list = new_summaries


async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        description_type: Type of description being summarized (e.g., "entity", "relation")
        description_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing LLM function, tokenizer, and settings
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Summarized description string
    """
    use_llm_func: callable = global_config["llm_model_func"]
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=8)

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    summary_length_recommended = global_config["summary_length_recommended"]

    # Initialize PromptManager for language-aware prompt retrieval
    prompt_manager = PromptManager(PROMPTS)

    # Determine language for prompts based on language setting
    prompt_language = (
        SupportedLanguage.VIETNAMESE
        if language == "Vietnamese"
        else SupportedLanguage.ENGLISH
    )

    # Get prompt template using PromptManager
    prompt_template = prompt_manager.get_prompt(
        "summarize_entity_descriptions", language=prompt_language
    )

    # Convert descriptions to JSONL format and apply token-based truncation
    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]

    # Create list of JSON objects with "Description" field
    json_descriptions = [{"Description": desc} for desc in description_list]

    # Use truncate_list_by_token_size for length truncation
    truncated_json_descriptions = truncate_list_by_token_size(
        json_descriptions,
        key=lambda x: json.dumps(x, ensure_ascii=False),
        max_token_size=summary_context_size,
        tokenizer=tokenizer,
    )

    # Convert to JSONL format (one JSON object per line)
    joined_descriptions = "\n".join(
        json.dumps(desc, ensure_ascii=False) for desc in truncated_json_descriptions
    )

    # Prepare context for the prompt
    context_base = dict(
        description_type=description_type,
        description_name=description_name,
        description_list=joined_descriptions,
        summary_length=summary_length_recommended,
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)

    # Use LLM function with cache (higher priority for summary generation)
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
    )

    # Check summary token length against embedding limit
    embedding_token_limit = global_config.get("embedding_token_limit")
    if embedding_token_limit is not None and summary:
        tokenizer = global_config["tokenizer"]
        summary_token_count = len(tokenizer.encode(summary))
        threshold = int(embedding_token_limit)

        if summary_token_count > threshold:
            logger.warning(
                f"Summary tokens({summary_token_count}) exceeds embedding_token_limit({embedding_token_limit}) "
                f" for {description_type}: {description_name}"
            )

    return summary


async def rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, list[str]],
    relationships_to_rebuild: dict[tuple[str, str], list[str]],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
) -> None:
    """Rebuild entity and relationship descriptions from cached extraction results with parallel processing

    This method uses cached LLM extraction results instead of calling LLM again,
    following the same approach as the insert process. Now with parallel processing
    controlled by llm_model_max_async and using get_storage_keyed_lock for data consistency.

    Args:
        entities_to_rebuild: Dict mapping entity_name -> list of remaining chunk_ids
        relationships_to_rebuild: Dict mapping (src, tgt) -> list of remaining chunk_ids
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_storage: Text chunks storage
        llm_response_cache: LLM response cache
        global_config: Global configuration containing llm_model_max_async
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        entity_chunks_storage: KV storage maintaining full chunk IDs per entity
        relation_chunks_storage: KV storage maintaining full chunk IDs per relation
    """
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Get all referenced chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)

    status_message = f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk extractions (parallel processing)"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Get cached extraction results for these chunks using storage
    # cached_results： chunk_id -> [list of (extraction_result, create_time) from LLM cache sorted by create_time of the first extraction_result]
    cached_results = await _get_cached_extraction_results(
        llm_response_cache,
        all_referenced_chunk_ids,
        text_chunks_storage=text_chunks_storage,
    )

    if not cached_results:
        status_message = "No cached extraction results found, cannot rebuild"
        logger.warning(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
        return

    # Process cached results to get entities and relationships for each chunk
    chunk_entities = {}  # chunk_id -> {entity_name: [entity_data]}
    chunk_relationships = {}  # chunk_id -> {(src, tgt): [relationship_data]}

    for chunk_id, results in cached_results.items():
        try:
            # Handle multiple extraction results per chunk
            chunk_entities[chunk_id] = defaultdict(list)
            chunk_relationships[chunk_id] = defaultdict(list)

            # process multiple LLM extraction results for a single chunk_id
            for result in results:
                entities, relationships = await _rebuild_from_extraction_result(
                    text_chunks_storage=text_chunks_storage,
                    chunk_id=chunk_id,
                    extraction_result=result[0],
                    timestamp=result[1],
                )

                # Merge entities and relationships from this extraction result
                # Compare description lengths and keep the better version for the same chunk_id
                for entity_name, entity_list in entities.items():
                    if entity_name not in chunk_entities[chunk_id]:
                        # New entity for this chunk_id
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    elif len(chunk_entities[chunk_id][entity_name]) == 0:
                        # Empty list, add the new entities
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_entities[chunk_id][entity_name][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(entity_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new entity that has longer description
                            chunk_entities[chunk_id][entity_name] = list(entity_list)
                        # Otherwise keep existing version

                # Compare description lengths and keep the better version for the same chunk_id
                for rel_key, rel_list in relationships.items():
                    if rel_key not in chunk_relationships[chunk_id]:
                        # New relationship for this chunk_id
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    elif len(chunk_relationships[chunk_id][rel_key]) == 0:
                        # Empty list, add the new relationships
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_relationships[chunk_id][rel_key][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(rel_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new relationship that has longer description
                            chunk_relationships[chunk_id][rel_key] = list(rel_list)
                        # Otherwise keep existing version

        except Exception as e:
            status_message = (
                f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
            )
            logger.info(status_message)  # Per requirement, change to info
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            continue

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0

    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        entity_chunks_storage=entity_chunks_storage,
                    )
                    rebuilt_entities_count += 1
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild `{entity_name}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # Sort src and tgt to ensure order-independent lock key generation
            sorted_key_parts = sorted([src, tgt])
            async with get_storage_keyed_lock(
                sorted_key_parts,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        entities_vdb=entities_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        relation_chunks_storage=relation_chunks_storage,
                        entity_chunks_storage=entity_chunks_storage,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                    )
                    rebuilt_relationships_count += 1
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild `{src}`~`{tgt}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    # Create tasks for parallel processing
    tasks = []

    # Add entity rebuilding tasks
    for entity_name, chunk_ids in entities_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
        tasks.append(task)

    # Add relationship rebuilding tasks
    for (src, tgt), chunk_ids in relationships_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
        tasks.append(task)

    # Log parallel processing start
    status_message = f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Execute all tasks in parallel with semaphore control and early failure detection
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                # Task completed successfully, retrieve result to mark as processed
                task.result()
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Re-raise the first exception to notify the caller
        raise first_exception

    # Final status report
    status_message = f"KG rebuild completed: {rebuilt_entities_count} entities and {rebuilt_relationships_count} relationships rebuilt successfully."
    if failed_entities_count > 0 or failed_relationships_count > 0:
        status_message += f" Failed: {failed_entities_count} entities, {failed_relationships_count} relationships."

    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _get_cached_extraction_results(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
) -> dict[str, list[str]]:
    """Get cached extraction results for specific chunk IDs

    This function retrieves cached LLM extraction results for the given chunk IDs and returns
    them sorted by creation time. The results are sorted at two levels:
    1. Individual extraction results within each chunk are sorted by create_time (earliest first)
    2. Chunks themselves are sorted by the create_time of their earliest extraction result

    Args:
        llm_response_cache: LLM response cache storage
        chunk_ids: Set of chunk IDs to get cached results for
        text_chunks_storage: Text chunks storage for retrieving chunk data and LLM cache references

    Returns:
        Dict mapping chunk_id -> list of extraction_result_text, where:
        - Keys (chunk_ids) are ordered by the create_time of their first extraction result
        - Values (extraction results) are ordered by create_time within each chunk
    """
    cached_results = {}

    # Collect all LLM cache IDs from chunks
    all_cache_ids = set()

    # Read from storage
    chunk_data_list = await text_chunks_storage.get_by_ids(list(chunk_ids))
    for chunk_data in chunk_data_list:
        if chunk_data and isinstance(chunk_data, dict):
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if llm_cache_list:
                all_cache_ids.update(llm_cache_list)
        else:
            logger.warning(f"Chunk data is invalid or None: {chunk_data}")

    if not all_cache_ids:
        logger.warning(f"No LLM cache IDs found for {len(chunk_ids)} chunk IDs")
        return cached_results

    # Batch get LLM cache entries
    cache_data_list = await llm_response_cache.get_by_ids(list(all_cache_ids))

    # Process cache entries and group by chunk_id
    valid_entries = 0
    for cache_entry in cache_data_list:
        if (
            cache_entry is not None
            and isinstance(cache_entry, dict)
            and cache_entry.get("cache_type") == "extract"
            and cache_entry.get("chunk_id") in chunk_ids
        ):
            chunk_id = cache_entry["chunk_id"]
            extraction_result = cache_entry["return"]
            create_time = cache_entry.get(
                "create_time", 0
            )  # Get creation time, default to 0
            valid_entries += 1

            # Support multiple LLM caches per chunk
            if chunk_id not in cached_results:
                cached_results[chunk_id] = []
            # Store tuple with extraction result and creation time for sorting
            cached_results[chunk_id].append((extraction_result, create_time))

    # Sort extraction results by create_time for each chunk and collect earliest times
    chunk_earliest_times = {}
    for chunk_id in cached_results:
        # Sort by create_time (x[1]), then extract only extraction_result (x[0])
        cached_results[chunk_id].sort(key=lambda x: x[1])
        # Store the earliest create_time for this chunk (first item after sorting)
        chunk_earliest_times[chunk_id] = cached_results[chunk_id][0][1]

    # Sort cached_results by the earliest create_time of each chunk
    sorted_chunk_ids = sorted(
        chunk_earliest_times.keys(), key=lambda chunk_id: chunk_earliest_times[chunk_id]
    )

    # Rebuild cached_results in sorted order
    sorted_cached_results = {}
    for chunk_id in sorted_chunk_ids:
        sorted_cached_results[chunk_id] = cached_results[chunk_id]

    logger.info(
        f"Found {valid_entries} valid cache entries, {len(sorted_cached_results)} chunks with results"
    )
    return sorted_cached_results  # each item: list(extraction_result, create_time)


async def _rebuild_single_entity(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name: str,
    chunk_ids: list[str],
    chunk_entities: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    entity_chunks_storage: BaseKVStorage | None = None,
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild a single entity from cached extraction results"""

    # Get current entity data
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    if not current_entity:
        return

    # Helper function to update entity in both graph and vector storage
    async def _update_entity_storage(
        final_description: str,
        entity_type: str,
        file_paths: list[str],
        source_chunk_ids: list[str],
        truncation_info: str = "",
    ):
        try:
            # Update entity in graph storage (critical path)
            updated_entity_data = {
                **current_entity,
                "description": final_description,
                "entity_type": entity_type,
                "source_id": GRAPH_FIELD_SEP.join(source_chunk_ids),
                "file_path": GRAPH_FIELD_SEP.join(file_paths)
                if file_paths
                else current_entity.get("file_path", "unknown_source"),
                "created_at": int(time.time()),
                "truncate": truncation_info,
            }
            await knowledge_graph_inst.upsert_node(entity_name, updated_entity_data)

            # Update entity in vector database (equally critical)
            entity_vdb_id = compute_mdhash_id(
                entity_name,
                prefix="ent-",
                workspace=global_config.get("workspace", "default"),
            )
            entity_content = f"{entity_name}\n{final_description}"

            vdb_data = {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }

            # Use safe operation wrapper - VDB failure must throw exception
            await safe_vdb_operation_with_exception(
                operation=lambda: entities_vdb.upsert(vdb_data),
                operation_name="rebuild_entity_upsert",
                entity_name=entity_name,
                max_retries=3,
                retry_delay=0.1,
            )

        except Exception as e:
            error_msg = f"Failed to update entity storage for `{entity_name}`: {e}"
            logger.error(error_msg)
            raise  # Re-raise exception

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if entity_chunks_storage is not None and normalized_chunk_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_entity"],
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # Collect all entity data from relevant (limited) chunks
    all_entity_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_entities and entity_name in chunk_entities[chunk_id]:
            all_entity_data.extend(chunk_entities[chunk_id][entity_name])

    if not all_entity_data:
        logger.warning(
            f"No entity data found for `{entity_name}`, trying to rebuild from relationships"
        )

        # Get all edges connected to this entity
        edges = await knowledge_graph_inst.get_node_edges(entity_name)
        if not edges:
            logger.warning(f"No relations attached to entity `{entity_name}`")
            return

        # Collect relationship data to extract entity information
        relationship_descriptions = []
        file_paths = set()

        # Get edge data for all connected relationships
        for src_id, tgt_id in edges:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data:
                if edge_data.get("description"):
                    relationship_descriptions.append(edge_data["description"])

                if edge_data.get("file_path"):
                    edge_file_paths = edge_data["file_path"].split(GRAPH_FIELD_SEP)
                    file_paths.update(edge_file_paths)

        # deduplicate descriptions
        description_list = list(dict.fromkeys(relationship_descriptions))

        # Generate final description from relationships or fallback to current
        if description_list:
            final_description, _ = await _handle_entity_relation_summary(
                "Entity",
                entity_name,
                description_list,
                GRAPH_FIELD_SEP,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            final_description = current_entity.get("description", "")

        entity_type = current_entity.get("entity_type", "UNKNOWN")
        await _update_entity_storage(
            final_description,
            entity_type,
            file_paths,
            limited_chunk_ids,
        )
        return

    # Process cached entity data
    descriptions = []
    entity_types = []
    file_paths_list = []
    seen_paths = set()

    for entity_data in all_entity_data:
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        if entity_data.get("file_path"):
            file_path = entity_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply MAX_FILE_PATHS limit
    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{entity_name}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    entity_types = list(dict.fromkeys(entity_types))

    # Get most common entity type
    entity_type = (
        max(set(entity_types), key=entity_types.count)
        if entity_types
        else current_entity.get("entity_type", "UNKNOWN")
    )

    # Generate final description from entities or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = current_entity.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    await _update_entity_storage(
        final_description,
        entity_type,
        file_paths_list,
        limited_chunk_ids,
        truncation_info,
    )

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{entity_name}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    logger.info(status_message)
    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _rebuild_single_relationship(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    src: str,
    tgt: str,
    chunk_ids: list[str],
    chunk_relationships: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild a single relationship from cached extraction results

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    if not current_relationship:
        return

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if relation_chunks_storage is not None and normalized_chunk_ids:
        storage_key = make_relation_chunk_key(src, tgt)
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_relation"],
        limit_method,
        identifier=f"`{src}`~`{tgt}`",
    )

    # Collect all relationship data from relevant chunks
    all_relationship_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_relationships:
            # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
            for edge_key in [(src, tgt), (tgt, src)]:
                if edge_key in chunk_relationships[chunk_id]:
                    all_relationship_data.extend(
                        chunk_relationships[chunk_id][edge_key]
                    )

    if not all_relationship_data:
        logger.warning(f"No relation data found for `{src}-{tgt}`")
        return

    # Merge descriptions and keywords
    descriptions = []
    keywords = []
    weights = []
    file_paths_list = []
    seen_paths = set()

    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            keywords.append(rel_data["keywords"])
        if rel_data.get("weight"):
            weights.append(rel_data["weight"])
        if rel_data.get("file_path"):
            file_path = rel_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{src}`~`{tgt}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    keywords = list(dict.fromkeys(keywords))

    combined_keywords = (
        ", ".join(set(keywords))
        if keywords
        else current_relationship.get("keywords", "")
    )

    weight = sum(weights) if weights else current_relationship.get("weight", 1.0)

    # Generate final description from relations or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Relation",
            f"{src}-{tgt}",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        # fallback to keep current(unchanged)
        final_description = current_relationship.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    # Update relationship in graph storage
    updated_relationship_data = {
        **current_relationship,
        "description": final_description
        if final_description
        else current_relationship.get("description", ""),
        "keywords": combined_keywords,
        "weight": weight,
        "source_id": GRAPH_FIELD_SEP.join(limited_chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join([fp for fp in file_paths_list if fp])
        if file_paths_list
        else current_relationship.get("file_path", "unknown_source"),
        "truncate": truncation_info,
    }

    # Ensure both endpoint nodes exist before writing the edge back
    # (certain storage backends require pre-existing nodes).
    node_description = (
        updated_relationship_data["description"]
        if updated_relationship_data.get("description")
        else current_relationship.get("description", "")
    )
    node_source_id = updated_relationship_data.get("source_id", "")
    node_file_path = updated_relationship_data.get("file_path", "unknown_source")

    for node_id in {src, tgt}:
        if not (await knowledge_graph_inst.has_node(node_id)):
            node_created_at = int(time.time())
            node_data = {
                "entity_id": node_id,
                "source_id": node_source_id,
                "description": node_description,
                "entity_type": "UNKNOWN",
                "file_path": node_file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(node_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None and limited_chunk_ids:
                await entity_chunks_storage.upsert(
                    {
                        node_id: {
                            "chunk_ids": limited_chunk_ids,
                            "count": len(limited_chunk_ids),
                        }
                    }
                )

            # Update entity_vdb for the newly created entity
            if entities_vdb is not None:
                entity_vdb_id = compute_mdhash_id(
                    node_id,
                    prefix="ent-",
                    workspace=global_config.get("workspace", "default"),
                )
                entity_content = f"{node_id}\n{node_description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": node_id,
                        "source_id": node_source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": node_file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entities_vdb.upsert(payload),
                    operation_name="rebuild_added_entity_upsert",
                    entity_name=node_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    # Sort src and tgt to ensure consistent ordering (smaller string first)
    if src > tgt:
        src, tgt = tgt, src
    try:
        rel_vdb_id = compute_mdhash_id(
            src + tgt,
            prefix="rel-",
            workspace=global_config.get("workspace", "default"),
        )
        rel_vdb_id_reverse = compute_mdhash_id(
            tgt + src,
            prefix="rel-",
            workspace=global_config.get("workspace", "default"),
        )

        # Delete old vector records first (both directions to be safe)
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )

        # Insert new vector record
        rel_content = f"{combined_keywords}\t{src}\n{tgt}\n{final_description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": updated_relationship_data["source_id"],
                "content": rel_content,
                "keywords": combined_keywords,
                "description": final_description,
                "weight": weight,
                "file_path": updated_relationship_data["file_path"],
            }
        }

        # Use safe operation wrapper - VDB failure must throw exception
        await safe_vdb_operation_with_exception(
            operation=lambda: relationships_vdb.upsert(vdb_data),
            operation_name="rebuild_relationship_upsert",
            entity_name=f"{src}-{tgt}",
            max_retries=3,
            retry_delay=0.2,
        )

    except Exception as e:
        error_msg = f"Failed to rebuild relationship storage for `{src}-{tgt}`: {e}"
        logger.error(error_msg)
        raise  # Re-raise exception

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{src}`~`{tgt}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    # Add truncation info from apply_source_ids_limit if truncation occurred
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f" ({limit_method}:{len(limited_chunk_ids)}/{len(normalized_chunk_ids)})"
        )
        status_message += truncation_info

    logger.info(status_message)

    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    already_node: dict | None = None,
    existing_full_source_ids: list[str] | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    # 1. Get existing node data from knowledge graph
    if already_node:
        existing_entity_type = already_node.get("entity_type")
        # Coerce to str before any string operations: non-string values from
        # API/custom graph paths would otherwise raise TypeError on the comma check.
        if (
            not isinstance(existing_entity_type, str)
            or not existing_entity_type.strip()
        ):
            existing_entity_type = "UNKNOWN"
        # Sanitize entity_type read back from DB to prevent dirty data from propagating
        if "," in existing_entity_type:
            original = existing_entity_type
            tokens = [t.strip() for t in existing_entity_type.split(",")]
            non_empty = [t for t in tokens if t]
            existing_entity_type = non_empty[0] if non_empty else "UNKNOWN"
            logger.warning(
                f"Entity type read from DB contains comma, taking first non-empty token: '{original}' -> '{existing_entity_type}'"
            )
        already_entity_types.append(existing_entity_type)

        existing_source_id = already_node.get("source_id") or ""
        already_source_ids.extend(existing_source_id.split(GRAPH_FIELD_SEP))

        existing_file_path = already_node.get("file_path") or "unknown_source"
        already_file_paths.extend(existing_file_path.split(GRAPH_FIELD_SEP))

        existing_desc = (already_node.get("description") or "").strip()
        if existing_desc:
            already_description.extend(existing_desc.split(GRAPH_FIELD_SEP))

    new_source_ids = [dp["source_id"] for dp in nodes_data if dp.get("source_id")]

    if existing_full_source_ids is None:
        existing_full_source_ids = []
        if entity_chunks_storage is not None:
            stored_chunks = await entity_chunks_storage.get_by_id(entity_name)
            if stored_chunks and isinstance(stored_chunks, dict):
                existing_full_source_ids = [
                    chunk_id
                    for chunk_id in stored_chunks.get("chunk_ids", [])
                    if chunk_id
                ]

        if not existing_full_source_ids:
            existing_full_source_ids = [
                chunk_id for chunk_id in already_source_ids if chunk_id
            ]

    # 2. Merging new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    # Removed individual entity_chunks_storage.upsert to batch it outside

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_entity")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # 4. Only keep nodes not filter by apply_source_ids_limit if limit_method is KEEP
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        filtered_nodes = []
        for dp in nodes_data:
            source_id = dp.get("source_id")
            # Skip descriptions sourced from chunks dropped by the limitation cap
            if (
                source_id
                and source_id not in allowed_source_ids
                and source_id not in existing_full_source_ids
            ):
                continue
            filtered_nodes.append(dp)
        nodes_data = filtered_nodes
    else:  # In FIFO mode, keep all nodes - truncation happens at source_ids level only
        nodes_data = list(nodes_data)

    # 5. Check if we need to skip summary due to source_ids limit
    if (
        limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
        and len(existing_full_source_ids) >= max_source_limit
        and not nodes_data
    ):
        if already_node:
            logger.info(
                f"Skipped `{entity_name}`: KEEP old chunks {already_source_ids}/{len(full_source_ids)}"
            )
            existing_node_data = dict(already_node)
            return existing_node_data
        else:
            logger.error(f"Internal Error: already_node missing for `{entity_name}`")
            raise ValueError(
                f"Internal Error: already_node missing for `{entity_name}`"
            )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize entity type by highest count
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 7. Deduplicate nodes by description, keeping first occurrence in the same document
    unique_nodes = {}
    for dp in nodes_data:
        desc = dp.get("description")
        if not desc:
            continue
        if desc not in unique_nodes:
            unique_nodes[desc] = dp

    # Sort description by timestamp, then by description length when timestamps are the same
    sorted_nodes = sorted(
        unique_nodes.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    sorted_descriptions = [dp["description"] for dp in sorted_nodes]

    # Combine already_description with sorted new sorted descriptions
    description_list = already_description + sorted_descriptions
    if not description_list:
        fallback_description = f"Entity {entity_name}"
        logger.warning(
            f"Entity `{entity_name}` has no description; fallback to `{fallback_description}`"
        )
        description_list = [fallback_description]

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during entity summary")

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        entity_name,
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Indicating file_path has been truncated before

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in nodes_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )
        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{entity_name}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)
    legal_status_metadata = best_legal_status_metadata(
        ([already_node] if already_node else []) + list(nodes_data)
    )

    # 10.Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(nodes_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # Truncate description to prevent Milvus Metadata limits (64KB)
    if len(description) > 30000:
        description = description[:30000] + "...(truncated)"

    # 11. Update both graph and vector db
    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
        truncate=truncation_info,
    )
    node_data.update(legal_status_metadata)

    node_data["entity_name"] = entity_name
    data_for_vdb = None
    if entity_vdb is not None:
        entity_vdb_id = compute_mdhash_id(
            str(entity_name),
            prefix="ent-",
            workspace=global_config.get("workspace", "default"),
        )
        entity_content = f"{entity_name}\n{description}"
        data_for_vdb = {
            entity_vdb_id: {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "content": entity_content,
                "source_id": source_id,
                "file_path": file_path,
                **legal_status_metadata,
            }
        }

    return node_data, full_source_ids, data_for_vdb


def _empty_edge_merge_result():
    return None, None, None, [], {}, [], []


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock: asyncio.Lock = None,
    llm_response_cache: BaseKVStorage = None,
    relation_chunks_storage: BaseKVStorage = None,
    already_edge: dict | None = None,
    existing_full_source_ids: list[str] = None,
    entity_chunks_storage: BaseKVStorage = None,
    new_entity_chunks_to_upsert: dict | None = None,
    entity_vdb: BaseVectorStorage = None,
    existing_nodes_cache: dict[str, dict | None] | None = None,
    existing_entity_chunks_cache: dict[str, list[str] | None] | None = None,
):
    node_updates = []
    node_vdb_updates = {}
    edge_vdb_data = None
    edge_vdb_delete_ids = []
    added_entities = []
    if src_id == tgt_id:
        logger.warning(f"Skipping self-loop relation `{src_id}`~`{tgt_id}`")
        return _empty_edge_merge_result()

    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    # 1. Get existing edge data from graph storage
    if already_edge:
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 1.0 if missing
            already_weights.append(already_edge.get("weight", 1.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    already_edge["source_id"].split(GRAPH_FIELD_SEP)
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    already_edge["file_path"].split(GRAPH_FIELD_SEP)
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.extend(
                    already_edge["description"].split(GRAPH_FIELD_SEP)
                )

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    new_source_ids = [dp["source_id"] for dp in edges_data if dp.get("source_id")]

    is_directed = any(edge.get("directed") for edge in edges_data)
    storage_key = make_relation_chunk_key(src_id, tgt_id, directed=is_directed)

    if existing_full_source_ids is None:
        existing_full_source_ids = []
        if relation_chunks_storage is not None:
            stored_chunks = await relation_chunks_storage.get_by_id(storage_key)
            if stored_chunks and isinstance(stored_chunks, dict):
                existing_full_source_ids = [
                    chunk_id
                    for chunk_id in stored_chunks.get("chunk_ids", [])
                    if chunk_id
                ]

        if not existing_full_source_ids:
            existing_full_source_ids = [
                chunk_id for chunk_id in already_source_ids if chunk_id
            ]

    # 2. Merge new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    # Removed individual relation_chunks_storage.upsert to batch it outside

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_relation")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{src_id}`~`{tgt_id}`",
    )
    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    # 4. Only keep edges with source_id in the final source_ids list if in KEEP mode
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        filtered_edges = []
        for dp in edges_data:
            source_id = dp.get("source_id")
            # Skip relationship fragments sourced from chunks dropped by keep oldest cap
            if (
                source_id
                and source_id not in allowed_source_ids
                and source_id not in existing_full_source_ids
            ):
                continue
            filtered_edges.append(dp)
        edges_data = filtered_edges
    else:  # In FIFO mode, keep all edges - truncation happens at source_ids level only
        edges_data = list(edges_data)

    # 5. Check if we need to skip summary due to source_ids limit
    if (
        limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
        and len(existing_full_source_ids) >= max_source_limit
        and not edges_data
    ):
        if already_edge:
            logger.info(
                f"Skipped `{src_id}`~`{tgt_id}`: KEEP old chunks  {already_source_ids}/{len(full_source_ids)}"
            )
            existing_edge_data = dict(already_edge)
            existing_edge_data.setdefault("src_id", src_id)
            existing_edge_data.setdefault("tgt_id", tgt_id)
            existing_edge_data.setdefault("directed", is_directed)
            return (
                existing_edge_data,
                full_source_ids,
                None,
                [],
                {},
                [],
                [],
            )
        else:
            logger.error(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )
            raise ValueError(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize weight by summing new edges and existing weights
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)

    # 6.2 Finalize keywords by merging existing and new keywords
    all_keywords = set()
    # Process already_keywords (which are comma-separated)
    for keyword_str in already_keywords:
        if keyword_str:  # Skip empty strings
            all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
    # Process new keywords from edges_data
    for edge in edges_data:
        if edge.get("keywords"):
            all_keywords.update(
                k.strip() for k in edge["keywords"].split(",") if k.strip()
            )
    # Join all unique keywords with commas
    keywords = ",".join(sorted(all_keywords))

    # 7. Deduplicate by description, keeping first occurrence in the same document
    unique_edges = {}
    for dp in edges_data:
        description_value = dp.get("description")
        if not description_value:
            continue
        if description_value not in unique_edges:
            unique_edges[description_value] = dp

    # Sort description by timestamp, then by description length (largest to smallest) when timestamps are the same
    sorted_edges = sorted(
        unique_edges.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    sorted_descriptions = [dp["description"] for dp in sorted_edges]

    # Combine already_description with sorted new descriptions
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Relation {src_id}~{tgt_id} has no description")
        raise ValueError(f"Relation {src_id}~{tgt_id} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during relation summary"
                )

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Relation",
        f"({src_id}, {tgt_id})",
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS limit
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Track if already_file_paths contains placeholder

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        # Check if this is a placeholder record
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in edges_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )

        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{src_id}`~`{tgt_id}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)
    legal_status_metadata = best_legal_status_metadata(
        ([already_edge] if already_edge else []) + list(edges_data)
    )

    # 10. Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(edges_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # Truncate description to prevent Milvus Metadata limits (64KB)
    if len(description) > 30000:
        description = description[:30000] + "...(truncated)"

    # 11. Update both graph and vector db
    for need_insert_id in [src_id, tgt_id]:
        if existing_nodes_cache is not None and need_insert_id in existing_nodes_cache:
            existing_node = existing_nodes_cache[need_insert_id]
        else:
            # Optimization: Use get_node instead of has_node + get_node.
            existing_node = await knowledge_graph_inst.get_node(need_insert_id)
            if existing_nodes_cache is not None:
                existing_nodes_cache[need_insert_id] = existing_node

        if existing_node is None:
            # Node doesn't exist - create new node
            node_created_at = int(time.time())
            node_data = {
                "entity_id": need_insert_id,
                "source_id": source_id,
                "description": description,
                "entity_type": "UNKNOWN",
                "file_path": file_path,
                "created_at": node_created_at,
                "truncate": "",
                **legal_status_metadata,
            }
            await knowledge_graph_inst.upsert_node(need_insert_id, node_data=node_data)
            if existing_nodes_cache is not None:
                existing_nodes_cache[need_insert_id] = node_data

            # Update entity_chunks_storage for the newly created entity
            if (
                entity_chunks_storage is not None
                and new_entity_chunks_to_upsert is not None
            ):
                chunk_ids = [chunk_id for chunk_id in full_source_ids if chunk_id]
                if chunk_ids:
                    new_entity_chunks_to_upsert[need_insert_id] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }
                    if existing_entity_chunks_cache is not None:
                        existing_entity_chunks_cache[need_insert_id] = chunk_ids

            if entity_vdb is not None:
                entity_vdb_id = compute_mdhash_id(
                    need_insert_id,
                    prefix="ent-",
                    workspace=global_config.get("workspace", "default"),
                )
                entity_content = f"{need_insert_id}\n{description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": need_insert_id,
                        "source_id": source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": file_path,
                        **legal_status_metadata,
                    }
                }
                node_vdb_updates.update(vdb_data)

                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                    operation_name="added_entity_upsert",
                    entity_name=need_insert_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

            # Track entities added during edge processing
            entity_data = {
                "entity_name": need_insert_id,
                "entity_type": "UNKNOWN",
                "description": description,
                "source_id": source_id,
                "file_path": file_path,
                "created_at": node_created_at,
                **legal_status_metadata,
            }
            added_entities.append(entity_data)
        else:
            # Node exists - update its source_ids by merging with new source_ids
            updated = False  # Track if any update occurred

            # 1. Get existing full source_ids from entity_chunks_storage
            if (
                existing_entity_chunks_cache is not None
                and need_insert_id in existing_entity_chunks_cache
            ):
                existing_full_source_ids = [
                    chunk_id
                    for chunk_id in existing_entity_chunks_cache.get(
                        need_insert_id, []
                    )
                    if chunk_id
                ]
            else:
                existing_full_source_ids = []

            if (
                not existing_full_source_ids
                and entity_chunks_storage is not None
                and (
                    existing_entity_chunks_cache is None
                    or need_insert_id not in existing_entity_chunks_cache
                )
            ):
                stored_chunks = await entity_chunks_storage.get_by_id(need_insert_id)
                if stored_chunks and isinstance(stored_chunks, dict):
                    existing_full_source_ids = [
                        chunk_id
                        for chunk_id in stored_chunks.get("chunk_ids", [])
                        if chunk_id
                    ]
                if existing_entity_chunks_cache is not None:
                    existing_entity_chunks_cache[need_insert_id] = existing_full_source_ids

            # If not in entity_chunks_storage, get from graph database
            if not existing_full_source_ids:
                if existing_node.get("source_id"):
                    existing_full_source_ids = existing_node["source_id"].split(
                        GRAPH_FIELD_SEP
                    )

            # 2. Merge with new source_ids from this relationship
            new_source_ids_from_relation = [
                chunk_id for chunk_id in source_ids if chunk_id
            ]
            merged_full_source_ids = merge_source_ids(
                existing_full_source_ids, new_source_ids_from_relation
            )

            # 3. Save merged full list to entity_chunks_storage (conditional)
            if (
                entity_chunks_storage is not None
                and merged_full_source_ids != existing_full_source_ids
                and new_entity_chunks_to_upsert is not None
            ):
                updated = True
                new_entity_chunks_to_upsert[need_insert_id] = {
                    "chunk_ids": merged_full_source_ids,
                    "count": len(merged_full_source_ids),
                }
                if existing_entity_chunks_cache is not None:
                    existing_entity_chunks_cache[need_insert_id] = merged_full_source_ids

            # 4. Apply source_ids limit for graph and vector db
            limit_method = global_config.get(
                "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
            )
            max_source_limit = global_config.get("max_source_ids_per_entity")
            limited_source_ids = apply_source_ids_limit(
                merged_full_source_ids,
                max_source_limit,
                limit_method,
                identifier=f"`{need_insert_id}`",
            )

            # 5. Update graph database and vector database with limited source_ids (conditional)
            limited_source_id_str = GRAPH_FIELD_SEP.join(limited_source_ids)

            if limited_source_id_str != existing_node.get("source_id", ""):
                updated = True
                updated_node_data = {
                    **existing_node,
                    "source_id": limited_source_id_str,
                }
                node_updates.append((need_insert_id, updated_node_data))
                if existing_nodes_cache is not None:
                    existing_nodes_cache[need_insert_id] = updated_node_data

                # Update vector database
                if entity_vdb is not None:
                    entity_vdb_id = compute_mdhash_id(
                        need_insert_id,
                        prefix="ent-",
                        workspace=global_config.get("workspace", "default"),
                    )
                    entity_content = (
                        f"{need_insert_id}\n{existing_node.get('description', '')}"
                    )
                    vdb_data = {
                        entity_vdb_id: {
                            "content": entity_content,
                            "entity_name": need_insert_id,
                            "source_id": limited_source_id_str,
                            "entity_type": existing_node.get("entity_type", "UNKNOWN"),
                            "file_path": existing_node.get(
                                "file_path", "unknown_source"
                            ),
                        }
                    }
                    node_vdb_updates.update(vdb_data)

            # 6. Log once at the end if any update occurred
            existing_priority = int(existing_node.get("legal_status_priority") or 0)
            new_priority = int(legal_status_metadata.get("legal_status_priority") or 0)
            if legal_status_metadata and new_priority > existing_priority:
                updated = True
                updated_node_data = {
                    **existing_node,
                    **legal_status_metadata,
                }
                node_updates.append((need_insert_id, updated_node_data))
                if existing_nodes_cache is not None:
                    existing_nodes_cache[need_insert_id] = updated_node_data
                if entity_vdb is not None:
                    entity_vdb_id = compute_mdhash_id(
                        need_insert_id,
                        prefix="ent-",
                        workspace=global_config.get("workspace", "default"),
                    )
                    entity_content = (
                        f"{need_insert_id}\n{existing_node.get('description', '')}"
                    )
                    node_vdb_updates[entity_vdb_id] = {
                        "content": entity_content,
                        "entity_name": need_insert_id,
                        "source_id": existing_node.get("source_id", ""),
                        "entity_type": existing_node.get("entity_type", "UNKNOWN"),
                        "file_path": existing_node.get("file_path", "unknown_source"),
                        **legal_status_metadata,
                    }

            if updated:
                status_message = f"Chunks appended from relation: `{need_insert_id}`"
                logger.info(status_message)
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = status_message
                        pipeline_status["history_messages"].append(status_message)

    edge_created_at = int(time.time())

    edge_data = dict(
        weight=weight,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=edge_created_at,
        truncate=truncation_info,
        src_id=src_id,
        tgt_id=tgt_id,
        directed=is_directed,
        **legal_status_metadata,
    )

    # Undirected relations use canonical ordering; directed legal relations keep source -> target.
    if not is_directed and src_id > tgt_id:
        src_id, tgt_id = tgt_id, src_id

    if relationships_vdb is not None:
        rel_vdb_id = compute_mdhash_id(
            src_id + tgt_id,
            prefix="rel-",
            workspace=global_config.get("workspace", "default"),
        )
        rel_vdb_id_reverse = compute_mdhash_id(
            tgt_id + src_id,
            prefix="rel-",
            workspace=global_config.get("workspace", "default"),
        )
        edge_vdb_delete_ids = (
            [rel_vdb_id] if is_directed else [rel_vdb_id, rel_vdb_id_reverse]
        )
        rel_content = f"{keywords}\t{src_id}\n{tgt_id}\n{description}"
        edge_vdb_data = {
            rel_vdb_id: {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": source_id,
                "content": rel_content,
                "keywords": keywords,
                "description": description,
                "weight": weight,
                "file_path": file_path,
                "directed": is_directed,
                **legal_status_metadata,
            }
        }

    return (
        edge_data,
        full_source_ids,
        edge_vdb_data,
        node_updates,
        node_vdb_updates,
        edge_vdb_delete_ids,
        added_entities,
    )


async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,
    full_relations_storage: BaseKVStorage = None,
    doc_id: str = None,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
) -> dict[str, list[dict[str, Any]]]:
    """Two-phase merge: process all entities first, then all relationships

    This approach ensures data consistency by:
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)
    3. Phase 3: Update full_entities and full_relations storage with final results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        full_entities_storage: Storage for document entity lists
        full_relations_storage: Storage for document relation lists
        doc_id: Document ID for storage indexing
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
        entity_chunks_storage: Storage tracking full chunk lists per entity
        relation_chunks_storage: Storage tracking full chunk lists per relation
        current_file_number: Current file number for logging
        total_files: Total files for logging
        file_path: File path for logging
    """

    # Check for cancellation at the start of merge
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during merge phase")

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    def edge_group_key(edge_key: tuple, edges: list[dict]) -> tuple:
        """Keep directed legal edges ordered while canonicalizing regular KG edges."""
        if any(edge.get("directed") for edge in edges if isinstance(edge, dict)):
            for edge in edges:
                if isinstance(edge, dict) and edge.get("src_id") and edge.get("tgt_id"):
                    return edge["src_id"], edge["tgt_id"]
            return edge_key
        return tuple(sorted(edge_key))

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        if isinstance(maybe_nodes, dict):
            for entity_name, entities in maybe_nodes.items():
                all_nodes[entity_name].extend(entities)
        elif isinstance(maybe_nodes, list):
            # Fallback if LLM parser returns a list (or empty list) instead of dict
            for item in maybe_nodes:
                if isinstance(item, dict) and item.get("entity_name"):
                    all_nodes[item["entity_name"]].append(item)

        # Collect edges. Regular KG edges are undirected; legal hierarchy/document edges are directed.
        if isinstance(maybe_edges, dict):
            for edge_key, edges in maybe_edges.items():
                all_edges[edge_group_key(edge_key, edges)].extend(edges)
        elif isinstance(maybe_edges, list):
            # Fallback if LLM parser returns a list (or empty list) instead of dict
            for item in maybe_edges:
                if isinstance(item, dict) and item.get("src_id") and item.get("tgt_id"):
                    if item.get("directed"):
                        edge_key = (item["src_id"], item["tgt_id"])
                    else:
                        edge_key = tuple(sorted([item["src_id"], item["tgt_id"]]))
                    all_edges[edge_key].append(item)

    def _pre_merge_descriptions_in_memory(nodes_data: list[dict]) -> list[dict]:
        """Deduplicate identical descriptions within the batch on RAM."""
        seen_descriptions = set()
        unique_nodes = []
        for node in nodes_data:
            desc = node.get("description", "")
            desc = desc.strip() if desc else ""
            if desc and desc not in seen_descriptions:
                seen_descriptions.add(desc)
                unique_nodes.append(node)
            elif not desc:
                unique_nodes.append(node)
        return unique_nodes

    for entity_name in all_nodes:
        all_nodes[entity_name] = _pre_merge_descriptions_in_memory(
            all_nodes[entity_name]
        )

    for edge_key in all_edges:
        all_edges[edge_key] = _pre_merge_descriptions_in_memory(all_edges[edge_key])
    # ============================================

    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # === NEW: Cross-batch caches ===
    _entity_cache: dict[str, dict | None] = {}
    _entity_chunks_cache: dict[str, list[str] | None] = {}
    _relation_cache: dict[tuple, dict | None] = {}
    _relation_chunks_cache: dict[str, list[str] | None] = {}
    # ===============================

    # ===== Phase 1: Process all entities concurrently in batches =====
    log_message = (
        f"Phase 1: Processing {total_entities_count} entities from {doc_id} (Batching)"
    )
    logger.info(log_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

    entity_names = list(all_nodes.keys())
    workspace = global_config.get("workspace", "")
    namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"

    processed_entities = []
    BATCH_SIZE = 1000

    for i in range(0, len(entity_names), BATCH_SIZE):
        batch_entity_names = entity_names[i : i + BATCH_SIZE]

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    raise PipelineCancelledException(
                        "User cancelled during entity merge"
                    )

        async def _process_entity_batch(names_batch):
            async with get_storage_keyed_lock(
                names_batch, namespace=namespace, enable_logging=False
            ):
                # Fetch missing from cache
                uncached_names = [n for n in names_batch if n not in _entity_cache]
                if uncached_names:
                    fetched_nodes = await knowledge_graph_inst.get_nodes_batch(
                        uncached_names
                    )
                    _entity_cache.update(fetched_nodes)
                    for n in uncached_names:
                        if n not in _entity_cache:
                            _entity_cache[n] = None

                    if entity_chunks_storage is not None:
                        chunk_results = await entity_chunks_storage.get_by_ids(
                            uncached_names
                        )
                        for idx, result in enumerate(chunk_results):
                            name = uncached_names[idx]
                            if result and isinstance(result, dict):
                                _entity_chunks_cache[name] = [
                                    chunk_id
                                    for chunk_id in result.get("chunk_ids", [])
                                    if chunk_id
                                ]
                            else:
                                _entity_chunks_cache[name] = []

                entity_chunks_to_upsert = {}
                batch_processed_entities = []

                async def _locked_process_entity_name(entity_name, entities):
                    async with semaphore:
                        try:
                            logger.debug(f"Processing entity {entity_name}")
                            already_node = _entity_cache.get(entity_name)
                            existing_full_source_ids = _entity_chunks_cache.get(
                                entity_name
                            )

                            (
                                node_data,
                                full_source_ids,
                                data_for_vdb,
                            ) = await _merge_nodes_then_upsert(
                                entity_name,
                                entities,
                                knowledge_graph_inst,
                                entity_vdb,
                                global_config,
                                pipeline_status,
                                pipeline_status_lock,
                                llm_response_cache,
                                entity_chunks_storage,
                                already_node=already_node,
                                existing_full_source_ids=existing_full_source_ids,
                            )
                            return entity_name, node_data, full_source_ids, data_for_vdb

                        except Exception as e:
                            error_msg = f"Error processing entity `{entity_name}`: {e}"
                            logger.error(error_msg)
                            try:
                                if (
                                    pipeline_status is not None
                                    and pipeline_status_lock is not None
                                ):
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = error_msg
                                        pipeline_status["history_messages"].append(
                                            error_msg
                                        )
                            except Exception as status_error:
                                logger.error(
                                    f"Failed to update pipeline status: {status_error}"
                                )
                            prefixed_exception = create_prefixed_exception(
                                e, f"`{entity_name}`"
                            )
                            raise prefixed_exception from e

                tasks = [
                    asyncio.create_task(
                        _locked_process_entity_name(name, all_nodes[name])
                    )
                    for name in names_batch
                ]

                results = await asyncio.gather(*tasks, return_exceptions=False)

                batch_nodes_data = []
                batch_vdb_data = {}
                for entity_name, node_data, full_source_ids, data_for_vdb in results:
                    batch_processed_entities.append(node_data)
                    batch_nodes_data.append((entity_name, node_data))
                    if data_for_vdb:
                        batch_vdb_data.update(data_for_vdb)

                    if entity_chunks_storage is not None and full_source_ids:
                        entity_chunks_to_upsert[entity_name] = {
                            "chunk_ids": full_source_ids,
                            "count": len(full_source_ids),
                        }

                if entity_chunks_storage is not None and entity_chunks_to_upsert:
                    await entity_chunks_storage.upsert(entity_chunks_to_upsert)
                    for name, chunk_data in entity_chunks_to_upsert.items():
                        _entity_chunks_cache[name] = chunk_data["chunk_ids"]

                # Batch upsert to Graph DB
                if batch_nodes_data:
                    await knowledge_graph_inst.upsert_nodes(batch_nodes_data)
                    for name, node_data in batch_nodes_data:
                        _entity_cache[name] = node_data

                # Batch upsert to Vector DB
                if entity_vdb is not None and batch_vdb_data:

                    def operation_func(payload=batch_vdb_data):
                        if hasattr(entity_vdb, "bulk_upsert") and len(payload) >= 100:
                            return entity_vdb.bulk_upsert(payload)
                        return entity_vdb.upsert(payload)

                    await safe_vdb_operation_with_exception(
                        operation=operation_func,
                        operation_name="batch_entity_upsert",
                        entity_name=f"batch_{len(batch_vdb_data)}",
                        max_retries=3,
                        retry_delay=0.1,
                    )

                return batch_processed_entities

        batch_results = await _process_entity_batch(batch_entity_names)
        processed_entities.extend(batch_results)

    # ===== Phase 2: Process all relationships concurrently in batches =====
    log_message = f"Phase 2: Processing {total_relations_count} relations from {doc_id} (Batching)"
    logger.info(log_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

    edge_keys = list(all_edges.keys())
    processed_edges = []
    all_added_entities = []

    for i in range(0, len(edge_keys), BATCH_SIZE):
        batch_edge_keys = edge_keys[i : i + BATCH_SIZE]

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    raise PipelineCancelledException(
                        "User cancelled during relation merge"
                    )

        async def _process_edge_batch(keys_batch):
            flat_lock_keys = []
            for edge_key in keys_batch:
                flat_lock_keys.extend(edge_key)
            flat_lock_keys = list(set(flat_lock_keys))

            async with get_storage_keyed_lock(
                flat_lock_keys, namespace=namespace, enable_logging=False
            ):
                pairs_to_fetch = []
                for k in keys_batch:
                    if (k[0], k[1]) not in _relation_cache and (
                        k[1],
                        k[0],
                    ) not in _relation_cache:
                        pairs_to_fetch.append({"src": k[0], "tgt": k[1]})

                if pairs_to_fetch:
                    fetched_edges = await knowledge_graph_inst.get_edges_batch(
                        pairs_to_fetch
                    )
                    for (src, tgt), edge_data in fetched_edges.items():
                        _relation_cache[(src, tgt)] = edge_data
                    for pair in pairs_to_fetch:
                        k_f = (pair["src"], pair["tgt"])
                        if (
                            k_f not in _relation_cache
                            and (k_f[1], k_f[0]) not in _relation_cache
                        ):
                            _relation_cache[k_f] = None

                uncached_endpoint_names = [
                    name for name in flat_lock_keys if name not in _entity_cache
                ]
                if uncached_endpoint_names:
                    fetched_nodes = await knowledge_graph_inst.get_nodes_batch(
                        uncached_endpoint_names
                    )
                    _entity_cache.update(fetched_nodes)
                    for name in uncached_endpoint_names:
                        if name not in _entity_cache:
                            _entity_cache[name] = None

                uncached_endpoint_chunk_names = [
                    name for name in flat_lock_keys if name not in _entity_chunks_cache
                ]
                if entity_chunks_storage is not None and uncached_endpoint_chunk_names:
                    chunk_results = await entity_chunks_storage.get_by_ids(
                        uncached_endpoint_chunk_names
                    )
                    for idx, result in enumerate(chunk_results):
                        name = uncached_endpoint_chunk_names[idx]
                        if result and isinstance(result, dict):
                            _entity_chunks_cache[name] = [
                                chunk_id
                                for chunk_id in result.get("chunk_ids", [])
                                if chunk_id
                            ]
                        else:
                            _entity_chunks_cache[name] = []

                storage_keys = []
                for edge_key in keys_batch:
                    is_directed = any(
                        edge.get("directed") for edge in all_edges[edge_key]
                    )
                    storage_keys.append(
                        make_relation_chunk_key(
                            edge_key[0], edge_key[1], directed=is_directed
                        )
                    )

                uncached_storage_keys = [
                    k for k in storage_keys if k not in _relation_chunks_cache
                ]
                if relation_chunks_storage is not None and uncached_storage_keys:
                    chunk_results = await relation_chunks_storage.get_by_ids(
                        uncached_storage_keys
                    )
                    for idx, result in enumerate(chunk_results):
                        s_key = uncached_storage_keys[idx]
                        if result and isinstance(result, dict):
                            _relation_chunks_cache[s_key] = [
                                chunk_id
                                for chunk_id in result.get("chunk_ids", [])
                                if chunk_id
                            ]
                        else:
                            _relation_chunks_cache[s_key] = []

                new_entity_chunks_to_upsert = {}
                batch_processed_edges = []
                batch_added_entities = []

                async def _locked_process_edge(edge_key, edges, storage_key):
                    async with semaphore:
                        try:
                            logger.debug(f"Processing relation {edge_key}")
                            already_edge = _relation_cache.get(
                                (edge_key[0], edge_key[1])
                            )
                            if not already_edge:
                                already_edge = _relation_cache.get(
                                    (edge_key[1], edge_key[0])
                                )

                            existing_full_source_ids = _relation_chunks_cache.get(
                                storage_key
                            )

                            merge_res = await _merge_edges_then_upsert(
                                src_id=edge_key[0],
                                tgt_id=edge_key[1],
                                edges_data=edges,
                                knowledge_graph_inst=knowledge_graph_inst,
                                relationships_vdb=relationships_vdb,
                                global_config=global_config,
                                pipeline_status=pipeline_status,
                                pipeline_status_lock=pipeline_status_lock,
                                llm_response_cache=llm_response_cache,
                                relation_chunks_storage=relation_chunks_storage,
                                already_edge=already_edge,
                                existing_full_source_ids=existing_full_source_ids,
                                entity_chunks_storage=entity_chunks_storage,
                                new_entity_chunks_to_upsert=new_entity_chunks_to_upsert,
                                entity_vdb=entity_vdb,
                                existing_nodes_cache=_entity_cache,
                                existing_entity_chunks_cache=_entity_chunks_cache,
                            )
                            if merge_res is None:
                                merge_res = _empty_edge_merge_result()

                            if not (
                                isinstance(merge_res, tuple) and len(merge_res) == 7
                            ):
                                logger.error(
                                    f"Invalid merge result for relation `{edge_key[0]}`~`{edge_key[1]}`: {type(merge_res).__name__}"
                                )
                                merge_res = _empty_edge_merge_result()

                            (
                                edge_data,
                                full_source_ids,
                                edge_vdb_data,
                                node_updates,
                                node_vdb_updates,
                                edge_vdb_delete_ids,
                                added_entities,
                            ) = merge_res
                            return (
                                edge_key,
                                storage_key,
                                edge_data,
                                full_source_ids,
                                edge_vdb_data,
                                node_updates,
                                node_vdb_updates,
                                edge_vdb_delete_ids,
                                added_entities,
                            )

                        except Exception as e:
                            sorted_edge_key = sorted([edge_key[0], edge_key[1]])
                            error_msg = (
                                f"Error processing relation `{sorted_edge_key}`: {e}"
                            )
                            logger.error(error_msg)
                            try:
                                if (
                                    pipeline_status is not None
                                    and pipeline_status_lock is not None
                                ):
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = error_msg
                                        pipeline_status["history_messages"].append(
                                            error_msg
                                        )
                            except Exception as status_error:
                                logger.error(
                                    f"Failed to update pipeline status: {status_error}"
                                )
                            prefixed_exception = create_prefixed_exception(
                                e, f"{sorted_edge_key}"
                            )
                            raise prefixed_exception from e

                tasks = []
                for idx, edge_key in enumerate(keys_batch):
                    tasks.append(
                        asyncio.create_task(
                            _locked_process_edge(
                                edge_key, all_edges[edge_key], storage_keys[idx]
                            )
                        )
                    )

                results = await asyncio.gather(*tasks, return_exceptions=False)

                relation_chunks_to_upsert = {}
                batch_edges_data = []
                batch_edge_vdb_data = {}
                batch_edge_vdb_delete_ids = []
                batch_node_updates = []
                batch_node_vdb_updates = {}

                for (
                    edge_key,
                    storage_key,
                    edge_data,
                    full_source_ids,
                    edge_vdb_data,
                    node_updates,
                    node_vdb_updates,
                    edge_vdb_delete_ids,
                    added_entities,
                ) in results:
                    batch_added_entities.extend(added_entities)
                    if edge_data is not None:
                        batch_processed_edges.append(edge_data)
                        batch_edges_data.append(
                            (edge_data["src_id"], edge_data["tgt_id"], edge_data)
                        )
                    if edge_vdb_data:
                        batch_edge_vdb_data.update(edge_vdb_data)
                    if edge_vdb_delete_ids:
                        batch_edge_vdb_delete_ids.extend(edge_vdb_delete_ids)
                    if node_updates:
                        batch_node_updates.extend(node_updates)
                    if node_vdb_updates:
                        batch_node_vdb_updates.update(node_vdb_updates)

                    if relation_chunks_storage is not None and full_source_ids:
                        relation_chunks_to_upsert[storage_key] = {
                            "chunk_ids": full_source_ids,
                            "count": len(full_source_ids),
                        }

                if relation_chunks_storage is not None and relation_chunks_to_upsert:
                    await relation_chunks_storage.upsert(relation_chunks_to_upsert)
                    for s_key, chunk_data in relation_chunks_to_upsert.items():
                        _relation_chunks_cache[s_key] = chunk_data["chunk_ids"]

                if entity_chunks_storage is not None and new_entity_chunks_to_upsert:
                    await entity_chunks_storage.upsert(new_entity_chunks_to_upsert)
                    for name, chunk_data in new_entity_chunks_to_upsert.items():
                        _entity_chunks_cache[name] = chunk_data["chunk_ids"]

                # Batch Graph DB updates
                if batch_node_updates:
                    await knowledge_graph_inst.upsert_nodes(batch_node_updates)
                    for name, node_data in batch_node_updates:
                        _entity_cache[name] = node_data
                if batch_edges_data:
                    await knowledge_graph_inst.upsert_edges(batch_edges_data)
                    for src, tgt, edge_data in batch_edges_data:
                        _relation_cache[(src, tgt)] = edge_data

                # Batch Vector DB updates
                if relationships_vdb is not None:
                    if batch_edge_vdb_delete_ids:
                        try:
                            await relationships_vdb.delete(batch_edge_vdb_delete_ids)
                        except Exception as e:
                            logger.debug(
                                f"Could not delete old relationship vector records: {e}"
                            )
                    if batch_edge_vdb_data:

                        def edge_operation_func(payload=batch_edge_vdb_data):
                            if (
                                hasattr(relationships_vdb, "bulk_upsert")
                                and len(payload) >= 100
                            ):
                                return relationships_vdb.bulk_upsert(payload)
                            return relationships_vdb.upsert(payload)

                        await safe_vdb_operation_with_exception(
                            operation=edge_operation_func,
                            operation_name="batch_relationship_upsert",
                            entity_name=f"batch_{len(batch_edge_vdb_data)}",
                            max_retries=3,
                            retry_delay=0.1,
                        )

                if entity_vdb is not None and batch_node_vdb_updates:

                    def node_operation_func(payload=batch_node_vdb_updates):
                        if hasattr(entity_vdb, "bulk_upsert") and len(payload) >= 100:
                            return entity_vdb.bulk_upsert(payload)
                        return entity_vdb.upsert(payload)

                    await safe_vdb_operation_with_exception(
                        operation=node_operation_func,
                        operation_name="batch_node_vdb_update",
                        entity_name=f"batch_{len(batch_node_vdb_updates)}",
                        max_retries=3,
                        retry_delay=0.1,
                    )

                return batch_processed_edges, batch_added_entities

        batch_edges, batch_entities = await _process_edge_batch(batch_edge_keys)
        processed_edges.extend(batch_edges)
        all_added_entities.extend(batch_entities)

    # ===== Phase 3: Update full_entities and full_relations storage =====
    if full_entities_storage and full_relations_storage and doc_id:
        try:
            # Merge all entities: original entities + entities added during edge processing
            final_entity_names = set()

            # Add original processed entities
            for entity_data in processed_entities:
                if entity_data and entity_data.get("entity_name"):
                    final_entity_names.add(entity_data["entity_name"])

            # Add entities that were added during relationship processing
            for added_entity in all_added_entities:
                if added_entity and added_entity.get("entity_name"):
                    final_entity_names.add(added_entity["entity_name"])

            # Collect all relation pairs
            final_relation_pairs = set()
            for edge_data in processed_edges:
                if edge_data:
                    src_id = edge_data.get("src_id")
                    tgt_id = edge_data.get("tgt_id")
                    if src_id and tgt_id:
                        if edge_data.get("directed"):
                            relation_pair = (src_id, tgt_id)
                        else:
                            relation_pair = tuple(sorted([src_id, tgt_id]))
                        final_relation_pairs.add(relation_pair)

            log_message = f"Phase 3: Updating final {len(final_entity_names)}({len(processed_entities)}+{len(all_added_entities)}) entities and  {len(final_relation_pairs)} relations from {doc_id}"
            logger.info(log_message)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            # Update storage
            if final_entity_names:
                await full_entities_storage.upsert(
                    {
                        doc_id: {
                            "entity_names": list(final_entity_names),
                            "count": len(final_entity_names),
                        }
                    }
                )

            if final_relation_pairs:
                await full_relations_storage.upsert(
                    {
                        doc_id: {
                            "relation_pairs": [
                                list(pair) for pair in final_relation_pairs
                            ],
                            "count": len(final_relation_pairs),
                        }
                    }
                )

            logger.debug(
                f"Updated entity-relation index for document {doc_id}: {len(final_entity_names)} entities (original: {len(processed_entities)}, added: {len(all_added_entities)}), {len(final_relation_pairs)} relations"
            )

            # ===== [NEW PostgreSQL Dual-Write relational mapping block] =====
            try:
                from lightrag.kg.postgres_impl import ClientManager
                import uuid
                import asyncpg

                # 1. Fetch ClientManager shared pool and verify active connection
                db = await ClientManager.get_client()
                if db is not None:
                    workspace_id = global_config.get("workspace", "")
                    if not workspace_id:
                        workspace_id = "default"

                    # 1.1 Helper to safely convert keys to UUID objects
                    def to_uuid(val: Any) -> uuid.UUID:
                        if isinstance(val, uuid.UUID):
                            return val
                        if not isinstance(val, str):
                            return uuid.UUID(str(val))
                        for prefix in [
                            "chunk-",
                            "ent-",
                            "rel-",
                            "doc-",
                            "node-",
                            "edge-",
                        ]:
                            if val.startswith(prefix):
                                val = val[len(prefix) :]
                                break
                        clean_val = val.replace("-", "")
                        if len(clean_val) == 32:
                            try:
                                return uuid.UUID(clean_val)
                            except ValueError:
                                pass
                        return uuid.uuid5(uuid.NAMESPACE_DNS, str(val))

                    ws_uuid = to_uuid(workspace_id)
                    doc_uuid = to_uuid(doc_id)
                    logger.info(
                        f"[{doc_id}] PostgreSQL client acquired for dual-write mapping. Workspace UUID: {ws_uuid}, Doc UUID: {doc_uuid}"
                    )

                    # 1.2 Prepare Entity & Chunk-Entity mapping lists
                    entity_insert_data = []
                    chunk_entity_insert_data = []
                    seen_entities = set()

                    for entity_data in processed_entities + all_added_entities:
                        if not entity_data or not entity_data.get("entity_name"):
                            continue
                        entity_name = entity_data["entity_name"]
                        if entity_name in seen_entities:
                            continue
                        seen_entities.add(entity_name)

                        entity_uuid = to_uuid(
                            compute_mdhash_id(
                                entity_name, prefix="ent-", workspace=workspace_id
                            )
                        )
                        entity_insert_data.append((entity_uuid, ws_uuid, entity_name))

                        # Map associated chunk IDs (from source_id)
                        source_id_str = entity_data.get("source_id", "")
                        if source_id_str:
                            for chunk_id_str in source_id_str.split(GRAPH_FIELD_SEP):
                                if chunk_id_str.strip():
                                    chunk_uuid = to_uuid(chunk_id_str.strip())
                                    chunk_entity_insert_data.append(
                                        (chunk_uuid, entity_uuid)
                                    )

                    # 1.3 Prepare Relation & Chunk-Relation mapping lists
                    relation_insert_data = []
                    chunk_relation_insert_data = []
                    seen_relations = set()

                    for edge_data in processed_edges:
                        if not edge_data:
                            continue
                        src_name = edge_data["src_id"]
                        tgt_name = edge_data["tgt_id"]
                        relation_type = edge_data.get("keywords", "UNKNOWN")

                        # Ensure any referenced source or target entities exist in tbl_lightrag_entity
                        for ent_name in [src_name, tgt_name]:
                            if ent_name not in seen_entities:
                                seen_entities.add(ent_name)
                                entity_uuid = to_uuid(
                                    compute_mdhash_id(
                                        ent_name, prefix="ent-", workspace=workspace_id
                                    )
                                )
                                entity_insert_data.append(
                                    (entity_uuid, ws_uuid, ent_name)
                                )

                        # Undirected relations use canonical ordering; directed legal
                        # relations keep source -> target.
                        if not edge_data.get("directed") and src_name > tgt_name:
                            src_name, tgt_name = tgt_name, src_name

                        relation_uuid = to_uuid(
                            compute_mdhash_id(
                                src_name + tgt_name,
                                prefix="rel-",
                                workspace=workspace_id,
                            )
                        )

                        if relation_uuid in seen_relations:
                            continue
                        seen_relations.add(relation_uuid)

                        src_entity_uuid = to_uuid(
                            compute_mdhash_id(
                                src_name, prefix="ent-", workspace=workspace_id
                            )
                        )
                        tgt_entity_uuid = to_uuid(
                            compute_mdhash_id(
                                tgt_name, prefix="ent-", workspace=workspace_id
                            )
                        )

                        relation_insert_data.append(
                            (
                                relation_uuid,
                                ws_uuid,
                                src_entity_uuid,
                                tgt_entity_uuid,
                                relation_type,
                            )
                        )

                        # Map associated chunk IDs (from source_id)
                        source_id_str = edge_data.get("source_id", "")
                        if source_id_str:
                            for chunk_id_str in source_id_str.split(GRAPH_FIELD_SEP):
                                if chunk_id_str.strip():
                                    chunk_uuid = to_uuid(chunk_id_str.strip())
                                    chunk_relation_insert_data.append(
                                        (chunk_uuid, relation_uuid)
                                    )

                    # All document types (including legal) use the shared lightrag tables
                    tbl_entity = "tbl_lightrag_entity"
                    tbl_relation = "tbl_lightrag_relation"
                    tbl_chunk = "tbl_lightrag_document_chunk"
                    tbl_chunk_entity = "tbl_chunk_entity"
                    tbl_chunk_relation = "tbl_chunk_relation"

                    # 1.4 Perform Transactional bulk insert operations with automatic retries
                    async def _run_inserts(connection: asyncpg.Connection) -> None:
                        async with connection.transaction():
                            # Insert entities
                            if entity_insert_data:
                                await connection.execute("""
                                    CREATE TEMP TABLE _tmp_entity (
                                        id UUID, workspace_id UUID, name VARCHAR(1000)
                                    ) ON COMMIT DROP
                                """)
                                await connection.copy_records_to_table(
                                    "_tmp_entity", records=entity_insert_data
                                )
                                await connection.execute(f"""
                                    INSERT INTO {tbl_entity} (id, workspace_id, name, create_time, update_time)
                                    SELECT id, workspace_id, name, NOW(), NOW() FROM _tmp_entity
                                    ON CONFLICT (id) DO UPDATE SET update_time = NOW()
                                """)

                            # Insert relations
                            if relation_insert_data:
                                await connection.execute("""
                                    CREATE TEMP TABLE _tmp_relation (
                                        id UUID, workspace_id UUID, 
                                        source_entity_id UUID, target_entity_id UUID,
                                        relation_type VARCHAR(1000)
                                    ) ON COMMIT DROP
                                """)
                                await connection.copy_records_to_table(
                                    "_tmp_relation", records=relation_insert_data
                                )
                                await connection.execute(f"""
                                    INSERT INTO {tbl_relation} (id, workspace_id, source_entity_id, target_entity_id, relation_type, create_time, update_time)
                                    SELECT id, workspace_id, source_entity_id, target_entity_id, relation_type, NOW(), NOW()
                                    FROM _tmp_relation
                                    ON CONFLICT (id) DO UPDATE SET update_time = NOW()
                                """)

                            # Insert chunk-entity mappings
                            if chunk_entity_insert_data:
                                await connection.execute("""
                                    CREATE TEMP TABLE _tmp_chunk_entity (
                                        chunk_id UUID, entity_id UUID
                                    ) ON COMMIT DROP
                                """)
                                await connection.copy_records_to_table(
                                    "_tmp_chunk_entity", records=chunk_entity_insert_data
                                )
                                await connection.execute(f"""
                                    INSERT INTO {tbl_chunk_entity} (chunk_id, entity_id)
                                    SELECT t.chunk_id, t.entity_id
                                    FROM _tmp_chunk_entity t
                                    JOIN {tbl_chunk} c ON c.id = t.chunk_id
                                    JOIN {tbl_entity} e ON e.id = t.entity_id
                                    ON CONFLICT (chunk_id, entity_id) DO NOTHING
                                """)

                            # Insert chunk-relation mappings
                            if chunk_relation_insert_data:
                                await connection.execute("""
                                    CREATE TEMP TABLE _tmp_chunk_relation (
                                        chunk_id UUID, relation_id UUID
                                    ) ON COMMIT DROP
                                """)
                                await connection.copy_records_to_table(
                                    "_tmp_chunk_relation",
                                    records=chunk_relation_insert_data,
                                )
                                await connection.execute(f"""
                                    INSERT INTO {tbl_chunk_relation} (chunk_id, relation_id)
                                    SELECT t.chunk_id, t.relation_id
                                    FROM _tmp_chunk_relation t
                                    JOIN {tbl_chunk} c ON c.id = t.chunk_id
                                    JOIN {tbl_relation} r ON r.id = t.relation_id
                                    ON CONFLICT (chunk_id, relation_id) DO NOTHING
                                """)

                    await db._run_with_retry(_run_inserts)
                    logger.info(
                        f"[{doc_id}] PostgreSQL relational tables dual-write successfully completed: "
                        f"{len(entity_insert_data)} entities, {len(relation_insert_data)} relations."
                    )

            except Exception as dwe:
                logger.error(
                    f"[{doc_id}] PostgreSQL dual-write exception: {dwe}", exc_info=True
                )

        except Exception as e:
            logger.error(
                f"Failed to update entity-relation index for document {doc_id}: {e}"
            )
            # Don't raise exception to avoid affecting main flow

    log_message = f"Completed merging: {len(processed_entities)} entities, {len(all_added_entities)} extra entities, {len(processed_edges)} relations"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    return {
        "entities": processed_entities,
        "added_entities": all_added_entities,
        "relations": processed_edges,
    }


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
    prompt_prefix: str = "entity_extraction",
    structured_response_format: Any = None,
    force_directed_relations: bool = False,
    conditional_gleaning: bool = False,
) -> list:
    """Extract entities and relationships from text chunks using LLM.

    This function processes text chunks to extract knowledge graph entities and relationships
    using a multi-pass gleaning approach. It supports cancellation, caching, and progress tracking.

    Args:
        chunks: Dictionary mapping chunk IDs to TextChunkSchema objects containing:
            - content (str): The text content to process
            - tokens (int): Token count
            - full_doc_id (str): Document identifier
            - chunk_order_index (int): Order in document
            - file_path (str, optional): Source file path
        global_config: Configuration dictionary containing:
            - llm_model_func: LLM function for entity extraction
            - entity_extract_max_gleaning: Maximum gleaning iterations
            - tokenizer: Tokenizer instance
            - addon_params: Additional parameters (language, entity_types)
        pipeline_status: Optional status dictionary for tracking progress and cancellation
        pipeline_status_lock: Optional lock for thread-safe status updates
        llm_response_cache: Optional cache storage for LLM responses
        text_chunks_storage: Optional storage for persisting chunk extraction results

    Returns:
        List of tuples containing:
            - maybe_nodes (list): Extracted entity nodes with metadata
            - maybe_edges (list): Extracted relationship edges with metadata

    Raises:
        PipelineCancelledException: If cancellation is requested via pipeline_status

    Notes:
        - Uses multi-pass gleaning to improve extraction quality
        - Supports language-specific prompts via PromptManager
        - Caches LLM responses to avoid redundant API calls
        - Updates pipeline status with progress information
    """
    # Check for cancellation at the start of entity extraction
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during entity extraction"
                )

    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    addon_params = global_config.get("addon_params", {})

    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    skip_failed_chunks = bool(
        addon_params.get("entity_extraction_skip_failed_chunks")
    ) or _env_bool("ENTITY_EXTRACTION_SKIP_FAILED_CHUNKS", False)
    max_failed_chunk_ratio = _env_float(
        "ENTITY_EXTRACTION_MAX_FAILED_CHUNK_RATIO", 0.35
    )
    fail_on_all_chunks_failed = _env_bool(
        "ENTITY_EXTRACTION_FAIL_ON_ALL_CHUNKS_FAILED", True
    )
    fail_on_too_many_skipped_chunks = _env_bool(
        "ENTITY_EXTRACTION_FAIL_ON_TOO_MANY_SKIPPED_CHUNKS", False
    )
    legal_structured_max_input_tokens = _env_int(
        "LEGAL_STRUCTURED_MAX_INPUT_TOKENS", 650
    )

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)
    entity_types = global_config["addon_params"].get(
        "entity_types", DEFAULT_ENTITY_TYPES
    )

    # Initialize PromptManager for language-aware prompt retrieval
    prompt_manager = PromptManager(PROMPTS)

    # Determine language for prompts based on language setting
    prompt_language = (
        SupportedLanguage.VIETNAMESE
        if language == "Vietnamese"
        else SupportedLanguage.ENGLISH
    )

    # Get examples using PromptManager
    examples_key = f"{prompt_prefix}_examples"
    examples_prompt = prompt_manager.get_prompt(examples_key, language=prompt_language)
    examples = (
        "\n".join(examples_prompt)
        if isinstance(examples_prompt, list)
        else examples_prompt
    )

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    processed_chunks = 0
    total_chunks = len(ordered_chunks)
    failed_chunks = 0
    failed_chunk_errors: list[str] = []
    tokenizer = global_config.get("tokenizer")
    extract_context_limit = max(
        1, _env_int("LLM_EXTRACT_MODEL_CONTEXT_TOKENS", 16384)
    )
    extract_context_safety_margin = max(
        0, _env_int("LLM_EXTRACT_CONTEXT_SAFETY_MARGIN", 512)
    )
    extract_min_output_tokens = max(
        1, _env_int("ENTITY_EXTRACTION_MIN_OUTPUT_TOKENS", 512)
    )

    def _env_int_optional(name: str) -> int | None:
        value = os.getenv(name)
        if value is None or not value.strip():
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _extract_reserved_output_tokens() -> int:
        for env_name in (
            "OPENAI_LLM_MAX_COMPLETION_TOKENS_EXTRACT",
            "OPENAI_LLM_MAX_TOKENS_EXTRACT",
            "OPENAI_LLM_DEFAULT_MAX_COMPLETION_TOKENS_EXTRACT",
            "OPENAI_LLM_DEFAULT_MAX_TOKENS_EXTRACT",
        ):
            value = _env_int_optional(env_name)
            if value is not None and value > 0:
                return value
        return 1024

    extract_reserved_output_tokens = max(
        extract_min_output_tokens, _extract_reserved_output_tokens()
    )

    def _prompt_input_tokens(
        system_prompt: str,
        user_prompt: str,
        history_messages: list[dict[str, str]] | None = None,
    ) -> int:
        if tokenizer is None:
            return 0
        prompt_parts = [system_prompt or "", user_prompt or ""]
        if history_messages:
            prompt_parts.append(json.dumps(history_messages, ensure_ascii=False))
        return len(tokenizer.encode("\n".join(prompt_parts)))

    def _available_content_tokens(
        system_prompt: str,
        empty_user_prompt: str,
    ) -> int:
        prompt_overhead_tokens = _prompt_input_tokens(system_prompt, empty_user_prompt)
        return (
            extract_context_limit
            - extract_context_safety_margin
            - extract_reserved_output_tokens
            - prompt_overhead_tokens
        )

    def _request_fits_context(
        system_prompt: str,
        user_prompt: str,
        history_messages: list[dict[str, str]] | None = None,
    ) -> tuple[bool, int, int]:
        input_tokens = _prompt_input_tokens(system_prompt, user_prompt, history_messages)
        requested_tokens = (
            input_tokens
            + extract_reserved_output_tokens
            + extract_context_safety_margin
        )
        return requested_tokens <= extract_context_limit, input_tokens, requested_tokens

    def _split_content_by_token_budget(
        content: str,
        content_token_budget: int,
    ) -> list[str]:
        if tokenizer is None:
            return [content]
        if content_token_budget < 1:
            raise RuntimeError(
                "Entity extraction prompt overhead leaves no room for chunk content: "
                f"context_limit={extract_context_limit}, "
                f"reserved_output={extract_reserved_output_tokens}, "
                f"safety_margin={extract_context_safety_margin}"
            )
        tokens = tokenizer.encode(content)
        if len(tokens) <= content_token_budget:
            return [content]
        return [
            tokenizer.decode(tokens[start : start + content_token_budget])
            for start in range(0, len(tokens), content_token_budget)
        ]

    def _merge_extraction_maps(target: dict, source: dict | list) -> None:
        if isinstance(source, dict):
            for key, values in source.items():
                if isinstance(values, list):
                    target[key].extend(values)
                elif values:
                    target[key].append(values)
        elif isinstance(source, list):
            for item in source:
                if not isinstance(item, dict):
                    continue
                item_key = item.get("entity_name") or (
                    item.get("src_id"),
                    item.get("tgt_id"),
                )
                if item_key:
                    target[item_key].append(item)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process one source chunk, splitting internally when the LLM prompt is too large."""
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        file_path = chunk_dp.get("file_path", "unknown_source")
        cache_keys_collector = []

        system_prompt_key = f"{prompt_prefix}_system_prompt"
        user_prompt_key = f"{prompt_prefix}_user_prompt"
        continue_prompt_key = (
            prompt_prefix.replace("entity_extraction", "entity_continue_extraction")
            + "_user_prompt"
        )
        delimiter_system_prompt_template = prompt_manager.get_prompt(
            system_prompt_key, language=prompt_language
        )
        delimiter_user_prompt_template = prompt_manager.get_prompt(
            user_prompt_key, language=prompt_language
        )
        delimiter_continue_prompt_template = prompt_manager.get_prompt(
            continue_prompt_key, language=prompt_language
        )
        delimiter_system_prompt = delimiter_system_prompt_template.format(
            **context_base
        )
        empty_delimiter_user_prompt = delimiter_user_prompt_template.format(
            **{**context_base, "input_text": ""}
        )

        def _build_segment_prompts(segment_content: str) -> dict[str, Any]:
            delimiter_user_prompt = delimiter_user_prompt_template.format(
                **{**context_base, "input_text": segment_content}
            )
            delimiter_continue_prompt = delimiter_continue_prompt_template.format(
                **{**context_base, "input_text": segment_content}
            )
            segment_token_count = int(chunk_dp.get("tokens") or 0)
            if tokenizer is not None:
                segment_token_count = len(tokenizer.encode(segment_content))

            use_structured_output = (
                structured_response_format is not None
                and prompt_prefix.startswith("legal_entity_extraction")
                and legal_structured_max_input_tokens > 0
                and segment_token_count <= legal_structured_max_input_tokens
            )
            system_prompt = delimiter_system_prompt
            user_prompt = delimiter_user_prompt
            continue_prompt = delimiter_continue_prompt
            if use_structured_output:
                system_prompt, user_prompt = build_structured_legal_extraction_prompts(
                    segment_content, entity_types, language
                )
                continue_prompt = build_structured_legal_continue_prompt(
                    segment_content, language
                )
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "continue_prompt": continue_prompt,
                "delimiter_system_prompt": delimiter_system_prompt,
                "delimiter_user_prompt": delimiter_user_prompt,
                "delimiter_continue_prompt": delimiter_continue_prompt,
                "use_structured_output": use_structured_output,
            }

        initial_prompts = _build_segment_prompts(content)
        fits_context, input_tokens, requested_tokens = _request_fits_context(
            initial_prompts["system_prompt"], initial_prompts["user_prompt"]
        )
        if not fits_context and tokenizer is not None:
            content_token_budget = _available_content_tokens(
                delimiter_system_prompt, empty_delimiter_user_prompt
            )
            segments = _split_content_by_token_budget(content, content_token_budget)
            logger.warning(
                "%s: splitting entity extraction chunk into %s segments "
                "(input_tokens=%s, requested=%s, context_limit=%s, "
                "content_budget=%s)",
                chunk_key,
                len(segments),
                input_tokens,
                requested_tokens,
                extract_context_limit,
                content_token_budget,
            )
        else:
            segments = [content]

        async def _call_extraction_llm(
            user_prompt: str,
            system_prompt: str,
            prompts: dict[str, Any],
            *,
            history_messages: list[dict[str, str]] | None = None,
            response_format: Any = None,
        ) -> tuple[str, int]:
            fits, call_input_tokens, call_requested_tokens = _request_fits_context(
                system_prompt, user_prompt, history_messages
            )
            if not fits:
                raise RuntimeError(
                    "Entity extraction prompt exceeds LLM context before request: "
                    f"chunk_id={chunk_key}, input_tokens={call_input_tokens}, "
                    f"reserved_output={extract_reserved_output_tokens}, "
                    f"safety_margin={extract_context_safety_margin}, "
                    f"requested={call_requested_tokens}, "
                    f"context_limit={extract_context_limit}"
                )

            if response_format is None:
                return await use_llm_func_with_cache(
                    user_prompt,
                    use_llm_func,
                    system_prompt=system_prompt,
                    llm_response_cache=llm_response_cache,
                    history_messages=history_messages,
                    cache_type="extract",
                    chunk_id=chunk_key,
                    cache_keys_collector=cache_keys_collector,
                )
            try:
                return await use_llm_func_with_cache(
                    user_prompt,
                    use_llm_func,
                    system_prompt=system_prompt,
                    llm_response_cache=llm_response_cache,
                    history_messages=history_messages,
                    cache_type="extract",
                    chunk_id=chunk_key,
                    cache_keys_collector=cache_keys_collector,
                    response_format=response_format,
                )
            except Exception as exc:
                if not should_retry_without_response_format(exc):
                    raise
                logger.warning(
                    "%s: structured extraction unsupported, falling back to "
                    "delimiter prompts: %s",
                    chunk_key,
                    exc,
                )
                fallback_user_prompt = (
                    prompts["delimiter_continue_prompt"]
                    if history_messages
                    else prompts["delimiter_user_prompt"]
                )
                fits, fallback_input_tokens, fallback_requested_tokens = (
                    _request_fits_context(
                        prompts["delimiter_system_prompt"],
                        fallback_user_prompt,
                        history_messages,
                    )
                )
                if not fits:
                    raise RuntimeError(
                        "Entity extraction fallback prompt exceeds LLM context "
                        f"before request: chunk_id={chunk_key}, "
                        f"input_tokens={fallback_input_tokens}, "
                        f"reserved_output={extract_reserved_output_tokens}, "
                        f"safety_margin={extract_context_safety_margin}, "
                        f"requested={fallback_requested_tokens}, "
                        f"context_limit={extract_context_limit}"
                    ) from exc
                return await use_llm_func_with_cache(
                    fallback_user_prompt,
                    use_llm_func,
                    system_prompt=prompts["delimiter_system_prompt"],
                    llm_response_cache=llm_response_cache,
                    history_messages=history_messages,
                    cache_type="extract",
                    chunk_id=chunk_key,
                    cache_keys_collector=cache_keys_collector,
                )

        async def _extract_segment(segment_content: str) -> tuple[dict, dict]:
            prompts = _build_segment_prompts(segment_content)
            response_format = (
                structured_response_format
                if prompts["use_structured_output"]
                else None
            )
            final_result, timestamp = await _call_extraction_llm(
                prompts["user_prompt"],
                prompts["system_prompt"],
                prompts,
                response_format=response_format,
            )
            history = pack_user_ass_to_openai_messages(
                prompts["user_prompt"], final_result
            )

            (
                maybe_nodes,
                maybe_edges,
                initial_parse_stats,
            ) = await _process_extraction_result(
                final_result,
                chunk_key,
                timestamp,
                file_path,
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
                force_directed_relations=force_directed_relations,
                return_stats=True,
            )

            should_run_gleaning = entity_extract_max_gleaning > 0
            if should_run_gleaning and conditional_gleaning:
                should_run_gleaning = (
                    not maybe_nodes
                    and not maybe_edges
                    or initial_parse_stats["dropped_records"] > 0
                    or initial_parse_stats["parse_errors"] > 0
                )

            if should_run_gleaning:
                fits, glean_input_tokens, glean_requested_tokens = (
                    _request_fits_context(
                        prompts["system_prompt"],
                        prompts["continue_prompt"],
                        history,
                    )
                )
                if not fits:
                    logger.warning(
                        "Gleaning stopped for chunk %s: input_tokens=%s, "
                        "requested=%s, context_limit=%s",
                        chunk_key,
                        glean_input_tokens,
                        glean_requested_tokens,
                        extract_context_limit,
                    )
                else:
                    glean_result, timestamp = await _call_extraction_llm(
                        prompts["continue_prompt"],
                        prompts["system_prompt"],
                        prompts,
                        history_messages=history,
                        response_format=response_format,
                    )
                    (
                        glean_nodes,
                        glean_edges,
                        _glean_parse_stats,
                    ) = await _process_extraction_result(
                        glean_result,
                        chunk_key,
                        timestamp,
                        file_path,
                        tuple_delimiter=context_base["tuple_delimiter"],
                        completion_delimiter=context_base["completion_delimiter"],
                        force_directed_relations=force_directed_relations,
                        return_stats=True,
                    )

                    for entity_name, glean_entities in glean_nodes.items():
                        if entity_name in maybe_nodes:
                            original_desc_len = len(
                                maybe_nodes[entity_name][0].get("description", "")
                                or ""
                            )
                            glean_desc_len = len(
                                glean_entities[0].get("description", "") or ""
                            )
                            if glean_desc_len > original_desc_len:
                                maybe_nodes[entity_name] = list(glean_entities)
                        else:
                            maybe_nodes[entity_name] = list(glean_entities)

                    for edge_key, glean_edge_list in glean_edges.items():
                        if edge_key in maybe_edges:
                            original_desc_len = len(
                                maybe_edges[edge_key][0].get("description", "") or ""
                            )
                            glean_desc_len = len(
                                glean_edge_list[0].get("description", "") or ""
                            )
                            if glean_desc_len > original_desc_len:
                                maybe_edges[edge_key] = list(glean_edge_list)
                        else:
                            maybe_edges[edge_key] = list(glean_edge_list)

            return maybe_nodes, maybe_edges

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for segment_index, segment_content in enumerate(segments, start=1):
            if len(segments) > 1:
                logger.info(
                    "%s: extracting split segment %s/%s",
                    chunk_key,
                    segment_index,
                    len(segments),
                )
            segment_nodes, segment_edges = await _extract_segment(segment_content)
            _merge_extraction_maps(maybe_nodes, segment_nodes)
            _merge_extraction_maps(maybe_edges, segment_edges)

        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        maybe_nodes = dict(maybe_nodes)
        maybe_edges = dict(maybe_edges)
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = (
            f"Chunk {processed_chunks} of {total_chunks} extracted "
            f"{entities_count} Ent + {relations_count} Rel {chunk_key}"
        )
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        nonlocal failed_chunks, processed_chunks
        async with semaphore:
            # Check for cancellation before processing chunk
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during chunk processing"
                        )

            try:
                return await _process_single_content(chunk)
            except Exception as e:
                chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
                if skip_failed_chunks:
                    failed_chunks += 1
                    processed_chunks += 1
                    error_message = (
                        f"{chunk_id}: entity extraction skipped after chunk error: {e}"
                    )
                    failed_chunk_errors.append(error_message)
                    logger.exception(error_message)
                    if (
                        pipeline_status is not None
                        and pipeline_status_lock is not None
                    ):
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = error_message
                            pipeline_status["history_messages"].append(
                                error_message
                            )
                    return {}, {}
                prefixed_exception = create_prefixed_exception(e, chunk_id)
                raise prefixed_exception from e

    def _group_chunks_by_token_budget(
        chunks_list: list, max_tokens: int
    ) -> list[list]:
        sub_batches = []
        current_batch = []
        current_tokens = 0
        for chunk in chunks_list:
            tokens = chunk[1].get("tokens", 0)
            if current_tokens + tokens > max_tokens and current_batch:
                sub_batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(chunk)
            current_tokens += tokens
        if current_batch:
            sub_batches.append(current_batch)
        return sub_batches

    TOKEN_BUDGET = int(
        os.getenv(
            "LLM_MODEL_MAX_TOKEN_BUDGET",
            global_config.get("llm_model_max_token_budget", 150000),
        )
    )
    sub_batches = _group_chunks_by_token_budget(ordered_chunks, max_tokens=TOKEN_BUDGET)

    chunk_results = []

    for sub_batch_idx, sub_batch in enumerate(sub_batches):
        log_message = (
            f"Processing sub-batch {sub_batch_idx + 1}/{len(sub_batches)} "
            f"({len(sub_batch)} chunks)"
        )
        logger.info(log_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        tasks = []
        for c in sub_batch:
            task = asyncio.create_task(_process_with_semaphore(c))
            tasks.append(task)

        # Wait for tasks in current sub-batch to complete or for the first exception to occur
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        first_exception = None

        for task in done:
            try:
                exception = task.exception()
                if exception is not None:
                    if first_exception is None:
                        first_exception = exception
                else:
                    chunk_results.append(task.result())
            except Exception as e:
                if first_exception is None:
                    first_exception = e

        # If any task failed, cancel all pending tasks and raise the first exception
        if first_exception is not None:
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                await asyncio.wait(pending)

            progress_prefix = f"C[{processed_chunks + 1}/{total_chunks}]"
            prefixed_exception = create_prefixed_exception(
                first_exception, progress_prefix
            )
            raise prefixed_exception from first_exception

    should_fail, max_allowed_failed_chunks, failure_message = (
        _skipped_entity_chunk_failure(
            failed_chunks,
            total_chunks,
            max_failed_chunk_ratio,
            fail_on_all_chunks_failed=fail_on_all_chunks_failed,
            fail_on_too_many_skipped_chunks=fail_on_too_many_skipped_chunks,
            first_errors=failed_chunk_errors,
        )
    )
    if failed_chunks:
        logger.warning(
            "Entity extraction skipped %s/%s chunks (max allowed %s). First errors: %s",
            failed_chunks,
            total_chunks,
            max_allowed_failed_chunks,
            failed_chunk_errors[:3],
        )
        if should_fail:
            raise RuntimeError(failure_message)

    # If all tasks completed successfully, chunk_results already contains the results
    return chunk_results


async def extract_legal_entities(
    chunks: dict[str, Any],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: Any = None,
    text_chunks_storage: Any = None,
) -> list:
    """Extract entities and relationships from legal document chunks.

    This function implements the specific hierarchy for legal documents:
    Level 0: tvpl_key
    Level 1: Chương
    Level 2: Điều
    Level 3: Khoản
    Level 4: Điểm

    And attaches related documents as leaves.
    It also calls extract_entities to let LLM extract domain entities.
    """
    return await run_legal_entity_extraction(
        chunks,
        global_config,
        pipeline_status,
        pipeline_status_lock,
        llm_response_cache,
        text_chunks_storage,
        extract_entities,
        _process_extraction_result,
    )


async def correct_query_spelling(
    query: str,
    llm_func: callable,
    language: str = "english",
) -> str:
    """
    Correct spelling errors in user query before processing.

    Args:
        query: Original user query
        llm_func: LLM function for spell correction
        language: Query language (english/vietnamese)

    Returns:
        str: Corrected query
    """
    if not query or len(query.strip()) < 3:
        return query

    # Select appropriate prompt based on language
    prompt_key = (
        "query_spell_correction"
        if language.lower() == "english"
        else "query_spell_correction_vi"
    )
    examples_key = (
        "query_spell_correction_examples"
        if language.lower() == "english"
        else "query_spell_correction_examples_vi"
    )

    prompt = PROMPTS[prompt_key].format(
        examples="\n".join(PROMPTS[examples_key]), query=query
    )

    try:
        corrected = await llm_func(
            prompt,
            system_prompt="You are a spell checker. Return only the corrected query.",
            max_tokens=len(query.split()) * 3,  # Allow some expansion
            temperature=0.1,  # Low temperature for consistency
        )

        # Validate correction
        if corrected and len(corrected.strip()) > 0:
            if corrected.strip() != query.strip():
                logger.info(
                    f"[Spell Correction] Original: '{query}' → Corrected: '{corrected.strip()}'"
                )
            return corrected.strip()
        else:
            logger.warning("[Spell Correction] Empty result, using original query")
            return query

    except Exception as e:
        logger.error(f"[Spell Correction] Failed: {e}, using original query")
        return query


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: BaseVectorStorage = None,
) -> QueryResult | None:
    """
    Execute knowledge graph query and return unified QueryResult object.

    Args:
        query: Query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks storage
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt
        chunks_vdb: Document chunks vector database

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Based on different query_param settings, different fields will be populated:
        - only_need_context=True: content contains context string
        - only_need_prompt=True: content contains complete prompt
        - stream=True: response_iterator contains streaming response, raw_data contains complete data
        - default: content contains LLM response text, raw_data contains complete data

        Returns None when no relevant context could be constructed for the query.
    """
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    # Spell correction before keyword extraction
    if query_param.enable_spell_correction:
        try:
            from lightrag.language_detector import detect_language

            detected_lang = detect_language(query)
            query = await correct_query_spelling(
                query, llm_func=use_model_func, language=detected_lang
            )
        except Exception as e:
            logger.warning(
                f"[Spell Correction] Failed to correct query: {e}, using original"
            )

    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            logger.warning(f"Forced low_level_keywords to origin query: {query}")
            ll_keywords = [query]
        else:
            return QueryResult(content=PROMPTS["fail_response"])

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Create citation tracker if citations are enabled
    citation_tracker = None
    if query_param.enable_citations:
        from lightrag.citation import CitationTracker

        citation_tracker = CitationTracker()

    # Build query context (unified interface)
    context_result = await _build_query_context(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
        citation_tracker=citation_tracker,
    )

    if context_result is None:
        logger.info("[kg_query] No query context could be built; returning no-result.")
        return None

    # Return different content based on query parameters
    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(
            content=context_result.context, raw_data=context_result.raw_data
        )

    language_lock = _build_language_lock_instruction(query)
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else ""
    user_prompt = f"{user_prompt}{language_lock}" if user_prompt else language_lock
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Initialize PromptManager for language-aware prompt retrieval
    prompt_manager = PromptManager(PROMPTS)

    # Auto-detect language from query or use system prompt
    if system_prompt:
        sys_prompt_temp = system_prompt
    else:
        sys_prompt_temp = prompt_manager.get_prompt(
            "rag_response", auto_detect_text=query
        )

    # Add citation instructions if citations are enabled
    if query_param.enable_citations and citation_tracker:
        from lightrag.citation import append_citation_instruction_to_prompt
        from lightrag.language_detector import detect_language

        # Detect language for citation instructions
        detected_lang = detect_language(query)
        language = "Vietnamese" if detected_lang == "vi" else "English"

        # Append citation instructions to system prompt
        sys_prompt_temp = append_citation_instruction_to_prompt(
            sys_prompt_temp,
            enable_citations=True,
            doc_type=None,  # Support both legal and general documents
            language=language,
        )

    sys_prompt = sys_prompt_temp.format(
        response_type=response_type,
        user_prompt=user_prompt,
        context_data=context_result.context,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=context_result.raw_data)

    # Call LLM
    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

    # Handle cache
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        hl_keywords_str,
        ll_keywords_str,
        query_param.user_prompt or "",
        query_param.enable_rerank,
        "lang_lock_v1",
    )

    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )

    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
            cache_type="query",
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "hl_keywords": hl_keywords_str,
                "ll_keywords": ll_keywords_str,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        # Render citations if enabled
        if query_param.enable_citations and citation_tracker:
            from lightrag.citation import CitationRenderer
            from lightrag.language_detector import detect_language
            from lightrag.citation_metrics import get_global_metrics_tracker

            # Detect language for citation rendering
            detected_lang = detect_language(query)
            language = "Vietnamese" if detected_lang == "vi" else "English"

            # Get global metrics tracker
            metrics_tracker = get_global_metrics_tracker()

            # Create renderer and render citations
            renderer = CitationRenderer(
                citation_tracker=citation_tracker,
                format=query_param.citation_format,
                language=language,
                citation_order=query_param.citation_order,
                metrics_tracker=metrics_tracker,
            )

            # Render citations in the response
            rendered_response, citation_list = renderer.render(response)

            # Track query-level usage metrics
            metrics_tracker.track_query(
                enable_citations=True,
                citation_count=citation_tracker.get_citation_count(),
                citation_format=query_param.citation_format,
                workspaces=citation_list.workspaces_used,
            )

            return QueryResult(
                content=rendered_response,
                raw_data=context_result.raw_data,
                citations=citation_list,
                citation_count=citation_tracker.get_citation_count(),
            )
        else:
            # Track query without citations
            from lightrag.citation_metrics import get_global_metrics_tracker

            metrics_tracker = get_global_metrics_tracker()
            metrics_tracker.track_query(
                enable_citations=False,
                citation_count=0,
            )

        return QueryResult(content=response, raw_data=context_result.raw_data)
    else:
        # Streaming response (AsyncIterator)
        # Note: Citation rendering for streaming responses is not yet implemented
        # The response will be streamed without citation formatting
        return QueryResult(
            response_iterator=response,
            raw_data=context_result.raw_data,
            is_streaming=True,
        )


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords using extract_keywords_only function with conversation history support
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv, query_param.conversation_history
    )
    return hl_keywords, ll_keywords


async def analyze_query(
    query: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> dict[str, Any]:
    """
    Consolidate normalization, classification, and entity extraction into a single LLM call.
    Returns a JSON object with:
    - normalized_query: Corrected query
    - intent: 'BYPASS', 'GLOBAL', or 'SEARCH'
    - high_level_keywords: List of broad concepts
    - low_level_keywords: List of specific entities
    """
    # Initialize PromptManager for language-aware prompt retrieval
    prompt_manager = PromptManager(PROMPTS)

    # Build history string
    history_str = ""
    if param.conversation_history:
        # Use last 5 turns for context to avoid token bloat
        history_str = "\n".join(
            [
                f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
                for m in param.conversation_history[-5:]
            ]
        )

    # Build the analysis prompt using PromptManager
    analysis_prompt = prompt_manager.get_prompt(
        "query_analysis", auto_detect_text=query
    ).format(query=query, history=history_str)

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(analysis_prompt))
    logger.info(
        f"[analyze_query] Analyzing query: '{query[:50]}...' ({len_of_prompts:,} tokens)"
    )

    # Call the LLM for unified analysis
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query analysis
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(
        analysis_prompt,
        system_prompt="You are a query analyzer. Output ONLY a valid JSON object.",
        history_messages=param.conversation_history,
    )

    # Parse JSON result
    result = remove_think_tags(result)
    try:
        analysis_data = json_repair.loads(result)
        if not analysis_data:
            logger.error("No JSON-like structure found in the LLM response.")
            raise ValueError("Empty LLM response")
    except Exception as e:
        logger.error(f"JSON parsing error in analyze_query: {e}")
        logger.error(f"LLM response: {result}")
        # Return safe fallback
        return {
            "normalized_query": query,
            "intent": "SEARCH",
            "high_level_keywords": [],
            "low_level_keywords": [],
        }

    return analysis_data


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # Initialize PromptManager for language-aware prompt retrieval
    prompt_manager = PromptManager(PROMPTS)

    # 1. Build the examples - auto-detect language from query text
    examples_prompt = prompt_manager.get_prompt(
        "keywords_extraction_examples", auto_detect_text=text
    )
    examples = (
        "\n".join(examples_prompt)
        if isinstance(examples_prompt, list)
        else examples_prompt
    )

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    # 2. Handle cache if needed - add cache type for keywords
    history_str = ""
    if conversation_history:
        # Use last 5 turns for context to avoid token bloat
        history_str = "\n".join(
            [
                f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
                for m in conversation_history[-5:]
            ]
        )

    args_hash = compute_args_hash(
        param.mode,
        text,
        language,
        history_str,
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        try:
            keywords_data = json_repair.loads(cached_response)
            return keywords_data.get("high_level_keywords", []), keywords_data.get(
                "low_level_keywords", []
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 3. Build the keyword-extraction prompt using PromptManager
    kw_prompt = prompt_manager.get_prompt(
        "keywords_extraction", auto_detect_text=text
    ).format(
        query=text,
        history=history_str,
        examples=examples,
        language=language,
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(kw_prompt))
    logger.debug(
        f"[extract_keywords] Sending to LLM: {len_of_prompts:,} tokens (Prompt: {len_of_prompts})"
    )

    # 4. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(
        kw_prompt,
        keyword_extraction=True,
        history_messages=conversation_history,
    )

    # 5. Parse out JSON from the LLM response
    result = remove_think_tags(result)
    try:
        keywords_data = json_repair.loads(result)
        if not keywords_data:
            logger.error("No JSON-like structure found in the LLM respond.")
            return [], []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"LLM respond: {result}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 6. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv.global_config.get("enable_llm_cache"):
            # Save to cache with query parameters
            queryparam_dict = {
                "mode": param.mode,
                "response_type": param.response_type,
                "top_k": param.top_k,
                "chunk_top_k": param.chunk_top_k,
                "max_entity_tokens": param.max_entity_tokens,
                "max_relation_tokens": param.max_relation_tokens,
                "max_total_tokens": param.max_total_tokens,
                "user_prompt": param.user_prompt or "",
                "enable_rerank": param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    mode=param.mode,
                    cache_type="keywords",
                    queryparam=queryparam_dict,
                ),
            )

    return hl_keywords, ll_keywords


async def _get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    This function performs vector search to find relevant text chunks for a query.
    Reranking and truncation will be handled later in the unified processing.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including chunk_top_k and ids
        query_embedding: Optional pre-computed query embedding to avoid redundant embedding calls

    Returns:
        List of text chunks with metadata
    """
    try:
        # Use chunk_top_k if specified, otherwise fall back to top_k
        search_top_k = query_param.chunk_top_k or query_param.top_k

        # Parallel query across all workspaces if other_rag_instances is specified
        other_rags = getattr(query_param, "other_rag_instances", [])
        vdbs = [chunks_vdb]
        for other_rag in other_rags:
            if hasattr(other_rag, "chunks_vdb") and other_rag.chunks_vdb:
                vdbs.append(other_rag.chunks_vdb)

        tasks = [
            vdb.query(query, top_k=search_top_k, query_embedding=query_embedding)
            for vdb in vdbs
        ]
        all_results = await asyncio.gather(*tasks)

        valid_chunks = []
        for vdb, results in zip(vdbs, all_results):
            for result in results:
                if "content" in result:
                    metadata = result.get("metadata") or {}
                    if isinstance(metadata, str):
                        try:
                            import json

                            metadata = json.loads(metadata)
                        except Exception:
                            metadata = {}
                    context = result.get("context", metadata.get("context", {}))
                    if isinstance(context, str):
                        try:
                            import json

                            context = json.loads(context)
                        except Exception:
                            context = {}

                    chunk_with_metadata = {
                        "content": result["content"],
                        "created_at": result.get("created_at", None),
                        "file_path": result.get(
                            "file_path", metadata.get("file_path", "unknown_source")
                        ),
                        "source_type": "vector",
                        "chunk_id": result.get("id"),
                        "workspace": getattr(
                            vdb, "workspace", ""
                        ),  # Associate with workspace!
                        "score": result.get("distance") or result.get("score") or 0.0,
                    }

                    # Carry over relevant metadata fields
                    chunk_with_metadata["metadata"] = {
                        "doc_type": result.get(
                            "doc_type", metadata.get("doc_type", "general")
                        ),
                        "file_path": result.get(
                            "file_path", metadata.get("file_path", "")
                        ),
                        "context": context,
                        **{
                            k: v
                            for k, v in metadata.items()
                            if k not in ["doc_type", "file_path", "context"]
                        },
                    }
                    valid_chunks.append(chunk_with_metadata)

        # Merge, sort by relevance score, and slice
        valid_chunks = sorted(
            valid_chunks, key=lambda x: x.get("score", 0.0), reverse=True
        )[:search_top_k]

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks retrieved across {len(vdbs)} workspaces"
        )
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return []


_LEGAL_STATUS_CONTEXT_KEYS = (
    "legal_status",
    "legal_status_code",
    "legal_status_priority",
)


def _as_metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _legal_status_records_from_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    if not isinstance(item, dict):
        return records

    records.append(item)

    item_legal_info = _as_metadata_dict(item.get("legal_info"))
    if item_legal_info:
        records.append(item_legal_info)

    metadata = _as_metadata_dict(item.get("metadata"))
    if metadata:
        records.append(metadata)
        metadata_legal_info = _as_metadata_dict(metadata.get("legal_info"))
        if metadata_legal_info:
            records.append(metadata_legal_info)

    return records


def _legal_status_metadata_from_item(item: dict[str, Any]) -> dict[str, Any]:
    records = _legal_status_records_from_item(item)
    status_metadata = best_legal_status_metadata(records)
    if status_metadata:
        return status_metadata

    direct_metadata = {}
    for record in records:
        for key in _LEGAL_STATUS_CONTEXT_KEYS:
            value = record.get(key)
            if value not in (None, "") and key not in direct_metadata:
                direct_metadata[key] = value
    return direct_metadata


def _legal_status_priority_from_item(item: dict[str, Any]) -> int:
    status_metadata = _legal_status_metadata_from_item(item)
    if not status_metadata:
        return 0
    try:
        return legal_status_sort_priority(status_metadata)
    except Exception:
        priority = status_metadata.get("legal_status_priority", 0)
        try:
            return int(priority)
        except (TypeError, ValueError):
            return 0


def _prioritize_items_by_legal_status(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for _, item in sorted(
            enumerate(items),
            key=lambda indexed_item: (
                -_legal_status_priority_from_item(indexed_item[1]),
                indexed_item[0],
            ),
        )
    ]


def _add_legal_status_context_fields(
    context_item: dict[str, Any], source_item: dict[str, Any]
) -> None:
    status_metadata = _legal_status_metadata_from_item(source_item)
    for key in _LEGAL_STATUS_CONTEXT_KEYS:
        value = status_metadata.get(key)
        if value not in (None, ""):
            context_item[key] = value


async def _perform_kg_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
) -> dict[str, Any]:
    """
    Pure search logic that retrieves raw entities, relations, and vector chunks.
    No token truncation or formatting - just raw search results.
    """

    # Initialize result containers
    local_entities = []
    local_relations = []
    global_entities = []
    global_relations = []
    vector_chunks = []
    chunk_tracking = {}

    # Handle different query modes

    # Track chunk sources and metadata for final logging
    chunk_tracking = {}  # chunk_id -> {source, frequency, order}

    # Pre-compute embeddings needed by the selected mode in a single batch call.
    # Only embed texts that the active retrieval branches will actually use:
    #   - query        → used by _get_vector_context (chunks VDB)
    #   - ll_keywords  → used by _get_node_data (entities VDB) in local/hybrid/mix
    #   - hl_keywords  → used by _get_edge_data (relationships VDB) in global/hybrid/mix
    # Batching avoids 2-3 sequential API round-trips.
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )

    actual_embedding_func = text_chunks_db.embedding_func
    query_embedding = None
    ll_embedding = None
    hl_embedding = None

    mode = query_param.mode
    need_ll = mode in ("local", "hybrid", "mix") and bool(ll_keywords)
    need_hl = mode in ("global", "hybrid", "mix") and bool(hl_keywords)

    if actual_embedding_func:
        texts_to_embed: list[str] = []
        text_purposes: list[str] = []

        if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
            texts_to_embed.append(query)
            text_purposes.append("query")

        if need_ll:
            texts_to_embed.append(ll_keywords)
            text_purposes.append("ll")

        if need_hl:
            texts_to_embed.append(hl_keywords)
            text_purposes.append("hl")

        if texts_to_embed:
            try:
                all_embeddings = await actual_embedding_func(
                    texts_to_embed, _priority=5
                )
                for i, purpose in enumerate(text_purposes):
                    if purpose == "query":
                        query_embedding = all_embeddings[i]
                    elif purpose == "ll":
                        ll_embedding = all_embeddings[i]
                    elif purpose == "hl":
                        hl_embedding = all_embeddings[i]
                logger.debug(
                    "Pre-computed %d embeddings in single batch (purposes: %s)",
                    len(texts_to_embed),
                    ", ".join(text_purposes),
                )
            except Exception as e:
                logger.warning(f"Failed to batch pre-compute embeddings: {e}")

    # Handle local and global modes
    if query_param.mode == "local" and len(ll_keywords) > 0:
        local_entities, local_relations = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
            query_embedding=ll_embedding,
        )

    elif query_param.mode == "global" and len(hl_keywords) > 0:
        global_relations, global_entities = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
            query_embedding=hl_embedding,
        )

    else:  # hybrid or mix mode
        if len(ll_keywords) > 0:
            local_entities, local_relations = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                query_param,
                query_embedding=ll_embedding,
            )
        if len(hl_keywords) > 0:
            global_relations, global_entities = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                query_param,
                query_embedding=hl_embedding,
            )

        # Get vector chunks for mix mode
        if query_param.mode == "mix" and chunks_vdb:
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
                query_embedding,
            )
            # Track vector chunks with source metadata
            for i, chunk in enumerate(vector_chunks):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id:
                    chunk_tracking[chunk_id] = {
                        "source": "C",
                        "frequency": 1,  # Vector chunks always have frequency 1
                        "order": i + 1,  # 1-based order in vector search results
                    }
                else:
                    logger.warning(f"Vector chunk missing chunk_id: {chunk}")

    # Round-robin merge entities
    final_entities = []
    seen_entities = set()
    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        # First from local
        if i < len(local_entities):
            entity = local_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

        # Then from global
        if i < len(global_entities):
            entity = global_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

    # Round-robin merge relations
    final_relations = []
    seen_relations = set()
    max_len = max(len(local_relations), len(global_relations))

    def relation_dedupe_key(relation: dict) -> tuple:
        if "src_tgt" in relation:
            src_id, tgt_id = relation["src_tgt"]
        else:
            src_id, tgt_id = relation.get("src_id"), relation.get("tgt_id")
        if relation.get("directed"):
            return src_id, tgt_id
        return tuple(sorted([src_id, tgt_id]))

    for i in range(max_len):
        # First from local
        if i < len(local_relations):
            relation = local_relations[i]
            rel_key = relation_dedupe_key(relation)

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

        # Then from global
        if i < len(global_relations):
            relation = global_relations[i]
            rel_key = relation_dedupe_key(relation)

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    final_entities = _prioritize_items_by_legal_status(final_entities)
    final_relations = _prioritize_items_by_legal_status(final_relations)

    logger.info(
        f"Raw search results: {len(final_entities)} entities, {len(final_relations)} relations, {len(vector_chunks)} vector chunks"
    )

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


async def _apply_token_truncation(
    search_result: dict[str, Any],
    query_param: QueryParam,
    global_config: dict[str, str],
) -> dict[str, Any]:
    """
    Apply token-based truncation to entities and relations for LLM efficiency.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.warning("No tokenizer found, skipping truncation")
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": search_result["final_entities"],
            "filtered_relations": search_result["final_relations"],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }

    # Get token limits from query_param with fallbacks
    max_entity_tokens = getattr(
        query_param,
        "max_entity_tokens",
        global_config.get("max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS),
    )
    max_relation_tokens = getattr(
        query_param,
        "max_relation_tokens",
        global_config.get("max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS),
    )

    final_entities = search_result["final_entities"]
    final_relations = search_result["final_relations"]

    # Create mappings from entity/relation identifiers to original data
    entity_id_to_original = {}
    relation_id_to_original = {}

    # Generate entities context for truncation
    entities_context = []
    for i, entity in enumerate(final_entities):
        entity_name = entity["entity_name"]
        created_at = entity.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Store mapping from entity name to original data
        entity_id_to_original[entity_name] = entity

        entity_context = {
            "entity": entity_name,
            "type": entity.get("entity_type", "UNKNOWN"),
            "description": entity.get("description", "UNKNOWN"),
            "created_at": created_at,
            "file_path": entity.get("file_path", "unknown_source"),
        }
        _add_legal_status_context_fields(entity_context, entity)
        entities_context.append(entity_context)

    # Generate relations context for truncation
    relations_context = []
    for i, relation in enumerate(final_relations):
        created_at = relation.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Handle different relation data formats
        if "src_tgt" in relation:
            entity1, entity2 = relation["src_tgt"]
        else:
            entity1, entity2 = relation.get("src_id"), relation.get("tgt_id")

        # Store mapping from relation pair to original data
        relation_key = (entity1, entity2)
        relation_id_to_original[relation_key] = relation

        relation_context = {
            "entity1": entity1,
            "entity2": entity2,
            "description": relation.get("description", "UNKNOWN"),
            "created_at": created_at,
            "file_path": relation.get("file_path", "unknown_source"),
        }
        _add_legal_status_context_fields(relation_context, relation)
        relations_context.append(relation_context)

    logger.debug(
        f"Before truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Apply token-based truncation
    if entities_context:
        # Remove file_path and created_at for token calculation
        entities_context_for_truncation = []
        for entity in entities_context:
            entity_copy = entity.copy()
            entity_copy.pop("file_path", None)
            entity_copy.pop("created_at", None)
            entities_context_for_truncation.append(entity_copy)

        entities_context = truncate_list_by_token_size(
            entities_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

    if relations_context:
        # Remove file_path and created_at for token calculation
        relations_context_for_truncation = []
        for relation in relations_context:
            relation_copy = relation.copy()
            relation_copy.pop("file_path", None)
            relation_copy.pop("created_at", None)
            relations_context_for_truncation.append(relation_copy)

        relations_context = truncate_list_by_token_size(
            relations_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

    logger.info(
        f"After truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Create filtered original data based on truncated context
    filtered_entities = []
    filtered_entity_id_to_original = {}
    if entities_context:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for entity in final_entities:
            name = entity.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                filtered_entities.append(entity)
                filtered_entity_id_to_original[name] = entity
                seen_nodes.add(name)

    filtered_relations = []
    filtered_relation_id_to_original = {}
    if relations_context:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        for relation in final_relations:
            src, tgt = relation.get("src_id"), relation.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = relation.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                filtered_relations.append(relation)
                filtered_relation_id_to_original[pair] = relation
                seen_edges.add(pair)

    return {
        "entities_context": entities_context,
        "relations_context": relations_context,
        "filtered_entities": filtered_entities,
        "filtered_relations": filtered_relations,
        "entity_id_to_original": filtered_entity_id_to_original,
        "relation_id_to_original": filtered_relation_id_to_original,
    }


async def _merge_all_chunks(
    filtered_entities: list[dict],
    filtered_relations: list[dict],
    vector_chunks: list[dict],
    query: str = "",
    knowledge_graph_inst: BaseGraphStorage = None,
    text_chunks_db: BaseKVStorage = None,
    query_param: QueryParam = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding: list[float] = None,
    citation_tracker: "CitationTracker" = None,
) -> list[dict]:
    """
    Merge chunks from different sources: vector_chunks + entity_chunks + relation_chunks.

    Args:
        filtered_entities: List of filtered entity dictionaries
        filtered_relations: List of filtered relation dictionaries
        vector_chunks: List of chunks from vector search
        query: User query string
        knowledge_graph_inst: Knowledge graph storage instance
        text_chunks_db: Text chunks key-value storage
        query_param: Query parameters
        chunks_vdb: Chunks vector database
        chunk_tracking: Dictionary for tracking chunk sources
        query_embedding: Query embedding vector
        citation_tracker: Citation tracker for tracking retrieved sources (optional)

    Returns:
        List of merged and deduplicated chunks
    """
    if chunk_tracking is None:
        chunk_tracking = {}

    # Get chunks from entities
    entity_chunks = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await _find_related_text_unit_from_entities(
            filtered_entities,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
            citation_tracker=citation_tracker,
        )

    # Get chunks from relations
    relation_chunks = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await _find_related_text_unit_from_relations(
            filtered_relations,
            query_param,
            text_chunks_db,
            entity_chunks,  # For deduplication
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
            citation_tracker=citation_tracker,
        )

    # Round-robin merge chunks from different sources with deduplication
    merged_chunks = []
    seen_chunk_ids = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)

    for i in range(max_len):
        # Add from vector chunks first (Naive mode)
        if i < len(vector_chunks):
            chunk = vector_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "workspace": chunk.get("workspace", ""),
                    }
                )

        # Add from entity chunks (Local mode)
        if i < len(entity_chunks):
            chunk = entity_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "workspace": chunk.get("workspace", ""),
                    }
                )

        # Add from relation chunks (Global mode)
        if i < len(relation_chunks):
            chunk = relation_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "workspace": chunk.get("workspace", ""),
                    }
                )

    logger.info(
        f"Round-robin merged chunks: {origin_len} -> {len(merged_chunks)} (deduplicated {origin_len - len(merged_chunks)})"
    )

    return merged_chunks


async def _build_context_str(
    entities_context: list[dict],
    relations_context: list[dict],
    merged_chunks: list[dict],
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    chunk_tracking: dict = None,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
    citation_tracker: "CitationTracker" = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the final LLM context string with token processing.
    This includes dynamic token calculation and final chunk truncation.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.error("Missing tokenizer, cannot build LLM context")
        # Return empty raw data structure when no tokenizer
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Missing tokenizer, cannot build LLM context."
        return "", empty_raw_data

    # Get token limits
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Initialize PromptManager for language-aware prompt retrieval
    prompt_manager = PromptManager(PROMPTS)

    # Get the system prompt template from global_config or use PromptManager with auto-detection
    if "system_prompt_template" in global_config:
        sys_prompt_template = global_config["system_prompt_template"]
    else:
        sys_prompt_template = prompt_manager.get_prompt(
            "rag_response", auto_detect_text=query
        )

    # Get kg_query_context template with auto-detection from query
    kg_context_template = prompt_manager.get_prompt(
        "kg_query_context", auto_detect_text=query
    )

    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    entities_str = "\n".join(
        json.dumps(entity, ensure_ascii=False) for entity in entities_context
    )
    relations_str = "\n".join(
        json.dumps(relation, ensure_ascii=False) for relation in relations_context
    )

    # Calculate preliminary kg context tokens
    pre_kg_context = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str="",
        reference_list_str="",
    )
    kg_context_tokens = len(tokenizer.encode(pre_kg_context))

    # Calculate preliminary system prompt tokens
    pre_sys_prompt = sys_prompt_template.format(
        context_data="",  # Empty for overhead calculation
        response_type=response_type,
        user_prompt=user_prompt,
    )
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))

    # Calculate available tokens for text chunks
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
    )

    logger.debug(
        f"Token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Apply token truncation to chunks using the dynamic limit
    truncated_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=merged_chunks,
        query_param=query_param,
        global_config=global_config,
        source_type=query_param.mode,
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    # Generate reference list from truncated chunks using the new common function
    reference_list, truncated_chunks = generate_reference_list_from_chunks(
        truncated_chunks
    )

    # Synchronize reference_id to CitationTracker
    if citation_tracker:
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id", "")
            doc_id = chunk_id.split(":", 1)[0] if ":" in chunk_id else ""
            cit = citation_tracker.get_citation_by_chunk_id(doc_id, chunk_id)
            if cit:
                cit.reference_id = chunk["reference_id"]

    # Rebuild chunks_context with truncated chunks
    # The actual tokens may be slightly less than available_chunk_tokens due to deduplication logic
    chunks_context = []
    for i, chunk in enumerate(truncated_chunks):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(chunks_context)} chunks"
    )

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context and not chunks_context:
        # Return empty raw data structure when no entities/relations
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        return "", empty_raw_data

    # output chunks tracking infomations
    # format: <source><frequency>/<order> (e.g., E5/2 R2/1 C1/1)
    if truncated_chunks and chunk_tracking:
        chunk_tracking_log = []
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_tracking:
                tracking_info = chunk_tracking[chunk_id]
                source = tracking_info["source"]
                frequency = tracking_info["frequency"]
                order = tracking_info["order"]
                chunk_tracking_log.append(f"{source}{frequency}/{order}")
            else:
                chunk_tracking_log.append("?0/0")

        if chunk_tracking_log:
            logger.info(f"Final chunks S+F/O: {' '.join(chunk_tracking_log)}")

    result = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    # Always return both context and complete data structure (unified approach)
    logger.debug(
        f"[_build_context_str] Converting to user format: {len(entities_context)} entities, {len(relations_context)} relations, {len(truncated_chunks)} chunks"
    )
    final_data = convert_to_user_format(
        entities_context,
        relations_context,
        truncated_chunks,
        reference_list,
        query_param.mode,
        entity_id_to_original,
        relation_id_to_original,
    )
    logger.debug(
        f"[_build_context_str] Final data after conversion: {len(final_data.get('entities', []))} entities, {len(final_data.get('relationships', []))} relationships, {len(final_data.get('chunks', []))} chunks"
    )
    return result, final_data


# Citation tracking helper functions
def _infer_legal_context_from_chunk_content(content: str) -> dict:
    """Recover legal hierarchy and HTML anchors when chunk metadata is unavailable."""

    import re

    context: dict[str, Any] = {}
    prefix_parts = re.findall(r"\[([^\]]+)\]", content[:1000])
    for part in prefix_parts:
        normalized = part.strip()
        if normalized.startswith("Chương"):
            context.setdefault("chapter", normalized)
        elif normalized.startswith("Điều"):
            context.setdefault("article", normalized)
        elif normalized.startswith("Khoản"):
            context.setdefault("clauses", normalized)
        elif normalized.startswith("Điểm"):
            context.setdefault("points", normalized)

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, "html.parser")
        anchors = []
        for block in soup.find_all(["p", "table"]):
            anchor_id = str(block.get("id") or "")
            anchor_name = str(block.get("name") or "")
            if not anchor_id and not anchor_name:
                continue

            text = block.get_text(" ", strip=True)
            tag = ""
            index = ""
            for label, pattern in (
                ("Chương", r"^Chương\s+([IVXLCDM0-9]+)"),
                ("Điều", r"^Điều\s+(\d+)"),
                ("Khoản", r"^\s*(?:Khoản\s+)?(\d+)[.:]"),
                ("Điểm", r"^\s*(?:Điểm\s+)?([a-z])[.)]"),
            ):
                match = re.match(pattern, text, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    tag = label
                    index = match.group(1)
                    break

            anchors.append(
                {
                    "tag": tag,
                    "index": index,
                    "id": anchor_id,
                    "name": anchor_name,
                    "text": text[:160],
                }
            )

        if anchors:
            context["citation_position"] = {
                "anchors": anchors,
                "html_ids": [anchor["id"] for anchor in anchors if anchor.get("id")],
                "html_names": [
                    anchor["name"] for anchor in anchors if anchor.get("name")
                ],
                "start": anchors[0],
                "end": anchors[-1],
            }
    except Exception as e:
        logger.debug(f"Failed to infer legal HTML anchors from chunk content: {e}")

    return context


def _track_chunk_citation(
    chunk: dict, citation_tracker: "CitationTracker", text_chunks_db: BaseKVStorage
) -> None:
    """Track a chunk in the citation tracker.

    Extracts metadata from chunk and tracks it with the citation tracker.
    Handles both doc_type="legal" and doc_type="general" documents.

    Args:
        chunk: Chunk dictionary with content and metadata
        citation_tracker: Citation tracker instance
        text_chunks_db: Text chunks storage for accessing global config
    """
    try:
        # Extract basic chunk information
        doc_id = chunk.get("doc_id") or chunk.get("full_doc_id", "")
        chunk_id = chunk.get("chunk_id") or chunk.get("id", "")
        workspace = chunk.get("workspace", "")
        content = chunk.get("content", "")

        # Extract metadata
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        # Determine document type (default to "general")
        doc_type = chunk.get("doc_type") or metadata.get("doc_type", "general")
        if doc_type == "general" and (
            chunk.get("legal_document_id") or "[Điều" in content or "[Chương" in content
        ):
            doc_type = "legal"

        # Extract common fields
        file_path = chunk.get("file_path", metadata.get("file_path", ""))
        relevance_score = chunk.get("score", metadata.get("relevance_score", 0.0))

        # Extract hierarchy context based on document type
        hierarchy_path = []
        legal_info = {}

        if doc_type == "legal":
            # Extract legal document hierarchy from chunk metadata
            context = metadata.get("context") or chunk.get("context", {})
            if isinstance(context, str):
                try:
                    context = json.loads(context)
                except Exception:
                    context = {}
            if not context:
                context = _infer_legal_context_from_chunk_content(content)
            chapter = context.get("chapter", "")
            article = context.get("article", "")

            if chapter:
                hierarchy_path.append(chapter)
            if article:
                hierarchy_path.append(article)
            if context.get("clauses"):
                hierarchy_path.append(context["clauses"])
            if context.get("points"):
                hierarchy_path.append(context["points"])

            status_metadata = best_legal_status_metadata([chunk, metadata, context])
            legal_document_id = chunk.get("legal_document_id") or metadata.get(
                "legal_document_id"
            )
            if legal_document_id and "legal_document_ids" not in status_metadata:
                status_metadata["legal_document_ids"] = str(legal_document_id)

            # Build legal_info dictionary
            legal_info = {
                "chapter": chapter,
                "article": article,
                "clause": context.get("clause") or context.get("clauses", ""),
                "clauses": context.get("clauses", ""),
                "points": context.get("points", ""),
                "citation_position": context.get("citation_position", {}),
                "chapter_anchor": context.get("chapter_anchor", {}),
                "article_anchor": context.get("article_anchor", {}),
                "document_type": metadata.get("document_type", ""),
                "document_number": metadata.get("document_number", ""),
                "issuing_authority": metadata.get("issuing_authority", ""),
                "effective_date": metadata.get("effective_date", ""),
                **status_metadata,
            }
        else:
            # Extract general document breadcrumb from chunk metadata
            context = metadata.get("context") or chunk.get("context", {})
            if isinstance(context, str):
                try:
                    context = json.loads(context)
                except Exception:
                    context = {}
            breadcrumb = context.get("breadcrumb", [])
            if breadcrumb:
                hierarchy_path = breadcrumb

        # Prepare chunk data for citation tracker
        chunk_data = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "workspace": workspace,
            "content": content,
            "metadata": {
                "doc_type": doc_type,
                "file_path": file_path,
                "hierarchy_path": hierarchy_path,
                "relevance_score": relevance_score,
            },
        }

        # Add legal_info if it's a legal document
        if doc_type == "legal" and legal_info:
            chunk_data["metadata"]["legal_info"] = legal_info

        # Track the chunk
        citation_tracker.track_chunk(chunk_data)

    except Exception as e:
        logger.warning(f"Failed to track chunk citation: {e}")


def _track_entity_citation(
    entity: dict, citation_tracker: "CitationTracker"
) -> int | None:
    """Track an entity in the citation tracker.

    Args:
        entity: Entity dictionary with name, description, and metadata
        citation_tracker: Citation tracker instance
    """
    try:
        entity_name = entity.get("entity_name", "")
        workspace = entity.get("workspace", "")
        description = entity.get("description", "")
        metadata = _as_metadata_dict(entity.get("metadata"))

        # Extract relevance score if available
        relevance_score = entity.get("score", metadata.get("relevance_score", 0.0))
        legal_status_metadata = _legal_status_metadata_from_item(entity)

        entity_data = {
            "entity_name": entity_name,
            "workspace": workspace,
            "description": description,
            "metadata": {
                "relevance_score": relevance_score,
                **legal_status_metadata,
            },
        }
        if legal_status_metadata:
            entity_data["metadata"]["legal_info"] = dict(legal_status_metadata)

        citation = citation_tracker.track_entity(entity_data)
        return citation.citation_id

    except Exception as e:
        logger.warning(f"Failed to track entity citation: {e}")
        return None


def _track_relation_citation(
    relation: dict, citation_tracker: "CitationTracker"
) -> int | None:
    """Track a relation in the citation tracker.

    Args:
        relation: Relation dictionary with src, tgt, description, and metadata
        citation_tracker: Citation tracker instance
    """
    try:
        # Handle different relation data formats
        if "src_tgt" in relation:
            src_id, tgt_id = relation["src_tgt"]
        else:
            src_id = relation.get("src_id", "")
            tgt_id = relation.get("tgt_id", "")

        relation_key = f"{src_id} -> {tgt_id}"
        workspace = relation.get("workspace", "")
        description = relation.get("description", "")
        metadata = _as_metadata_dict(relation.get("metadata"))

        # Extract relevance score if available
        relevance_score = relation.get("score", metadata.get("relevance_score", 0.0))
        legal_status_metadata = _legal_status_metadata_from_item(relation)

        relation_data = {
            "relation_key": relation_key,
            "workspace": workspace,
            "description": description,
            "metadata": {
                "relevance_score": relevance_score,
                **legal_status_metadata,
            },
        }
        if legal_status_metadata:
            relation_data["metadata"]["legal_info"] = dict(legal_status_metadata)

        citation = citation_tracker.track_relation(relation_data)
        return citation.citation_id

    except Exception as e:
        logger.warning(f"Failed to track relation citation: {e}")
        return None


# Now let's update the old _build_query_context to use the new architecture
async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
    citation_tracker: "CitationTracker" = None,
) -> QueryContextResult | None:
    """
    Main query context building function using the new 4-stage architecture:
    1. Search -> 2. Truncate -> 3. Merge chunks -> 4. Build LLM context

    Args:
        query: User query string
        ll_keywords: Low-level keywords
        hl_keywords: High-level keywords
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks key-value storage
        query_param: Query parameters
        chunks_vdb: Chunks vector database (optional)
        citation_tracker: Citation tracker for tracking retrieved sources (optional)

    Returns unified QueryContextResult containing both context and raw_data.
    """

    if not query:
        logger.warning("Query is empty, skipping context building")
        return None

    # Stage 1: Pure search
    search_result = await _perform_kg_search(
        query,
        ll_keywords,
        hl_keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode != "mix":
            return None
        else:
            if not search_result["chunk_tracking"]:
                return None

    # NEW: Track citations if citation_tracker is provided
    if citation_tracker:
        # Track vector chunks from naive retrieval
        if search_result.get("vector_chunks"):
            for chunk in search_result["vector_chunks"]:
                _track_chunk_citation(chunk, citation_tracker, text_chunks_db)

        # Track entities
        if search_result.get("final_entities"):
            for entity in search_result["final_entities"]:
                cit_id = _track_entity_citation(entity, citation_tracker)
                if cit_id is not None:
                    entity["citation_id"] = f"[ID: {cit_id}]"

        # Track relations
        if search_result.get("final_relations"):
            for relation in search_result["final_relations"]:
                cit_id = _track_relation_citation(relation, citation_tracker)
                if cit_id is not None:
                    relation["citation_id"] = f"[ID: {cit_id}]"

    # Stage 2: Apply token truncation for LLM efficiency
    truncation_result = await _apply_token_truncation(
        search_result,
        query_param,
        text_chunks_db.global_config,
    )

    # Stage 3: Merge chunks using filtered entities/relations
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query,
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
        citation_tracker=citation_tracker,
    )

    if (
        not merged_chunks
        and not truncation_result["entities_context"]
        and not truncation_result["relations_context"]
    ):
        return None

    # Stage 4: Build final LLM context with dynamic token processing
    # _build_context_str now always returns tuple[str, dict]
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks,
        query=query,
        query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
        citation_tracker=citation_tracker,
    )

    # Convert keywords strings to lists and add complete metadata to raw_data
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []

    # Add complete metadata to raw_data (preserve existing metadata including query_mode)
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    # Update keywords while preserving existing metadata
    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "entities_after_truncation": len(
            truncation_result.get("filtered_entities", [])
        ),
        "relations_after_truncation": len(
            truncation_result.get("filtered_relations", [])
        ),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[_build_query_context] Context length: {len(context) if context else 0}"
    )
    logger.debug(
        f"[_build_query_context] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, relationships: {len(raw_data.get('data', {}).get('relationships', []))}, chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
    )

    # CRITICAL: Check if context is too weak/empty to prevent hallucination
    # If context is very short (< 100 chars of actual content), it's likely irrelevant
    if context:
        # Remove markdown headers and whitespace to get actual content length
        content_only = (
            context.replace("# Knowledge Graph Data", "")
            .replace("## Entities", "")
            .replace("## Relationships", "")
            .replace("# Document Chunks", "")
            .strip()
        )
        if len(content_only) < 10:
            logger.warning(
                f"[_build_query_context] Context too short ({len(content_only)} chars), likely irrelevant. Returning None to prevent hallucination."
            )
            return None

    return QueryContextResult(context=context, raw_data=raw_data)


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding=None,
):
    logger.info(
        f"Query nodes: {query} (top_k:{query_param.top_k}, cosine:{entities_vdb.cosine_better_than_threshold})"
    )

    other_rags = getattr(query_param, "other_rag_instances", [])
    vdbs = [entities_vdb]
    for other_rag in other_rags:
        if hasattr(other_rag, "entities_vdb") and other_rag.entities_vdb:
            vdbs.append(other_rag.entities_vdb)

    tasks = [
        vdb.query(query, top_k=query_param.top_k, query_embedding=query_embedding)
        for vdb in vdbs
    ]
    all_results = await asyncio.gather(*tasks)

    # Resolve corresponding knowledge_graph_inst
    kgs = [knowledge_graph_inst]
    for other_rag in other_rags:
        if (
            hasattr(other_rag, "chunk_entity_relation_graph")
            and other_rag.chunk_entity_relation_graph
        ):
            kgs.append(other_rag.chunk_entity_relation_graph)

    kg_tasks = []
    workspace_mapping = []
    for kg, results in zip(kgs, all_results):
        if results:
            node_ids = [r["entity_name"] for r in results]
            kg_tasks.append(
                asyncio.gather(
                    kg.get_nodes_batch(node_ids),
                    kg.node_degrees_batch(node_ids),
                )
            )
            workspace_mapping.append((kg, results, node_ids))

    kg_results = await asyncio.gather(*kg_tasks)

    node_datas = []
    for (kg, results, node_ids), (nodes_dict, degrees_dict) in zip(
        workspace_mapping, kg_results
    ):
        node_datas_ws = [nodes_dict.get(nid) for nid in node_ids]
        node_degrees_ws = [degrees_dict.get(nid, 0) for nid in node_ids]

        for k, n, d in zip(results, node_datas_ws, node_degrees_ws):
            if n is not None:
                node_datas.append(
                    {
                        **n,
                        "entity_name": k["entity_name"],
                        "rank": d,
                        "created_at": k.get("created_at"),
                        "workspace": getattr(
                            kg, "workspace", ""
                        ),  # CRITICAL: add workspace info
                        "score": k.get("distance") or k.get("score") or 0.0,
                    }
                )

    if not node_datas:
        return [], []

    # Find related edges routing queries to respective graph databases
    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Local query: {len(node_datas)} entities, {len(use_relations)} relations"
    )

    # Entities are sorted by cosine similarity
    # Relations are sorted by rank + weight
    return node_datas, use_relations


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    other_rags = getattr(query_param, "other_rag_instances", [])
    kg_map = {getattr(knowledge_graph_inst, "workspace", ""): knowledge_graph_inst}
    for other_rag in other_rags:
        if (
            hasattr(other_rag, "chunk_entity_relation_graph")
            and other_rag.chunk_entity_relation_graph
        ):
            kg_map[getattr(other_rag, "workspace", "")] = (
                other_rag.chunk_entity_relation_graph
            )

    # Group node_datas by workspace
    grouped_nodes = {}
    for nd in node_datas:
        ws = nd.get("workspace", getattr(knowledge_graph_inst, "workspace", ""))
        grouped_nodes.setdefault(ws, []).append(nd)

    # For each group, retrieve edges concurrently
    tasks = []
    workspace_keys = []
    for ws, nds in grouped_nodes.items():
        kg = kg_map.get(ws, knowledge_graph_inst)
        node_names = [dp["entity_name"] for dp in nds]
        tasks.append(kg.get_nodes_edges_batch(node_names))
        workspace_keys.append((ws, kg))

    results = await asyncio.gather(*tasks)

    edge_tasks = []
    for (ws, kg), batch_edges_dict in zip(workspace_keys, results):
        all_edges = []
        seen = set()
        nds = grouped_nodes[ws]
        node_names = [dp["entity_name"] for dp in nds]
        for node_name in node_names:
            this_edges = batch_edges_dict.get(node_name, [])
            for e in this_edges:
                sorted_edge = tuple(sorted(e))
                if sorted_edge not in seen:
                    seen.add(sorted_edge)
                    all_edges.append(sorted_edge)

        if all_edges:
            edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
            edge_pairs_tuples = list(all_edges)
            edge_tasks.append(
                (
                    ws,
                    kg,
                    all_edges,
                    kg.get_edges_batch(edge_pairs_dicts),
                    kg.edge_degrees_batch(edge_pairs_tuples),
                )
            )

    if not edge_tasks:
        return []

    prop_tasks = [t[3] for t in edge_tasks]
    deg_tasks = [t[4] for t in edge_tasks]
    all_props, all_degs = await asyncio.gather(
        asyncio.gather(*prop_tasks), asyncio.gather(*deg_tasks)
    )

    all_edges_data = []
    for idx, (ws, kg, all_edges, _, _) in enumerate(edge_tasks):
        edge_data_dict = all_props[idx]
        edge_degrees_dict = all_degs[idx]
        for pair in all_edges:
            edge_props = edge_data_dict.get(pair)
            if edge_props is not None:
                if "weight" not in edge_props:
                    edge_props["weight"] = 1.0
                combined = {
                    "src_tgt": pair,
                    "rank": edge_degrees_dict.get(pair, 0),
                    "workspace": ws,
                    **edge_props,
                }
                all_edges_data.append(combined)

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    return all_edges_data


async def _find_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
    citation_tracker: "CitationTracker" = None,
):
    """
    Find text chunks related to entities using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity

    Args:
        node_datas: List of entity dictionaries
        query_param: Query parameters
        text_chunks_db: Text chunks key-value storage
        knowledge_graph_inst: Knowledge graph storage instance
        query: User query string
        chunks_vdb: Chunks vector database
        chunk_tracking: Dictionary for tracking chunk sources
        query_embedding: Query embedding vector
        citation_tracker: Citation tracker for tracking retrieved sources (optional)

    Returns:
        List of chunks related to entities
    """
    logger.debug(f"Finding text chunks from {len(node_datas)} entities")

    if not node_datas:
        return []

    # Step 1: Collect all text chunks for each entity
    entities_with_chunks = []
    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(
                entity["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )

    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned entities)
    chunk_occurrence_count = {}
    for entity_info in entities_with_chunks:
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier entity, so skip it

        # Update entity's chunks to deduplicated chunks
        entity_info["chunks"] = deduplicated_chunks

    # Step 3: Sort chunks for each entity by occurrence count (higher count = higher priority)
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        sorted_chunks = sorted(
            entity_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        entity_info["sorted_chunks"] = sorted_chunks
        total_entity_chunks += len(sorted_chunks)

    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    # Step 4: Apply the selected chunk selection algorithm
    # Pick by vector similarity:
    #     The order of text chunks aligns with the naive retrieval's destination.
    #     When reranking is disabled, the text chunks delivered to the LLM tend to favor naive retrieval.
    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(entities_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                other_rags = getattr(query_param, "other_rag_instances", [])
                chunks_vdbs = (
                    {getattr(chunks_vdb, "workspace", ""): chunks_vdb}
                    if chunks_vdb
                    else {}
                )
                for other_rag in other_rags:
                    if hasattr(other_rag, "chunks_vdb") and other_rag.chunks_vdb:
                        chunks_vdbs[getattr(other_rag, "workspace", "")] = (
                            other_rag.chunks_vdb
                        )

                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=entities_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                    chunks_vdbs=chunks_vdbs,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No entity-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Pick by entity and chunk weight:
        #     When reranking is disabled, delivered more solely KG related chunks to the LLM
        selected_chunk_ids = pick_by_weighted_polling(
            entities_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by weighted polling"
        )

    if not selected_chunk_ids:
        return []

    # Map chunk_id to its workspace
    chunk_to_workspace = {}
    for entity_info in entities_with_chunks:
        ws = entity_info.get("entity_data", {}).get("workspace", "")
        for chunk_id in entity_info.get("sorted_chunks", []):
            if chunk_id not in chunk_to_workspace and ws:
                chunk_to_workspace[chunk_id] = ws

    # Step 5: Batch retrieve chunk data across multiple workspaces
    unique_chunk_ids = list(dict.fromkeys(selected_chunk_ids))

    other_rags = getattr(query_param, "other_rag_instances", [])
    kv_map = {getattr(text_chunks_db, "workspace", ""): text_chunks_db}
    for other_rag in other_rags:
        if hasattr(other_rag, "text_chunks") and other_rag.text_chunks:
            kv_map[getattr(other_rag, "workspace", "")] = other_rag.text_chunks

    grouped_chunk_ids = {}
    for cid in unique_chunk_ids:
        ws = chunk_to_workspace.get(cid, getattr(text_chunks_db, "workspace", ""))
        grouped_chunk_ids.setdefault(ws, []).append(cid)

    fetch_tasks = []
    ws_keys = []
    for ws, cids in grouped_chunk_ids.items():
        kv = kv_map.get(ws, text_chunks_db)
        fetch_tasks.append(kv.get_by_ids(cids))
        ws_keys.append((ws, cids))

    fetch_results = await asyncio.gather(*fetch_tasks)

    chunk_data_lookup = {}
    for (ws, cids), res in zip(ws_keys, fetch_results):
        for cid, data in zip(cids, res):
            if data is not None:
                data_copy = data.copy()
                data_copy["workspace"] = ws
                chunk_data_lookup[cid] = data_copy

    chunk_data_list = [chunk_data_lookup.get(cid) for cid in unique_chunk_ids]

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final entity-related results
                }

            # NEW: Track citation if citation_tracker is provided
            if citation_tracker is not None:
                _track_chunk_citation(chunk_data_copy, citation_tracker, text_chunks_db)

    return result_chunks


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding=None,
):
    logger.info(
        f"Query edges: {keywords} (top_k:{query_param.top_k}, cosine:{relationships_vdb.cosine_better_than_threshold})"
    )

    other_rags = getattr(query_param, "other_rag_instances", [])
    vdbs = [relationships_vdb]
    for other_rag in other_rags:
        if hasattr(other_rag, "relationships_vdb") and other_rag.relationships_vdb:
            vdbs.append(other_rag.relationships_vdb)

    tasks = [
        vdb.query(keywords, top_k=query_param.top_k, query_embedding=query_embedding)
        for vdb in vdbs
    ]
    all_results = await asyncio.gather(*tasks)

    # Resolve corresponding knowledge_graph_inst
    kgs = [knowledge_graph_inst]
    for other_rag in other_rags:
        if (
            hasattr(other_rag, "chunk_entity_relation_graph")
            and other_rag.chunk_entity_relation_graph
        ):
            kgs.append(other_rag.chunk_entity_relation_graph)

    kg_tasks = []
    workspace_mapping = []
    for kg, results in zip(kgs, all_results):
        if results:
            edge_pairs_dicts = [
                {"src": r["src_id"], "tgt": r["tgt_id"]} for r in results
            ]
            kg_tasks.append(kg.get_edges_batch(edge_pairs_dicts))
            workspace_mapping.append((kg, results))

    all_edge_data_dicts = await asyncio.gather(*kg_tasks)

    edge_datas = []
    for (kg, results), edge_data_dict in zip(workspace_mapping, all_edge_data_dicts):
        for k in results:
            pair = (k["src_id"], k["tgt_id"])
            edge_props = edge_data_dict.get(pair)
            if edge_props is not None:
                if "weight" not in edge_props:
                    edge_props["weight"] = 1.0
                combined = {
                    "src_id": k["src_id"],
                    "tgt_id": k["tgt_id"],
                    "created_at": k.get("created_at", None),
                    "workspace": getattr(
                        kg, "workspace", ""
                    ),  # CRITICAL: add workspace info
                    **edge_props,
                }
                edge_datas.append(combined)

    # Find related entities routing query to respective graph databases
    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entities, {len(edge_datas)} relations"
    )

    return edge_datas, use_entities


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    other_rags = getattr(query_param, "other_rag_instances", [])
    kg_map = {getattr(knowledge_graph_inst, "workspace", ""): knowledge_graph_inst}
    for other_rag in other_rags:
        if (
            hasattr(other_rag, "chunk_entity_relation_graph")
            and other_rag.chunk_entity_relation_graph
        ):
            kg_map[getattr(other_rag, "workspace", "")] = (
                other_rag.chunk_entity_relation_graph
            )

    grouped_entities = {}
    for e in edge_datas:
        ws = e.get("workspace", getattr(knowledge_graph_inst, "workspace", ""))
        grouped_entities.setdefault(ws, set()).add(e["src_id"])
        grouped_entities.setdefault(ws, set()).add(e["tgt_id"])

    # Query get_nodes_batch for each workspace group
    tasks = []
    workspace_keys = []
    for ws, entities_set in grouped_entities.items():
        kg = kg_map.get(ws, knowledge_graph_inst)
        tasks.append(kg.get_nodes_batch(list(entities_set)))
        workspace_keys.append((ws, list(entities_set)))

    results = await asyncio.gather(*tasks)

    # Build lookup dictionary
    lookup = {}
    for (ws, entities_list), nodes_dict in zip(workspace_keys, results):
        for ent_name in entities_list:
            node = nodes_dict.get(ent_name)
            if node is not None:
                lookup[(ent_name, ws)] = node

    node_datas = []
    seen = set()
    for e in edge_datas:
        ws = e.get("workspace", getattr(knowledge_graph_inst, "workspace", ""))
        for field in ["src_id", "tgt_id"]:
            ent_name = e[field]
            key = (ent_name, ws)
            if key not in seen and key in lookup:
                node = lookup[key]
                node_datas.append(
                    {
                        **node,
                        "entity_name": ent_name,
                        "workspace": ws,  # CRITICAL: add workspace info
                    }
                )
                seen.add(key)

    return node_datas


async def _find_related_text_unit_from_relations(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    entity_chunks: list[dict] = None,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
    citation_tracker: "CitationTracker" = None,
):
    """
    Find text chunks related to relationships using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity

    Args:
        edge_datas: List of relation dictionaries
        query_param: Query parameters
        text_chunks_db: Text chunks key-value storage
        entity_chunks: List of chunks from entities (for deduplication)
        query: User query string
        chunks_vdb: Chunks vector database
        chunk_tracking: Dictionary for tracking chunk sources
        query_embedding: Query embedding vector
        citation_tracker: Citation tracker for tracking retrieved sources (optional)

    Returns:
        List of chunks related to relations
    """
    logger.debug(f"Finding text chunks from {len(edge_datas)} relations")

    if not edge_datas:
        return []

    # Step 1: Collect all text chunks for each relationship
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(
                relation["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                # Build relation identifier
                if "src_tgt" in relation:
                    src_id, tgt_id = relation["src_tgt"]
                else:
                    src_id, tgt_id = relation.get("src_id"), relation.get("tgt_id")

                if relation.get("directed"):
                    rel_key = (src_id, tgt_id)
                else:
                    rel_key = tuple(sorted([src_id, tgt_id]))

                relations_with_chunks.append(
                    {
                        "relation_key": rel_key,
                        "chunks": chunks,
                        "relation_data": relation,
                    }
                )

    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned relationships)
    # Also remove duplicates with entity_chunks

    # Extract chunk IDs from entity_chunks for deduplication
    entity_chunk_ids = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)

    chunk_occurrence_count = {}
    # Track unique chunk_ids that have been removed to avoid double counting
    removed_entity_chunk_ids = set()

    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            # Skip chunks that already exist in entity_chunks
            if chunk_id in entity_chunk_ids:
                # Only count each unique chunk_id once
                removed_entity_chunk_ids.add(chunk_id)
                continue

            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier relationship, so skip it

        # Update relationship's chunks to deduplicated chunks
        relation_info["chunks"] = deduplicated_chunks

    # Check if any relations still have chunks after deduplication
    relations_with_chunks = [
        relation_info
        for relation_info in relations_with_chunks
        if relation_info["chunks"]
    ]

    if not relations_with_chunks:
        logger.info(
            f"Find no additional relations-related chunks from {len(edge_datas)} relations"
        )
        return []

    # Step 3: Sort chunks for each relationship by occurrence count (higher count = higher priority)
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        sorted_chunks = sorted(
            relation_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        relation_info["sorted_chunks"] = sorted_chunks
        total_relation_chunks += len(sorted_chunks)

    logger.info(
        f"Find {total_relation_chunks} additional chunks in {len(relations_with_chunks)} relations (deduplicated {len(removed_entity_chunk_ids)})"
    )

    # Step 4: Apply the selected chunk selection algorithm
    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(relations_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                other_rags = getattr(query_param, "other_rag_instances", [])
                chunks_vdbs = (
                    {getattr(chunks_vdb, "workspace", ""): chunks_vdb}
                    if chunks_vdb
                    else {}
                )
                for other_rag in other_rags:
                    if hasattr(other_rag, "chunks_vdb") and other_rag.chunks_vdb:
                        chunks_vdbs[getattr(other_rag, "workspace", "")] = (
                            other_rag.chunks_vdb
                        )

                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=relations_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                    chunks_vdbs=chunks_vdbs,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No relation-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Apply linear gradient weighted polling algorithm
        selected_chunk_ids = pick_by_weighted_polling(
            relations_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by weighted polling"
        )

    logger.debug(
        f"KG related chunks: {len(entity_chunks)} from entitys, {len(selected_chunk_ids)} from relations"
    )

    if not selected_chunk_ids:
        return []

    # Map chunk_id to its workspace
    chunk_to_workspace = {}
    for relation_info in relations_with_chunks:
        ws = relation_info.get("relation_data", {}).get("workspace", "")
        for chunk_id in relation_info.get("chunks", []):
            if chunk_id not in chunk_to_workspace and ws:
                chunk_to_workspace[chunk_id] = ws

    # Step 5: Batch retrieve chunk data across multiple workspaces
    unique_chunk_ids = list(dict.fromkeys(selected_chunk_ids))

    other_rags = getattr(query_param, "other_rag_instances", [])
    kv_map = {getattr(text_chunks_db, "workspace", ""): text_chunks_db}
    for other_rag in other_rags:
        if hasattr(other_rag, "text_chunks") and other_rag.text_chunks:
            kv_map[getattr(other_rag, "workspace", "")] = other_rag.text_chunks

    grouped_chunk_ids = {}
    for cid in unique_chunk_ids:
        ws = chunk_to_workspace.get(cid, getattr(text_chunks_db, "workspace", ""))
        grouped_chunk_ids.setdefault(ws, []).append(cid)

    fetch_tasks = []
    ws_keys = []
    for ws, cids in grouped_chunk_ids.items():
        kv = kv_map.get(ws, text_chunks_db)
        fetch_tasks.append(kv.get_by_ids(cids))
        ws_keys.append((ws, cids))

    fetch_results = await asyncio.gather(*fetch_tasks)

    chunk_data_lookup = {}
    for (ws, cids), res in zip(ws_keys, fetch_results):
        for cid, data in zip(cids, res):
            if data is not None:
                data_copy = data.copy()
                data_copy["workspace"] = ws
                chunk_data_lookup[cid] = data_copy

    chunk_data_list = [chunk_data_lookup.get(cid) for cid in unique_chunk_ids]

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final relation-related results
                }

            # NEW: Track citation if citation_tracker is provided
            if citation_tracker is not None:
                _track_chunk_citation(chunk_data_copy, citation_tracker, text_chunks_db)

    return result_chunks


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    return_raw_data: Literal[True] = True,
) -> dict[str, Any]: ...


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    return_raw_data: Literal[False] = False,
) -> str | AsyncIterator[str]: ...


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> QueryResult | None:
    """
    Execute naive query and return unified QueryResult object.

    Args:
        query: Query string
        chunks_vdb: Document chunks vector database
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Returns None when no relevant chunks are retrieved.
    """

    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    # Spell correction before processing
    if query_param.enable_spell_correction:
        try:
            from lightrag.language_detector import detect_language

            detected_lang = detect_language(query)
            query = await correct_query_spelling(
                query, llm_func=use_model_func, language=detected_lang
            )
        except Exception as e:
            logger.warning(
                f"[Spell Correction] Failed to correct query: {e}, using original"
            )

    tokenizer: Tokenizer = global_config["tokenizer"]
    if not tokenizer:
        logger.error("Tokenizer not found in global configuration.")
        return QueryResult(content=PROMPTS["fail_response"])

    # Create citation tracker if citations are enabled
    citation_tracker = None
    if query_param.enable_citations:
        from lightrag.citation import CitationTracker

        citation_tracker = CitationTracker()

    # Use history-aware keywords for retrieval if conversation history is present
    search_query = query
    if query_param.conversation_history:
        hl_keywords, ll_keywords = await extract_keywords_only(
            query,
            query_param,
            global_config,
            hashing_kv,
            query_param.conversation_history,
        )
        if hl_keywords or ll_keywords:
            search_query = f"{' '.join(hl_keywords)} {' '.join(ll_keywords)}".strip()
            logger.debug(f"[naive_query] Expanded search query: {search_query}")

    chunks = await _get_vector_context(search_query, chunks_vdb, query_param, None)

    if chunks is None or len(chunks) == 0:
        logger.info(
            "[naive_query] No relevant document chunks found; returning no-result."
        )
        return None

    # Track citations if enabled
    if citation_tracker:
        for chunk in chunks:
            _track_chunk_citation(chunk, citation_tracker, None)

    # Calculate dynamic token limit for chunks
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Calculate system prompt template tokens (excluding content_data)
    language_lock = _build_language_lock_instruction(query)
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else ""
    user_prompt = f"{user_prompt}{language_lock}" if user_prompt else language_lock
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Use the provided system prompt or default
    # Initialize PromptManager for language-aware prompt retrieval
    prompt_manager = PromptManager(PROMPTS)

    # Auto-detect language from query or use system prompt
    if system_prompt:
        sys_prompt_template = system_prompt
    else:
        sys_prompt_template = prompt_manager.get_prompt(
            "naive_rag_response", auto_detect_text=query
        )

    # Add citation instructions if citations are enabled
    if query_param.enable_citations and citation_tracker:
        from lightrag.citation import append_citation_instruction_to_prompt
        from lightrag.language_detector import detect_language

        # Detect language for citation instructions
        detected_lang = detect_language(query)
        language = "Vietnamese" if detected_lang == "vi" else "English"

        # Append citation instructions to system prompt
        sys_prompt_template = append_citation_instruction_to_prompt(
            sys_prompt_template,
            enable_citations=True,
            doc_type=None,  # Support both legal and general documents
            language=language,
        )

    # Create a preliminary system prompt with empty content_data to calculate overhead
    pre_sys_prompt = sys_prompt_template.format(
        response_type=response_type,
        user_prompt=user_prompt,
        content_data="",  # Empty for overhead calculation
    )

    # Calculate available tokens for chunks
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200  # reserved for reference list and safety buffer
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + query_tokens + buffer_tokens
    )

    logger.debug(
        f"Naive query token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Process chunks using unified processing with dynamic token limit
    processed_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        source_type="vector",
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    # Generate reference list from processed chunks using the new common function
    reference_list, processed_chunks_with_ref_ids = generate_reference_list_from_chunks(
        processed_chunks
    )

    # Synchronize reference_id to CitationTracker
    if citation_tracker:
        for chunk in processed_chunks_with_ref_ids:
            chunk_id = chunk.get("chunk_id") or chunk.get("id", "")
            doc_id = chunk_id.split(":", 1)[0] if ":" in chunk_id else ""
            cit = citation_tracker.get_citation_by_chunk_id(doc_id, chunk_id)
            if cit:
                cit.reference_id = chunk["reference_id"]

    logger.info(f"Final context: {len(processed_chunks_with_ref_ids)} chunks")

    # Build raw data structure for naive mode using processed chunks with reference IDs
    raw_data = convert_to_user_format(
        [],  # naive mode has no entities
        [],  # naive mode has no relationships
        processed_chunks_with_ref_ids,
        reference_list,
        "naive",
    )

    # Add complete metadata for naive mode
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}
    raw_data["metadata"]["keywords"] = {
        "high_level": [],  # naive mode has no keyword extraction
        "low_level": [],  # naive mode has no keyword extraction
    }
    raw_data["metadata"]["processing_info"] = {
        "total_chunks_found": len(chunks),
        "final_chunks_count": len(processed_chunks_with_ref_ids),
    }

    # Build chunks_context from processed chunks with reference IDs
    chunks_context = []
    for i, chunk in enumerate(processed_chunks_with_ref_ids):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    naive_context_template = prompt_manager.get_prompt(
        "naive_query_context", auto_detect_text=query
    )
    context_content = naive_context_template.format(
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(content=context_content, raw_data=raw_data)

    sys_prompt = sys_prompt_template.format(
        response_type=query_param.response_type,
        user_prompt=user_prompt,
        content_data=context_content,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=raw_data)

    # Handle cache
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        query_param.user_prompt or "",
        query_param.enable_rerank,
        "lang_lock_v1",
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True,
            stream=query_param.stream,
            cache_type="query",
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response[len(sys_prompt) :]
                .replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        # Render citations if enabled
        if query_param.enable_citations and citation_tracker:
            from lightrag.citation import CitationRenderer
            from lightrag.language_detector import detect_language
            from lightrag.citation_metrics import get_global_metrics_tracker

            # Detect language for citation rendering
            detected_lang = detect_language(query)
            language = "Vietnamese" if detected_lang == "vi" else "English"

            # Get global metrics tracker
            metrics_tracker = get_global_metrics_tracker()

            # Create renderer and render citations
            renderer = CitationRenderer(
                citation_tracker=citation_tracker,
                format=query_param.citation_format,
                language=language,
                citation_order=query_param.citation_order,
                metrics_tracker=metrics_tracker,
            )

            # Render citations in the response
            rendered_response, citation_list = renderer.render(response)

            # Track query-level usage metrics
            metrics_tracker.track_query(
                enable_citations=True,
                citation_count=citation_tracker.get_citation_count(),
                citation_format=query_param.citation_format,
                workspaces=citation_list.workspaces_used,
            )

            return QueryResult(
                content=rendered_response,
                raw_data=raw_data,
                citations=citation_list,
                citation_count=citation_tracker.get_citation_count(),
            )
        else:
            # Track query without citations
            from lightrag.citation_metrics import get_global_metrics_tracker

            metrics_tracker = get_global_metrics_tracker()
            metrics_tracker.track_query(
                enable_citations=False,
                citation_count=0,
            )

        return QueryResult(content=response, raw_data=raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response, raw_data=raw_data, is_streaming=True
        )
