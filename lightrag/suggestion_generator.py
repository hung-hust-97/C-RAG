"""
Workspace Question Suggestions Generator

This module provides intelligent question suggestion generation for LightRAG workspaces.
It analyzes knowledge graph content (entities and relationships) to generate contextually
relevant questions that help users discover what they can ask about their workspace.

Key Features:
- LLM-based natural language question generation
- Template-based fallback for resilience
- Multi-language support (Vietnamese/English)
- 1-hour caching with workspace-scoped locking
- Thundering herd prevention for concurrent requests
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import HTTPException

from lightrag.utils import logger


class SuggestionGenerator:
    """Generates contextually relevant question suggestions for a workspace.
    
    This class analyzes a workspace's knowledge graph (entities and relationships)
    to generate natural language questions that users can ask. It uses LLM for
    generation with template-based fallback, implements caching for performance,
    and prevents thundering herd with workspace-scoped locking.
    
    Attributes:
        rag: LightRAG instance for accessing storage and LLM
        cache: In-memory cache for generated suggestions (TTL: 1 hour)
        cache_ttl: Time-to-live for cached suggestions in seconds
        _locks: Dictionary of workspace-scoped asyncio locks for thundering herd prevention
        _locks_lock: Lock for managing the locks dictionary
    """
    
    def __init__(self, rag_instance):
        """Initialize the suggestion generator.
        
        Args:
            rag_instance: LightRAG instance providing access to:
                - full_entities storage for entity data
                - full_relations storage for relationship data
                - llm_model_func for LLM-based generation
                - addon_params for language configuration
                - workspace identifier
        """
        self.rag = rag_instance
        self.cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        self.cache_ttl = 3600  # 1 hour in seconds
        
        # Workspace-scoped locks for preventing thundering herd
        # Each workspace gets its own lock to serialize generation on cache miss
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects the _locks dictionary
        
        logger.debug(f"SuggestionGenerator initialized for workspace: {self.rag.workspace}")
    
    async def _sample_entities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Sample top entities by occurrence count from knowledge graph.
        
        Queries the full_entities storage and returns the most frequently occurring
        entities. This sampling approach reduces LLM context size while focusing on
        the most important entities in the workspace.
        
        Args:
            limit: Maximum number of entities to return (default: 50)
        
        Returns:
            List of entity dictionaries, each containing:
                - id: Entity identifier
                - names: List of entity name variations
                - count: Occurrence frequency in the knowledge graph
            
            Returns empty list if knowledge graph has no entities.
        """
        try:
            # Get all entity IDs from storage
            all_entity_ids = list((await self.rag.full_entities.index_done_callback()).keys())
            
            if not all_entity_ids:
                logger.debug("No entities found in knowledge graph")
                return []
            
            # Retrieve entity data
            all_entities = await self.rag.full_entities.get_by_ids(all_entity_ids)
            
            if not all_entities:
                return []
            
            # Sort by count (occurrence frequency) in descending order
            sorted_entities = sorted(
                all_entities.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Return top N entities with metadata
            sampled = [
                {
                    "id": entity_id,
                    "names": entity_data.get("entity_names", []),
                    "count": entity_data.get("count", 0)
                }
                for entity_id, entity_data in sorted_entities[:limit]
            ]
            
            logger.debug(f"Sampled {len(sampled)} entities from {len(all_entity_ids)} total")
            return sampled
            
        except Exception as e:
            logger.error(f"Error sampling entities: {e}")
            return []
    
    async def _sample_relations(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Sample top relationships by occurrence count from knowledge graph.
        
        Queries the full_relations storage and returns the most frequently occurring
        relationships. This sampling approach reduces LLM context size while focusing
        on the most important relationships in the workspace.
        
        Args:
            limit: Maximum number of relationships to return (default: 30)
        
        Returns:
            List of relationship dictionaries, each containing:
                - id: Relationship identifier
                - pairs: List of [source, target, type] relationship tuples
                - count: Occurrence frequency in the knowledge graph
            
            Returns empty list if knowledge graph has no relationships.
        """
        try:
            # Get all relation IDs from storage
            all_relation_ids = list((await self.rag.full_relations.index_done_callback()).keys())
            
            if not all_relation_ids:
                logger.debug("No relationships found in knowledge graph")
                return []
            
            # Retrieve relation data
            all_relations = await self.rag.full_relations.get_by_ids(all_relation_ids)
            
            if not all_relations:
                return []
            
            # Sort by count (occurrence frequency) in descending order
            sorted_relations = sorted(
                all_relations.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )
            
            # Return top N relations with metadata
            sampled = [
                {
                    "id": relation_id,
                    "pairs": relation_data.get("relation_pairs", []),
                    "count": relation_data.get("count", 0)
                }
                for relation_id, relation_data in sorted_relations[:limit]
            ]
            
            logger.debug(f"Sampled {len(sampled)} relationships from {len(all_relation_ids)} total")
            return sampled
            
        except Exception as e:
            logger.error(f"Error sampling relationships: {e}")
            return []
    
    def _format_entities_for_prompt(self, entities: List[Dict[str, Any]]) -> str:
        """Convert entity list to readable text for LLM prompt.
        
        Extracts entity names from the entity_names array and formats them
        as a bulleted list suitable for inclusion in LLM prompts.
        
        Args:
            entities: List of entity dictionaries with 'names' field
        
        Returns:
            Formatted string with entity names as bulleted list
        """
        if not entities:
            return "No entities available"
        
        formatted_lines = []
        for entity in entities:
            # Get first name from entity_names array
            names = entity.get("names", [])
            if names:
                entity_name = names[0]
                formatted_lines.append(f"- {entity_name}")
        
        return "\n".join(formatted_lines) if formatted_lines else "No entities available"
    
    def _format_relations_for_prompt(self, relations: List[Dict[str, Any]]) -> str:
        """Convert relationship list to readable text for LLM prompt.
        
        Extracts source/target pairs from the relation_pairs array and formats
        them as a bulleted list suitable for inclusion in LLM prompts.
        
        Args:
            relations: List of relationship dictionaries with 'pairs' field
        
        Returns:
            Formatted string with relationships as bulleted list
        """
        if not relations:
            return "No relationships available"
        
        formatted_lines = []
        for relation in relations:
            # Get first pair from relation_pairs array
            pairs = relation.get("pairs", [])
            if pairs and len(pairs) > 0:
                pair = pairs[0]
                if len(pair) >= 2:
                    src, tgt = pair[0], pair[1]
                    formatted_lines.append(f"- {src} → {tgt}")
        
        return "\n".join(formatted_lines) if formatted_lines else "No relationships available"
    
    def _validate_question_format(self, question: str) -> bool:
        """Validate that question is compatible with query endpoint.
        
        Checks that the question meets length requirements and doesn't contain
        problematic characters that would cause parsing errors or compatibility
        issues with the /query endpoint.
        
        Args:
            question: Question string to validate
        
        Returns:
            True if question is valid, False otherwise
        
        Validation Rules:
            - Length must be between 10-150 characters
            - Must not contain null bytes (\x00)
            - Must not contain newlines (\n)
            - Must not contain carriage returns (\r)
        """
        # Check question length is between 10-150 characters
        if len(question) < 10 or len(question) > 150:
            return False
        
        # Check for problematic characters (null bytes, newlines, carriage returns)
        if any(char in question for char in ['\x00', '\n', '\r']):
            return False
        
        return True
    
    async def _generate_with_llm(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        count: int
    ) -> List[str]:
        """Generate questions using LLM based on sampled content.
        
        Constructs a language-specific prompt with entity and relationship context,
        calls the LLM to generate natural language questions, and filters the results
        to ensure quality and compatibility with the query endpoint.
        
        Args:
            entities: List of sampled entity dictionaries
            relations: List of sampled relationship dictionaries
            count: Number of questions to generate (should always be 20 for cache efficiency)
        
        Returns:
            List of generated question strings, filtered by length (10-150 characters)
            and trimmed to exactly `count` questions.
        
        Raises:
            Exception: If LLM call fails (caller should handle with template fallback)
        """
        # Retrieve language setting from addon_params
        language = self.rag.addon_params.get("language", "English")
        
        # Build entity and relation context using format helper methods
        entity_context = self._format_entities_for_prompt(entities[:20])
        relation_context = self._format_relations_for_prompt(relations[:15])
        
        # Construct language-specific prompts
        prompts = {
            "Vietnamese": f"""Dựa trên các thực thể và mối quan hệ sau từ cơ sở tri thức, hãy tạo {count} câu hỏi tự nhiên mà người dùng có thể hỏi.

Các thực thể chính:
{entity_context}

Các mối quan hệ chính:
{relation_context}

Yêu cầu:
- Tạo câu hỏi đa dạng: câu hỏi thực tế, câu hỏi về mối quan hệ, yêu cầu tóm tắt, so sánh
- Mỗi câu hỏi dài 10-150 ký tự
- Câu hỏi phải tự nhiên và dễ hiểu
- Không trùng lặp hoặc gần trùng lặp
- Chỉ trả về danh sách câu hỏi, mỗi câu một dòng, không đánh số

Câu hỏi:""",
            
            "English": f"""Based on the following entities and relationships from the knowledge base, generate {count} natural questions that users might ask.

Key entities:
{entity_context}

Key relationships:
{relation_context}

Requirements:
- Create diverse questions: factual queries, relationship queries, summary requests, comparisons
- Each question should be 10-150 characters
- Questions must be natural and easy to understand
- No duplicate or near-duplicate questions
- Return only the question list, one per line, without numbering

Questions:"""
        }
        
        # Select appropriate prompt based on language
        prompt = prompts.get(language, prompts["English"])
        
        logger.debug(f"Generating {count} questions with LLM for language: {language}")
        
        # Call LLM model function
        try:
            response = await self.rag.llm_model_func(
                prompt,
                system_prompt="You are a helpful assistant that generates relevant questions based on knowledge graph content."
            )
            
            logger.debug(f"LLM response received: {len(response)} characters")
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
        
        # Parse LLM response into list of questions (split by newline)
        raw_questions = [
            q.strip() 
            for q in response.strip().split('\n') 
            if q.strip()
        ]
        
        # Remove numbering if present (e.g., "1. ", "- ", "• ")
        cleaned_questions = []
        for q in raw_questions:
            # Strip common prefixes
            q = q.lstrip('0123456789.-•*) \t')
            if q:
                cleaned_questions.append(q)
        
        # Filter questions through _validate_question_format
        filtered_questions = []
        invalid_count = 0
        for q in cleaned_questions:
            if self._validate_question_format(q):
                filtered_questions.append(q)
            else:
                invalid_count += 1
                # Log warnings for invalid questions that are filtered out
                logger.warning(
                    f"Filtered out invalid question (length={len(q)}): {q[:50]}..."
                )
        
        logger.debug(
            f"Generated {len(raw_questions)} raw questions, "
            f"{len(filtered_questions)} valid after filtering, "
            f"{invalid_count} invalid filtered out"
        )
        
        # Ensure final question list meets count requirement after filtering
        if len(filtered_questions) < count:
            logger.warning(
                f"Only {len(filtered_questions)} valid questions generated, "
                f"requested {count}. Consider regenerating or using template fallback."
            )
        
        return filtered_questions[:count]
    
    def _generate_template_questions(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        count: int
    ) -> Dict[str, Any]:
        """Generate template-based questions when LLM fails.
        
        Uses predefined question templates with entity and relationship substitution
        to generate fallback questions. This ensures the system remains functional
        even when the LLM service is unavailable.
        
        Args:
            entities: List of sampled entity dictionaries
            relations: List of sampled relationship dictionaries
            count: Number of questions to generate (should always be 20 for cache efficiency)
        
        Returns:
            Dictionary containing:
                - questions: List of generated question strings
                - metadata: Generation metadata including timestamp, language, and method
        """
        import random
        
        language = self.rag.addon_params.get("language", "English")
        questions = []
        
        # Define template dictionaries for Vietnamese and English
        templates = {
            "Vietnamese": {
                "entity": [
                    "{entity} là gì?",
                    "Cho tôi biết thêm về {entity}",
                    "Thông tin về {entity}",
                    "{entity} có vai trò gì?",
                    "Giải thích về {entity}",
                    "Tại sao {entity} quan trọng?",
                    "{entity} hoạt động như thế nào?",
                    "Đặc điểm của {entity} là gì?"
                ],
                "relation": [
                    "Mối quan hệ giữa {src} và {tgt} là gì?",
                    "{src} liên quan đến {tgt} như thế nào?",
                    "So sánh {src} và {tgt}",
                    "{src} ảnh hưởng đến {tgt} ra sao?",
                    "Sự khác biệt giữa {src} và {tgt}?",
                    "{src} và {tgt} có điểm chung gì?"
                ],
                "general": [
                    "Tóm tắt nội dung chính",
                    "Các chủ đề chính là gì?",
                    "Danh sách các khái niệm quan trọng",
                    "Những điểm nổi bật trong workspace?",
                    "Thông tin tổng quan về workspace",
                    "Các vấn đề chính được đề cập?",
                    "Nội dung workspace bao gồm những gì?"
                ]
            },
            "English": {
                "entity": [
                    "What is {entity}?",
                    "Tell me more about {entity}",
                    "Information about {entity}",
                    "What role does {entity} play?",
                    "Explain {entity}",
                    "Why is {entity} important?",
                    "How does {entity} work?",
                    "What are the characteristics of {entity}?"
                ],
                "relation": [
                    "What is the relationship between {src} and {tgt}?",
                    "How does {src} relate to {tgt}?",
                    "Compare {src} and {tgt}",
                    "How does {src} affect {tgt}?",
                    "What's the difference between {src} and {tgt}?",
                    "What do {src} and {tgt} have in common?"
                ],
                "general": [
                    "Summarize the main content",
                    "What are the main topics?",
                    "List important concepts",
                    "What are the highlights in this workspace?",
                    "Overview of workspace information",
                    "What are the main issues discussed?",
                    "What does the workspace content include?"
                ]
            }
        }
        
        lang_templates = templates.get(language, templates["English"])
        
        logger.debug(f"Generating {count} template-based questions for language: {language}")
        
        # Generate entity questions (up to count/2)
        entity_limit = min(len(entities), count // 2)
        for entity in entities[:entity_limit]:
            if len(questions) >= count:
                break
            
            # Get entity name
            entity_name = entity["names"][0] if entity.get("names") else entity.get("id", "unknown")
            
            # Select random template and format
            template = random.choice(lang_templates["entity"])
            question = template.format(entity=entity_name)
            
            # Filter questions through _validate_question_format
            if self._validate_question_format(question):
                questions.append(question)
            else:
                # Log warnings for invalid questions that are filtered out
                logger.warning(
                    f"Filtered out invalid template entity question (length={len(question)}): {question[:50]}..."
                )
        
        # Generate relation questions (up to count/3)
        relation_limit = min(len(relations), count // 3)
        for relation in relations[:relation_limit]:
            if len(questions) >= count:
                break
            
            # Get relation pair
            pairs = relation.get("pairs", [])
            if pairs and len(pairs) > 0:
                pair = pairs[0]
                if len(pair) >= 2:
                    src, tgt = pair[0], pair[1]
                    
                    # Select random template and format
                    template = random.choice(lang_templates["relation"])
                    question = template.format(src=src, tgt=tgt)
                    
                    # Filter questions through _validate_question_format
                    if self._validate_question_format(question):
                        questions.append(question)
                    else:
                        # Log warnings for invalid questions that are filtered out
                        logger.warning(
                            f"Filtered out invalid template relation question (length={len(question)}): {question[:50]}..."
                        )
        
        # Fill remaining slots with general questions
        # Ensure final question list meets count requirement after filtering
        while len(questions) < count:
            question = random.choice(lang_templates["general"])
            if question not in questions:  # Avoid duplicates
                if self._validate_question_format(question):
                    questions.append(question)
                else:
                    # Log warnings for invalid questions that are filtered out
                    logger.warning(
                        f"Filtered out invalid template general question (length={len(question)}): {question[:50]}..."
                    )
        
        logger.debug(f"Generated {len(questions)} template-based questions")
        
        # Return exactly count questions with metadata
        return {
            "questions": questions[:count],
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "language": language,
                "generation_method": "template"
            }
        }
    
    def _generate_generic_questions(self, count: int) -> Dict[str, Any]:
        """Generate generic exploratory questions when workspace is empty.
        
        Returns predefined generic questions that help users understand what they
        can do with an empty workspace. These questions are appropriate when no
        documents have been uploaded yet.
        
        Args:
            count: Number of questions to return (should always be 20 for cache efficiency)
        
        Returns:
            Dictionary containing:
                - questions: List of generic question strings
                - metadata: Generation metadata including timestamp, language, and method
        """
        language = self.rag.addon_params.get("language", "English")
        
        logger.debug(f"Generating {count} generic questions for empty workspace, language: {language}")
        
        # Define generic exploratory questions for Vietnamese (20 questions)
        generic_questions = {
            "Vietnamese": [
                "Workspace này chứa những gì?",
                "Có tài liệu nào trong workspace không?",
                "Tôi có thể hỏi gì về workspace này?",
                "Workspace này được sử dụng để làm gì?",
                "Có dữ liệu nào đã được tải lên chưa?",
                "Làm thế nào để bắt đầu với workspace này?",
                "Workspace này có những tính năng gì?",
                "Tôi nên tải lên loại tài liệu nào?",
                "Workspace này hỗ trợ những định dạng file nào?",
                "Có hướng dẫn sử dụng workspace không?",
                "Workspace này có giới hạn dung lượng không?",
                "Tôi có thể chia sẻ workspace này không?",
                "Làm sao để tìm kiếm trong workspace?",
                "Workspace này có lịch sử truy vấn không?",
                "Tôi có thể xuất dữ liệu từ workspace không?",
                "Workspace này có tích hợp với công cụ nào?",
                "Làm thế nào để quản lý quyền truy cập?",
                "Workspace này có sao lưu tự động không?",
                "Tôi có thể xóa workspace này không?",
                "Có cách nào để tối ưu hóa workspace không?"
            ],
            # Define generic exploratory questions for English (20 questions)
            "English": [
                "What does this workspace contain?",
                "Are there any documents in this workspace?",
                "What can I ask about this workspace?",
                "What is this workspace used for?",
                "Has any data been uploaded yet?",
                "How do I get started with this workspace?",
                "What features does this workspace have?",
                "What types of documents should I upload?",
                "What file formats does this workspace support?",
                "Is there a guide for using this workspace?",
                "Does this workspace have storage limits?",
                "Can I share this workspace with others?",
                "How do I search within this workspace?",
                "Does this workspace keep query history?",
                "Can I export data from this workspace?",
                "What tools does this workspace integrate with?",
                "How do I manage access permissions?",
                "Does this workspace have automatic backups?",
                "Can I delete this workspace?",
                "Are there ways to optimize this workspace?"
            ]
        }
        
        # Return appropriate questions based on workspace language
        questions = generic_questions.get(language, generic_questions["English"])
        
        # Include metadata indicating "generic" generation method
        return {
            "questions": questions[:count],
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "language": language,
                "generation_method": "generic"
            }
        }
    
    def _slice_cached_result(self, cached_data: Dict[str, Any], count: int) -> Dict[str, Any]:
        """Slice cached result to return requested count of questions.
        
        This helper method extracts the requested number of questions from a cached
        result while preserving all metadata. This enables efficient caching of the
        maximum count (20 questions) while serving different count values.
        
        Args:
            cached_data: Dictionary containing full cached result with questions and metadata
            count: Number of questions to return (1-20)
        
        Returns:
            Dictionary containing:
                - questions: Sliced list of question strings (up to count)
                - metadata: Preserved metadata from cached_data
        """
        return {
            "questions": cached_data["questions"][:count],
            "metadata": cached_data["metadata"]
        }
    
    def invalidate_cache(self) -> None:
        """Invalidate all cached suggestions for this workspace.
        
        This method clears the cache dictionary when workspace content changes
        (e.g., documents added or deleted). The locks dictionary is preserved
        to maintain workspace-scoped locking behavior.
        
        This should be called after document operations that modify the knowledge
        graph to ensure users receive fresh suggestions based on updated content.
        """
        workspace_id = self.rag.workspace
        cache_key = f"{workspace_id}"
        
        if cache_key in self.cache:
            del self.cache[cache_key]
            logger.info(f"Invalidated suggestion cache for workspace {workspace_id}")
        else:
            logger.debug(f"No cached suggestions to invalidate for workspace {workspace_id}")
    
    async def generate_suggestions(
        self,
        count: int = 5,
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Generate question suggestions for the workspace with thundering herd prevention.
        
        This is the main orchestration method that coordinates caching, locking, sampling,
        and generation. It implements workspace-scoped locking to prevent the thundering
        herd problem where multiple concurrent requests would trigger duplicate LLM calls
        on cache expiry.
        
        **Thundering Herd Prevention:**
        - Uses workspace-scoped asyncio.Lock (one lock per workspace)
        - Only the first request generates suggestions on cache miss
        - Subsequent concurrent requests wait for the lock, then read from cache
        - Different workspaces can generate concurrently without blocking each other
        
        **Cache Strategy:**
        - Cache key: `{workspace_id}` (no count parameter)
        - Always generates and caches 20 questions (maximum)
        - Slices cached results to requested count when serving
        - TTL: 1 hour (3600 seconds)
        
        **Generation Flow:**
        1. Check cache (outside lock for performance)
        2. On cache miss, acquire workspace-scoped lock
        3. Double-check cache after acquiring lock (another request may have generated)
        4. If still missing, sample entities and relationships
        5. Generate 20 questions using LLM (with timeout) or templates (on failure)
        6. Cache full result (20 questions)
        7. Return sliced result for requested count
        
        Args:
            count: Number of suggestions to return (1-20, default: 5)
            timeout: Maximum time in seconds to wait for LLM generation (default: 10.0)
        
        Returns:
            Dictionary containing:
                - questions: List of question strings (up to count)
                - metadata: Generation metadata including:
                    - timestamp: ISO 8601 timestamp
                    - language: Workspace language (Vietnamese/English)
                    - generation_method: "llm", "template", or "generic"
        
        Raises:
            HTTPException: 504 Gateway Timeout if LLM generation exceeds timeout
        """
        workspace_id = self.rag.workspace
        
        # Build cache key from workspace ID only (no count parameter)
        # We always generate and cache the maximum count (20)
        cache_key = f"{workspace_id}"
        
        # Check cache for existing suggestions with TTL validation (outside lock for performance)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for workspace {workspace_id}, returning sliced result")
                # Return sliced cached data if valid (timestamp within 1 hour)
                return self._slice_cached_result(cached_data, count)
        
        # On cache miss, acquire workspace-scoped lock using double-check pattern
        # Get or create lock for workspace_id from self._locks dictionary (protected by self._locks_lock)
        async with self._locks_lock:
            if workspace_id not in self._locks:
                self._locks[workspace_id] = asyncio.Lock()
                logger.debug(f"Created new lock for workspace {workspace_id}")
            workspace_lock = self._locks[workspace_id]
        
        # Acquire workspace lock using async with workspace_lock
        async with workspace_lock:
            logger.debug(f"Acquired lock for workspace {workspace_id}")
            
            # Double-check cache after acquiring lock (another request may have generated while waiting)
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"Cache hit after lock acquisition for workspace {workspace_id}")
                    # If cache hit after lock, return sliced result
                    return self._slice_cached_result(cached_data, count)
            
            # If still cache miss after lock, call _sample_entities and _sample_relations
            logger.debug(f"Cache miss after lock, generating suggestions for workspace {workspace_id}")
            
            entities = await self._sample_entities(limit=50)
            relations = await self._sample_relations(limit=30)
            
            # Check if workspace has content (entities or relations exist)
            if not entities and not relations:
                # If empty, call _generate_generic_questions(20) and cache result
                logger.info(f"Empty workspace {workspace_id}, generating generic questions")
                result = self._generate_generic_questions(20)
                
                # Store full result (20 questions) in cache with current timestamp
                self.cache[cache_key] = (result, time.time())
                
                # Return sliced result for requested count
                return self._slice_cached_result(result, count)
            
            # If has content, wrap _generate_with_llm in asyncio.wait_for with timeout
            try:
                logger.debug(f"Generating with LLM for workspace {workspace_id}")
                
                questions = await asyncio.wait_for(
                    self._generate_with_llm(entities, relations, 20),
                    timeout=timeout
                )
                
                # Build result dictionary with 20 questions and metadata
                result = {
                    "questions": questions,
                    "metadata": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "language": self.rag.addon_params.get("language", "English"),
                        "generation_method": "llm"
                    }
                }
                
                logger.info(
                    f"Successfully generated {len(questions)} questions with LLM "
                    f"for workspace {workspace_id}"
                )
                
            except asyncio.TimeoutError:
                # On timeout, raise HTTPException with 504 status
                logger.error(
                    f"LLM generation timed out after {timeout}s for workspace {workspace_id}"
                )
                raise HTTPException(
                    status_code=504,
                    detail=f"Suggestion generation timed out after {timeout} seconds"
                )
                
            except Exception as e:
                # On LLM failure, catch exception and call _generate_template_questions
                logger.warning(
                    f"LLM generation failed for workspace {workspace_id}, "
                    f"falling back to templates: {e}"
                )
                
                result = self._generate_template_questions(entities, relations, 20)
            
            # Store full result (20 questions) in cache with current timestamp
            self.cache[cache_key] = (result, time.time())
            logger.debug(f"Cached {len(result['questions'])} questions for workspace {workspace_id}")
            
            # Return sliced result for requested count: result["questions"][:count]
            return self._slice_cached_result(result, count)
