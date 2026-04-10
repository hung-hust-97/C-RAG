"""
Factory functions for LightRAG instance creation.
Extracted from lightrag_server.py to support standalone usage (e.g. Celery workers).
"""

import os
import json
import asyncio
import re
import logging
from pathlib import Path
from lightrag import LightRAG
from lightrag.utils import logger, get_env_value, EmbeddingFunc
from lightrag.constants import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
)
from lightrag.types import GPTKeywordExtractionFormat

class LLMConfigCache:
    """Smart LLM and Embedding configuration cache class"""

    def __init__(self, args):
        self.args = args

        # Initialize configurations based on binding conditions
        self.openai_llm_options = None
        self.openai_llm_options_query = None
        self.openai_llm_options_extract = None
        self.gemini_llm_options = None
        self.gemini_embedding_options = None
        self.ollama_llm_options = None
        self.ollama_embedding_options = None

        # Only initialize and log OpenAI options when using OpenAI-related bindings
        if args.llm_binding in ["openai", "azure_openai"]:
            from lightrag.llm.binding_options import OpenAILLMOptions

            self.openai_llm_options = OpenAILLMOptions.options_dict(args)
            logger.info(f"OpenAI LLM Options: {self.openai_llm_options}")
            self.openai_llm_options_query = dict(self.openai_llm_options)
            self.openai_llm_options_extract = dict(self.openai_llm_options)

            # Allow independent output limits for query-answering and KG extraction.
            query_max_tokens = os.getenv("OPENAI_LLM_MAX_TOKENS_QUERY")
            if query_max_tokens and query_max_tokens.strip():
                self.openai_llm_options_query["max_tokens"] = int(query_max_tokens)

            query_max_completion_tokens = os.getenv(
                "OPENAI_LLM_MAX_COMPLETION_TOKENS_QUERY"
            )
            if query_max_completion_tokens and query_max_completion_tokens.strip():
                self.openai_llm_options_query["max_completion_tokens"] = int(
                    query_max_completion_tokens
                )

            extract_max_tokens = os.getenv("OPENAI_LLM_MAX_TOKENS_EXTRACT")
            if extract_max_tokens and extract_max_tokens.strip():
                self.openai_llm_options_extract["max_tokens"] = int(extract_max_tokens)

            extract_max_completion_tokens = os.getenv(
                "OPENAI_LLM_MAX_COMPLETION_TOKENS_EXTRACT"
            )
            if extract_max_completion_tokens and extract_max_completion_tokens.strip():
                self.openai_llm_options_extract["max_completion_tokens"] = int(
                    extract_max_completion_tokens
                )

            if (
                self.openai_llm_options_query != self.openai_llm_options
                or self.openai_llm_options_extract != self.openai_llm_options
            ):
                logger.info(
                    "OpenAI LLM split options enabled: "
                    f"query={self.openai_llm_options_query}, "
                    f"extract={self.openai_llm_options_extract}"
                )

        if args.llm_binding == "gemini":
            from lightrag.llm.binding_options import GeminiLLMOptions

            self.gemini_llm_options = GeminiLLMOptions.options_dict(args)
            logger.info(f"Gemini LLM Options: {self.gemini_llm_options}")

        # Only initialize and log Ollama LLM options when using Ollama LLM binding
        if args.llm_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                self.ollama_llm_options = OllamaLLMOptions.options_dict(args)
                logger.info(f"Ollama LLM Options: {self.ollama_llm_options}")
            except ImportError:
                logger.warning(
                    "OllamaLLMOptions not available, using default configuration"
                )
                self.ollama_llm_options = {}

        # Only initialize and log Ollama Embedding options when using Ollama Embedding binding
        if args.embedding_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaEmbeddingOptions

                self.ollama_embedding_options = OllamaEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Ollama Embedding Options: {self.ollama_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "OllamaEmbeddingOptions not available, using default configuration"
                )
                self.ollama_embedding_options = {}

        # Only initialize and log Gemini Embedding options when using Gemini Embedding binding
        if args.embedding_binding == "gemini":
            try:
                from lightrag.llm.binding_options import GeminiEmbeddingOptions

                self.gemini_embedding_options = GeminiEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Gemini Embedding Options: {self.gemini_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "GeminiEmbeddingOptions not available, using default configuration"
                )
                self.gemini_embedding_options = {}

    def get_openai_llm_options(self, cache_type: str | None) -> dict | None:
        if self.openai_llm_options is None:
            return None
        kind = (cache_type or "query").lower()
        if kind in {"extract", "summary"} and self.openai_llm_options_extract is not None:
            return dict(self.openai_llm_options_extract)
        if self.openai_llm_options_query is not None:
            return dict(self.openai_llm_options_query)
        return dict(self.openai_llm_options)

def create_optimized_openai_llm_func(config_cache: LLMConfigCache, args, llm_timeout: int):
    async def optimized_openai_alike_model_complete(
        prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs
    ) -> str:
        from lightrag.llm.openai import openai_complete_if_cache
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        request_cache_type = kwargs.pop("cache_type", "query")
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []
        kwargs["timeout"] = llm_timeout
        openai_options = config_cache.get_openai_llm_options(request_cache_type)
        if openai_options:
            kwargs.update(openai_options)
        return await openai_complete_if_cache(
            args.llm_model, prompt, system_prompt=system_prompt,
            history_messages=history_messages, base_url=args.llm_binding_host,
            api_key=args.llm_binding_api_key, **kwargs
        )
    return optimized_openai_alike_model_complete

def create_optimized_azure_openai_llm_func(config_cache: LLMConfigCache, args, llm_timeout: int):
    async def optimized_azure_openai_model_complete(
        prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs
    ) -> str:
        from lightrag.llm.azure_openai import azure_openai_complete_if_cache
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        request_cache_type = kwargs.pop("cache_type", "query")
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []
        kwargs["timeout"] = llm_timeout
        openai_options = config_cache.get_openai_llm_options(request_cache_type)
        if openai_options:
            kwargs.update(openai_options)
        return await azure_openai_complete_if_cache(
            args.llm_model, prompt, system_prompt=system_prompt,
            history_messages=history_messages, base_url=args.llm_binding_host,
            api_key=os.getenv("AZURE_OPENAI_API_KEY", args.llm_binding_api_key),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            **kwargs
        )
    return optimized_azure_openai_model_complete

def create_optimized_gemini_llm_func(config_cache: LLMConfigCache, args, llm_timeout: int):
    async def optimized_gemini_model_complete(
        prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs
    ) -> str:
        from lightrag.llm.gemini import gemini_complete_if_cache
        if history_messages is None:
            history_messages = []
        kwargs["timeout"] = llm_timeout
        if config_cache.gemini_llm_options is not None and "generation_config" not in kwargs:
            kwargs["generation_config"] = dict(config_cache.gemini_llm_options)
        return await gemini_complete_if_cache(
            args.llm_model, prompt, system_prompt=system_prompt,
            history_messages=history_messages, api_key=args.llm_binding_api_key,
            base_url=args.llm_binding_host, keyword_extraction=keyword_extraction, **kwargs
        )
    return optimized_gemini_model_complete

async def bedrock_model_complete(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
    from lightrag.llm.bedrock import bedrock_complete_if_cache
    from lightrag.api.config import global_args as args
    if history_messages is None:
        history_messages = []
    kwargs["temperature"] = get_env_value("BEDROCK_LLM_TEMPERATURE", 1.0, float)
    return await bedrock_complete_if_cache(
        args.llm_model, prompt, system_prompt=system_prompt,
        history_messages=history_messages, **kwargs
    )

def create_llm_model_func(binding: str, config_cache: LLMConfigCache, args, llm_timeout: int):
    try:
        if binding == "lollms":
            from lightrag.llm.lollms import lollms_model_complete
            return lollms_model_complete
        elif binding == "ollama":
            from lightrag.llm.ollama import ollama_model_complete
            return ollama_model_complete
        elif binding == "aws_bedrock":
            return bedrock_model_complete
        elif binding == "azure_openai":
            return create_optimized_azure_openai_llm_func(config_cache, args, llm_timeout)
        elif binding == "gemini":
            return create_optimized_gemini_llm_func(config_cache, args, llm_timeout)
        else:
            return create_optimized_openai_llm_func(config_cache, args, llm_timeout)
    except ImportError as e:
        raise Exception(f"Failed to import {binding} LLM binding: {e}")

def create_llm_model_kwargs(binding: str, args, llm_timeout: int) -> dict:
    if binding in ["lollms", "ollama"]:
        try:
            from lightrag.llm.binding_options import OllamaLLMOptions
            return {
                "host": args.llm_binding_host,
                "timeout": llm_timeout,
                "options": OllamaLLMOptions.options_dict(args),
                "api_key": args.llm_binding_api_key,
            }
        except ImportError as e:
            raise Exception(f"Failed to import {binding} options: {e}")
    return {}

def create_optimized_embedding_function(config_cache: LLMConfigCache, binding, model, host, api_key, args) -> EmbeddingFunc:
    provider_func = None
    provider_max_token_size = None
    provider_embedding_dim = None

    try:
        if binding == "openai":
            from lightrag.llm.openai import openai_embed
            provider_func = openai_embed
        elif binding == "ollama":
            from lightrag.llm.ollama import ollama_embed
            provider_func = ollama_embed
        elif binding == "gemini":
            from lightrag.llm.gemini import gemini_embed
            provider_func = gemini_embed
        elif binding == "jina":
            from lightrag.llm.jina import jina_embed
            provider_func = jina_embed
        elif binding == "azure_openai":
            from lightrag.llm.azure_openai import azure_openai_embed
            provider_func = azure_openai_embed
        elif binding == "aws_bedrock":
            from lightrag.llm.bedrock import bedrock_embed
            provider_func = bedrock_embed
        elif binding == "lollms":
            from lightrag.llm.lollms import lollms_embed
            provider_func = lollms_embed

        if provider_func and isinstance(provider_func, EmbeddingFunc):
            provider_max_token_size = provider_func.max_token_size
            provider_embedding_dim = provider_func.embedding_dim
    except ImportError as e:
        logger.warning(f"Could not import provider function for {binding}: {e}")

    final_max_token_size = args.embedding_token_limit or provider_max_token_size
    final_embedding_dim = args.embedding_dim if args.embedding_dim else provider_embedding_dim

    async def optimized_embedding_function(texts, embedding_dim=None, max_token_size=None):
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_embed
                actual_func = lollms_embed.func if isinstance(lollms_embed, EmbeddingFunc) else lollms_embed
                return await actual_func(texts, base_url=host, api_key=api_key)
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_embed
                actual_func = ollama_embed.func if isinstance(ollama_embed, EmbeddingFunc) else ollama_embed
                if config_cache.ollama_embedding_options is not None:
                    ollama_options = config_cache.ollama_embedding_options
                else:
                    from lightrag.llm.binding_options import OllamaEmbeddingOptions
                    ollama_options = OllamaEmbeddingOptions.options_dict(args)
                kwargs = {"texts": texts, "host": host, "api_key": api_key, "options": ollama_options, "max_token_size": max_token_size}
                if model: kwargs["embed_model"] = model
                return await actual_func(**kwargs)
            elif binding == "azure_openai":
                from lightrag.llm.azure_openai import azure_openai_embed
                actual_func = azure_openai_embed.func if isinstance(azure_openai_embed, EmbeddingFunc) else azure_openai_embed
                kwargs = {"texts": texts, "api_key": api_key, "embedding_dim": embedding_dim, "max_token_size": max_token_size}
                if model: kwargs["model"] = model
                return await actual_func(**kwargs)
            elif binding == "aws_bedrock":
                from lightrag.llm.bedrock import bedrock_embed
                actual_func = bedrock_embed.func if isinstance(bedrock_embed, EmbeddingFunc) else bedrock_embed
                kwargs = {"texts": texts}
                if model: kwargs["model"] = model
                return await actual_func(**kwargs)
            elif binding == "jina":
                from lightrag.llm.jina import jina_embed
                actual_func = jina_embed.func if isinstance(jina_embed, EmbeddingFunc) else jina_embed
                kwargs = {"texts": texts, "embedding_dim": embedding_dim, "base_url": host, "api_key": api_key, "max_token_size": max_token_size}
                if model: kwargs["model"] = model
                return await actual_func(**kwargs)
            elif binding == "gemini":
                from lightrag.llm.gemini import gemini_embed
                actual_func = gemini_embed.func if isinstance(gemini_embed, EmbeddingFunc) else gemini_embed
                if config_cache.gemini_embedding_options is not None:
                    gemini_options = config_cache.gemini_embedding_options
                else:
                    from lightrag.llm.binding_options import GeminiEmbeddingOptions
                    gemini_options = GeminiEmbeddingOptions.options_dict(args)
                kwargs = {"texts": texts, "base_url": host, "api_key": api_key, "embedding_dim": embedding_dim, "task_type": gemini_options.get("task_type", "RETRIEVAL_DOCUMENT"), "max_token_size": max_token_size}
                if model: kwargs["model"] = model
                return await actual_func(**kwargs)
            else:
                from lightrag.llm.openai import openai_embed
                actual_func = openai_embed.func if isinstance(openai_embed, EmbeddingFunc) else openai_embed
                kwargs = {"texts": texts, "base_url": host, "api_key": api_key, "embedding_dim": embedding_dim, "max_token_size": max_token_size}
                if model: kwargs["model"] = model
                return await actual_func(**kwargs)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} embedding: {e}")

    return EmbeddingFunc(
        embedding_dim=final_embedding_dim,
        func=optimized_embedding_function,
        max_token_size=final_max_token_size,
        send_dimensions=False,
        model_name=model,
    )

def build_rag_instance(workspace_id: str, args, config_cache: LLMConfigCache) -> LightRAG:
    # Resolve timeouts
    llm_timeout = get_env_value("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, int)
    embedding_timeout = get_env_value("EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT, int)
    
    # Embedding Function
    embedding_func = create_optimized_embedding_function(
        config_cache=config_cache,
        binding=args.embedding_binding,
        model=args.embedding_model,
        host=args.embedding_binding_host,
        api_key=args.embedding_binding_api_key,
        args=args,
    )

    # Rerank Function (Simplified for factory)
    rerank_model_func = None # Optional in basic build

    return LightRAG(
        working_dir=args.working_dir,
        workspace=workspace_id,
        llm_model_func=create_llm_model_func(args.llm_binding, config_cache, args, llm_timeout),
        llm_model_name=args.llm_model,
        llm_model_max_async=args.max_async,
        summary_max_tokens=args.summary_max_tokens,
        summary_context_size=args.summary_context_size,
        chunk_token_size=int(args.chunk_size),
        chunk_overlap_token_size=int(args.chunk_overlap_size),
        llm_model_kwargs=create_llm_model_kwargs(args.llm_binding, args, llm_timeout),
        embedding_func=embedding_func,
        default_llm_timeout=llm_timeout,
        default_embedding_timeout=embedding_timeout,
        kv_storage=args.kv_storage,
        graph_storage=args.graph_storage,
        vector_storage=args.vector_storage,
        doc_status_storage=args.doc_status_storage,
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": args.cosine_threshold},
        enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
        enable_llm_cache=args.enable_llm_cache,
        rerank_model_func=rerank_model_func,
        max_parallel_insert=args.max_parallel_insert,
        max_graph_nodes=args.max_graph_nodes,
        addon_params={
            "language": getattr(args, "summary_language", "English"),
        },
    )
