"""
Markdown preprocessing for OCR output.

This module provides markdown preprocessing capabilities to enhance entity
extraction from DeepSeek OCR output. It parses markdown syntax, extracts
structural information, and optimizes text for the RAG pipeline.
"""

import logging
import re

logger = logging.getLogger(__name__)


class MarkdownPreprocessor:
    """Preprocessor for markdown text from DeepSeek OCR.
    
    This class processes markdown-formatted OCR output to enhance entity
    extraction quality. It can preserve markdown structure for better
    semantic understanding or convert to plain text as needed.
    
    Attributes:
        preserve_structure: Whether to keep markdown structure for entity extraction
    """
    
    def __init__(self, preserve_structure: bool = True):
        """Initialize markdown preprocessor.
        
        Args:
            preserve_structure: Keep markdown structure for better entity extraction.
                When True, maintains semantic structure (headers, tables, lists).
                When False, converts to plain text while preserving content.
        """
        self.preserve_structure = preserve_structure
        logger.debug(
            f"MarkdownPreprocessor initialized with preserve_structure={preserve_structure}"
        )
    
    def preprocess(self, markdown_text: str) -> str:
        """Preprocess markdown text for entity extraction.
        
        Parses markdown syntax (headers, tables, lists, bold/italic), extracts
        structural information, and optimizes text for entity extraction while
        preserving semantic meaning.
        
        Args:
            markdown_text: Raw markdown text from OCR
            
        Returns:
            str: Preprocessed text optimized for entity extraction
            
        Preconditions:
            - markdown_text is not None (may be empty string)
            - markdown_text may contain markdown syntax
            
        Postconditions:
            - Returns processed text preserving semantic meaning
            - Markdown syntax is normalized or removed based on preserve_structure
            - Entity boundaries are preserved
            - Whitespace is normalized while preserving paragraph boundaries
        """
        if not markdown_text:
            return ""
        
        logger.debug(f"Preprocessing markdown text ({len(markdown_text)} chars)")
        
        lines = markdown_text.split("\n")
        processed_lines = []
        current_section = None
        
        for line in lines:
            # Extract headers (# Header, ## Subheader, etc.)
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                header_text = line.lstrip("#").strip()
                current_section = header_text
                
                if self.preserve_structure:
                    # Keep header with newlines for context
                    processed_lines.append(f"\n{header_text}\n")
                else:
                    # Just keep the text
                    processed_lines.append(header_text)
                
                logger.debug(f"Extracted header (level {level}): {header_text}")
                continue
            
            # Parse tables (| col1 | col2 |)
            if "|" in line and line.strip().startswith("|"):
                # Skip table separator lines (|---|---|)
                if re.match(r'^\s*\|[\s\-:]+\|[\s\-:|]*$', line):
                    continue
                
                # Extract table cells
                cells = [cell.strip() for cell in line.split("|")[1:-1]]
                if cells:
                    # Convert table to readable format
                    processed_lines.append(" ".join(cells))
                    logger.debug(f"Extracted table row with {len(cells)} cells")
                continue
            
            # Parse lists (-, *, 1., 2., etc.)
            # Match: "- item", "* item", "+ item", "1. item", "2. item", etc.
            # Must have whitespace after the marker
            list_match = re.match(r'^\s*([-*+])\s+(.+)', line) or re.match(r'^\s*(\d+\.)\s+(.+)', line)
            if list_match:
                # Extract list item text
                list_item = re.sub(r'^\s*[-*+]\s+', '', line).strip()
                list_item = re.sub(r'^\s*\d+\.\s+', '', list_item).strip()
                if list_item:
                    processed_lines.append(list_item)
                    logger.debug(f"Extracted list item: {list_item[:50]}...")
                continue
            
            # Handle emphasis (**bold**, *italic*)
            # Remove markdown syntax but keep text
            cleaned_line = line
            
            # Extract bold text (for logging potential entities)
            bold_matches = re.findall(r'\*\*(.+?)\*\*', line)
            for match in bold_matches:
                logger.debug(f"Found emphasized text (bold): {match}")
            
            # IMPORTANT: Remove bold syntax BEFORE italic to avoid conflicts
            # Bold uses ** and italic uses *, so process ** first
            cleaned_line = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned_line)
            
            # Now remove italic syntax (single *)
            cleaned_line = re.sub(r'\*(.+?)\*', r'\1', cleaned_line)
            
            # Remove links but keep link text
            cleaned_line = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', cleaned_line)
            
            # Remove inline code backticks
            cleaned_line = re.sub(r'`(.+?)`', r'\1', cleaned_line)
            
            # Add cleaned line (preserve empty lines for paragraph boundaries)
            processed_lines.append(cleaned_line)
        
        # Join processed lines
        processed_text = "\n".join(processed_lines)
        
        # Normalize whitespace
        # Max 2 consecutive newlines (preserve paragraph boundaries)
        processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
        
        # Remove trailing/leading whitespace
        processed_text = processed_text.strip()
        
        logger.info(
            f"Markdown preprocessing complete: "
            f"{len(markdown_text)} -> {len(processed_text)} chars"
        )
        
        return processed_text
    
    def extract_structure_hints(self, markdown_text: str) -> dict:
        """Extract structural information from markdown.
        
        Parses markdown text to identify structural elements including headers,
        tables, lists, and emphasis. Returns metadata that can be used to
        improve entity extraction and provide context to the RAG pipeline.
        
        Args:
            markdown_text: Raw markdown text
            
        Returns:
            dict: Structure hints with keys:
                - headers: List of dicts with 'level', 'text', 'line_number'
                - tables: List of dicts with 'cells', 'section', 'line_number'
                - lists: List of dicts with 'text', 'section', 'line_number'
                - emphasis: List of dicts with 'text', 'section', 'type'
                
        Preconditions:
            - markdown_text is not None (may be empty string)
            - markdown_text may contain markdown syntax
            
        Postconditions:
            - Returns dict with all required keys (lists may be empty)
            - All structural elements are identified and categorized
            - Line numbers are accurate (0-indexed)
            - Does not modify input parameter
        """
        if not markdown_text:
            return {
                "headers": [],
                "tables": [],
                "lists": [],
                "emphasis": []
            }
        
        logger.debug(f"Extracting structure hints from markdown ({len(markdown_text)} chars)")
        
        structure_metadata = {
            "headers": [],
            "tables": [],
            "lists": [],
            "emphasis": []
        }
        
        lines = markdown_text.split("\n")
        current_section = None
        line_number = 0
        
        for line in lines:
            # Extract headers (# Header, ## Subheader, etc.)
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                header_text = line.lstrip("#").strip()
                
                structure_metadata["headers"].append({
                    "level": level,
                    "text": header_text,
                    "line_number": line_number
                })
                
                current_section = header_text
                logger.debug(f"Found header (level {level}): {header_text}")
            
            # Parse tables (| col1 | col2 |)
            elif "|" in line and line.strip().startswith("|"):
                # Skip table separator lines (|---|---|)
                if not re.match(r'^\s*\|[\s\-:]+\|[\s\-:|]*$', line):
                    # Extract table cells
                    cells = [cell.strip() for cell in line.split("|")[1:-1]]
                    if cells:
                        structure_metadata["tables"].append({
                            "cells": cells,
                            "section": current_section,
                            "line_number": line_number
                        })
                        logger.debug(f"Found table row with {len(cells)} cells")
            
            # Parse lists (-, *, 1., 2., etc.)
            else:
                list_match = re.match(r'^\s*([-*+])\s+(.+)', line) or re.match(r'^\s*(\d+\.)\s+(.+)', line)
                if list_match:
                    # Extract list item text
                    list_item = re.sub(r'^\s*[-*+]\s+', '', line).strip()
                    list_item = re.sub(r'^\s*\d+\.\s+', '', list_item).strip()
                    
                    if list_item:
                        structure_metadata["lists"].append({
                            "text": list_item,
                            "section": current_section,
                            "line_number": line_number
                        })
                        logger.debug(f"Found list item: {list_item[:50]}...")
            
            # Extract emphasis (**bold**, *italic*)
            # Find bold text
            bold_matches = re.findall(r'\*\*(.+?)\*\*', line)
            for match in bold_matches:
                structure_metadata["emphasis"].append({
                    "text": match,
                    "section": current_section,
                    "type": "bold"
                })
                logger.debug(f"Found bold emphasis: {match}")
            
            # Find italic text (but not bold which uses **)
            # Use negative lookbehind and lookahead to avoid matching ** patterns
            italic_matches = re.findall(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', line)
            for match in italic_matches:
                structure_metadata["emphasis"].append({
                    "text": match,
                    "section": current_section,
                    "type": "italic"
                })
                logger.debug(f"Found italic emphasis: {match}")
            
            line_number += 1
        
        logger.info(
            f"Structure extraction complete: "
            f"{len(structure_metadata['headers'])} headers, "
            f"{len(structure_metadata['tables'])} table rows, "
            f"{len(structure_metadata['lists'])} list items, "
            f"{len(structure_metadata['emphasis'])} emphasized texts"
        )
        
        return structure_metadata
