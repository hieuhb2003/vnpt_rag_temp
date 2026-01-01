# =============================================================================
# Semantic Chunker - Split Documents into Retrieval Chunks
# =============================================================================
import re
from uuid import UUID, uuid4
from typing import List, Tuple
from dataclasses import dataclass

import tiktoken

from src.models.document import Section, Chunk
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Sentence:
    """A sentence with metadata for chunking."""
    text: str
    token_count: int


class Chunker:
    """Split document sections into semantic chunks for retrieval.

    Features:
    - Uses tiktoken tokenizer for accurate token counting
    - Respects sentence boundaries for both Vietnamese and English
    - Configurable chunk size and overlap
    - Preserves section metadata in chunks
    - Handles very long sentences gracefully
    """

    # Sentence boundaries for Vietnamese and English
    SENTENCE_SPLIT_PATTERN = re.compile(
        r'(?<=[.!?।॥])\s+(?=[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ])|'  # Sentence ending
        r'(?<=[.!?])\s+(?=[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ])|'  # After space + capital
        r'\n+'
    )

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding: str = "cl100k_base"
    ):
        """
        Initialize Chunker.

        Args:
            chunk_size: Target chunk size in tokens (default from settings)
            chunk_overlap: Overlap between chunks in tokens (default from settings)
            encoding: Tiktoken encoding name
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding)

        logger.info(
            "Initialized Chunker",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding=encoding
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to count tokens, using character fallback: {e}")
            # Rough estimate: 1 token ≈ 4 characters for English
            # Vietnamese might be slightly different
            return len(text) // 3

    def split_into_sentences(self, text: str) -> List[Sentence]:
        """
        Split text into sentences while preserving Vietnamese and English patterns.

        Args:
            text: Input text

        Returns:
            List of Sentence objects with token counts
        """
        if not text or not text.strip():
            return []

        # Split on sentence boundaries
        parts = self.SENTENCE_SPLIT_PATTERN.split(text)

        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                # Merge very short fragments (likely split incorrectly)
                if sentences and len(part) < 30:
                    # Append to previous sentence
                    sentences[-1].text += " " + part
                    sentences[-1].token_count = self.count_tokens(sentences[-1].text)
                else:
                    sentences.append(Sentence(
                        text=part,
                        token_count=self.count_tokens(part)
                    ))

        return sentences

    def chunk_section(
        self,
        section: Section,
        document_id: UUID
    ) -> List[Chunk]:
        """
        Split a section into chunks respecting sentence boundaries.

        Args:
            section: Section to chunk
            document_id: Parent document ID

        Returns:
            List of chunks from this section
        """
        if not section.content or not section.content.strip():
            logger.debug(
                f"Skipping empty section: {section.heading}",
                section_id=str(section.id)
            )
            return []

        logger.debug(
            "Chunking section",
            heading=section.heading,
            section_id=str(section.id)
        )

        # Split into sentences
        sentences = self.split_into_sentences(section.content)

        if not sentences:
            return []

        # Build chunks
        chunks = []
        current_chunk: List[str] = []
        current_tokens = 0
        position = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence.token_count > self.chunk_size and current_chunk:
                # Finish current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    tokens=current_tokens,
                    section=section,
                    document_id=document_id,
                    position=position
                ))
                position += 1

                # Start new chunk with overlap
                current_chunk, current_tokens = self._apply_overlap(sentences, chunks)

            # Add sentence to current chunk
            current_chunk.append(sentence.text)
            current_tokens += sentence.token_count

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(
                text=chunk_text,
                tokens=current_tokens,
                section=section,
                document_id=document_id,
                position=position
            ))

        logger.debug(
            "Created chunks for section",
            heading=section.heading,
            chunk_count=len(chunks)
        )

        return chunks

    def _apply_overlap(
        self,
        all_sentences: List[Sentence],
        existing_chunks: List[Chunk]
    ) -> Tuple[List[str], int]:
        """
        Apply overlap from previous chunk to maintain context.

        Args:
            all_sentences: All sentences in section
            existing_chunks: Chunks created so far

        Returns:
            Tuple of (new_chunk_sentences, new_chunk_token_count)
        """
        if self.chunk_overlap == 0 or not existing_chunks:
            return [], 0

        # Get sentences from last chunk for overlap
        last_chunk_text = existing_chunks[-1].content
        overlap_tokens = 0
        overlap_sentences = []

        # Work backwards from last sentence to find overlap
        sentences_reversed = list(reversed(all_sentences))

        for sentence in sentences_reversed:
            if sentence.text not in last_chunk_text:
                continue

            if overlap_tokens + sentence.token_count > self.chunk_overlap:
                break

            overlap_sentences.insert(0, sentence.text)
            overlap_tokens += sentence.token_count

        return overlap_sentences, overlap_tokens

    def _create_chunk(
        self,
        text: str,
        tokens: int,
        section: Section,
        document_id: UUID,
        position: int
    ) -> Chunk:
        """
        Create a Chunk object with proper metadata.

        Args:
            text: Chunk content
            tokens: Token count
            section: Parent section
            document_id: Document ID
            position: Position in document

        Returns:
            Chunk object
        """
        metadata = {
            "heading": section.heading,
            "level": section.level,
            "section_path": section.section_path,
        }

        # Add parent path for context
        if section.section_path:
            metadata["depth"] = len(section.section_path.split('.'))

        return Chunk(
            id=uuid4(),
            document_id=document_id,
            section_id=section.id,
            content=text.strip(),
            token_count=tokens,
            position=position,
            metadata=metadata
        )

    def chunk_document(
        self,
        sections: List[Section],
        document_id: UUID
    ) -> List[Chunk]:
        """
        Chunk all sections in a document.

        Args:
            sections: List of sections with tree structure
            document_id: Document UUID

        Returns:
            All chunks from all sections
        """
        logger.info(
            "Chunking document",
            section_count=len(sections),
            document_id=str(document_id)
        )

        all_chunks = []
        global_position = 0

        # Sort sections by position to maintain document order
        sorted_sections = sorted(sections, key=lambda s: s.position)

        for section in sorted_sections:
            section_chunks = self.chunk_section(section, document_id)

            # Update global position
            for chunk in section_chunks:
                chunk.position = global_position
                global_position += 1

            all_chunks.extend(section_chunks)

        # Log statistics
        if all_chunks:
            avg_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)
            max_tokens = max(c.token_count for c in all_chunks)
            min_tokens = min(c.token_count for c in all_chunks)

            logger.info(
                "Document chunking complete",
                total_chunks=len(all_chunks),
                avg_tokens=int(avg_tokens),
                min_tokens=min_tokens,
                max_tokens=max_tokens
            )
        else:
            logger.warning("No chunks created from document")

        return all_chunks

    def merge_chunks(self, chunks: List[Chunk]) -> str:
        """
        Merge chunks back into continuous text.

        Useful for reconstructing context during retrieval.

        Args:
            chunks: Chunks to merge (should be in order)

        Returns:
            Merged text
        """
        if not chunks:
            return ""

        # Sort by position
        sorted_chunks = sorted(chunks, key=lambda c: c.position)

        # Join with spaces
        return " ".join(c.content for c in sorted_chunks)


# Singleton instance
chunker = Chunker()
