import os
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.schemas import Chunk, DocumentMetadata
from core.config import settings
import pymupdf  # PyMuPDF
import docx
import logging

logger = logging.getLogger(__name__)

class DocumentIngestionService:
    def __init__(self):
        # Related section groups - these should be kept together
        self.section_groups = {
            'carrier_info': ['Carrier Details', 'Driver Details'],
            'customer_info': ['Customer Details', 'Shipper', 'Consignee'],
            'location_info': ['Pickup', 'Drop', 'Stops'],
            'rate_info': ['Rate Breakdown', 'Agreed Amount'],
            'commodity_info': ['Commodity', 'Description'],
            'instructions': ['Standing Instructions', 'Special Instructions', 'Shipper Instructions', 'Carrier Instructions']
        }
        
        # Semantic separators for logistics documents
        self.semantic_separators = [
            "\n### ",  # Section headers
            "\n## ",   # Major sections
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence breaks
            " ",       # Word breaks
            ""         # Character breaks (fallback)
        ]
        
        # Use larger chunk size for title-based chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE * 2,  # Larger chunks to combine sections
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=self.semantic_separators,
            is_separator_regex=False,
        )

    def process_file(self, file_content: bytes, filename: str) -> List[Chunk]:
        """
        Process a file (PDF, DOCX, TXT) and return semantically meaningful chunks.
        """
        text = ""
        file_ext = os.path.splitext(filename)[1].lower()

        try:
            logger.info(f"Processing file: {filename}")
            if file_ext == ".pdf":
                text = self._extract_text_from_pdf(file_content)
            elif file_ext == ".docx":
                text = self._extract_text_from_docx(file_content)
            elif file_ext == ".txt":
                text = file_content.decode("utf-8")
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            logger.debug(f"Extracted {len(text)} characters from {filename}")
            
            # Add semantic structure (section headers)
            text = self._add_semantic_structure(text)
            
            # Chunk the text with title-based grouping
            chunks = self._chunk_text_with_title_grouping(text, filename)
            
            logger.info(f"Generated {len(chunks)} title-based chunks from {filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
            return []

    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF bytes with page markers."""
        doc = pymupdf.open(stream=file_content, filetype="pdf")
        text = ""
        for i, page in enumerate(doc):
            # Add page marker for better context
            text += f"\n\n### Page {i+1}\n\n"
            text += page.get_text()
        return text

    def _extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX bytes."""
        from io import BytesIO
        doc = docx.Document(BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    def _add_semantic_structure(self, text: str) -> str:
        """
        Add semantic structure markers to logistics documents.
        This helps preserve meaningful sections during chunking.
        """
        # Common section headers in logistics documents
        section_patterns = [
            r'(Carrier Details)',
            r'(Rate Breakdown)',
            r'(Pickup)',
            r'(Drop)',
            r'(Stops)',
            r'(Standing Instructions)',
            r'(Special Instructions)',
            r'(Shipper & Carrier Instructions)',
            r'(Driver Details)',
            r'(Reference ID)',
            r'(Commodity)',
            r'(Description)',
        ]
        
        # Add markdown-style headers to section titles
        for pattern in section_patterns:
            text = re.sub(
                pattern,
                r'\n## \1\n',
                text,
                flags=re.IGNORECASE
            )
        
        # Clean up excessive whitespace while preserving structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def _chunk_text_with_title_grouping(self, text: str, filename: str) -> List[Chunk]:
        """
        Chunk text using title-based grouping - combines related sections together.
        This creates more meaningful chunks by keeping related information together.
        """
        # Split text into sections based on headers
        sections = re.split(r'(\n## [^\n]+\n)', text)
        
        # Group related sections
        grouped_sections = []
        current_group = ""
        current_group_name = None
        
        for i, section in enumerate(sections):
            if section.strip().startswith('## '):
                # This is a header
                section_name = section.strip().replace('## ', '').strip()
                
                # Check if this section should be grouped with previous
                should_group = False
                for group_name, group_sections in self.section_groups.items():
                    if any(gs.lower() in section_name.lower() for gs in group_sections):
                        if current_group_name == group_name:
                            should_group = True
                        else:
                            # Start new group
                            if current_group:
                                grouped_sections.append(current_group)
                            current_group = section
                            current_group_name = group_name
                        break
                
                if should_group:
                    current_group += section
                elif not current_group_name:
                    # No group match, treat as standalone
                    if current_group:
                        grouped_sections.append(current_group)
                    current_group = section
                    current_group_name = None
            else:
                # This is content
                current_group += section
        
        # Add final group
        if current_group:
            grouped_sections.append(current_group)
        
        # Now chunk each grouped section
        chunks = []
        chunk_id = 0
        
        for group in grouped_sections:
            if not group.strip():
                continue
                
            # Extract section name for metadata
            section_match = re.search(r'## ([^\n]+)', group)
            section_name = section_match.group(1) if section_match else "General"
            
            # Split large groups if needed
            group_chunks = self.text_splitter.split_text(group)
            
            for chunk_text in group_chunks:
                cleaned_chunk = self._clean_chunk(chunk_text)
                if cleaned_chunk:
                    metadata = DocumentMetadata(
                        filename=filename,
                        chunk_id=chunk_id,
                        source=f"{filename} - {section_name}",
                        chunk_type="text"
                    )
                    chunks.append(Chunk(text=cleaned_chunk, metadata=metadata))
                    chunk_id += 1
        
        return chunks

    def _extract_section_name(self, chunk_text: str) -> str:
        """Extract the section name from a chunk."""
        match = re.search(r'## ([^\n]+)', chunk_text)
        if match:
            return match.group(1).strip()
        return "General"

    def _clean_chunk(self, chunk_text: str) -> str:
        """Clean chunk text while preserving important structure."""
        # Remove excessive whitespace
        chunk_text = re.sub(r'\n{3,}', '\n\n', chunk_text)
        chunk_text = re.sub(r' {2,}', ' ', chunk_text)
        
        # Remove standalone page markers (but keep them if they have content)
        chunk_text = re.sub(r'^### Page \d+\s*$', '', chunk_text, flags=re.MULTILINE)
        
        return chunk_text.strip()

ingestion_service = DocumentIngestionService()
