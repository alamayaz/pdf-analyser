# Intelligent Document Analyzer - Docker Version
# Processes PDFs from input folder and outputs intelligent analysis JSON

import os
import json
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import statistics
import random
import time
from collections import Counter
import string


class IntelligentDocumentAnalyzer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the analyzer with local LLM"""
        print("🤖 Loading TinyLlama model...")

        # Set cache directory for Docker
        cache_dir = "/app/.cache" if os.path.exists("/app") else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu", cache_dir=cache_dir
        )
        print("✅ Model loaded successfully")

        # Will be populated based on persona and task
        self.persona_keywords = []
        self.task_keywords = []
        self.domain_keywords = []

    def extract_keywords_from_text(self, text: str, top_n: int = 15) -> List[str]:
        """Extract important keywords from text using TF-IDF-like approach"""
        # Tokenize and clean
        words = re.findall(r"\b[a-z]+\b", text.lower())

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "them",
            "their",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "just",
            "now",
        }

        # Filter words
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]

        # Count frequencies
        word_freq = Counter(meaningful_words)

        # Get top keywords
        top_keywords = [word for word, _ in word_freq.most_common(top_n)]

        return top_keywords

    def analyze_persona_and_task(self, persona: str, task: str) -> None:
        """Analyze persona and task to extract relevant keywords"""
        print("🧠 Analyzing persona and task requirements...")

        # Extract keywords from persona
        persona_words = self.extract_keywords_from_text(persona, top_n=10)

        # Extract keywords from task
        task_words = self.extract_keywords_from_text(task, top_n=15)

        # Use LLM to expand keywords based on context
        expanded_keywords = self.expand_keywords_with_llm(persona, task)

        self.persona_keywords = persona_words
        self.task_keywords = task_words
        self.domain_keywords = expanded_keywords

        print(f"  📌 Persona keywords: {', '.join(self.persona_keywords[:5])}...")
        print(f"  📌 Task keywords: {', '.join(self.task_keywords[:5])}...")
        print(f"  📌 Domain keywords: {', '.join(self.domain_keywords[:5])}...")

    def expand_keywords_with_llm(self, persona: str, task: str) -> List[str]:
        """Use LLM to suggest relevant keywords for the persona/task"""
        try:
            prompt = f"""<|system|>
You are a keyword extraction expert. Generate relevant keywords for document analysis.

<|user|>
Persona: {persona}
Task: {task}

List 10 important keywords that would help identify relevant content for this persona and task.
Output only the keywords separated by commas:

<|assistant|>"""

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            # Extract keywords from response
            keywords = [k.strip().lower() for k in generated_text.split(",")]
            # Filter out empty or very short keywords
            keywords = [k for k in keywords if len(k) > 3][:10]

            return keywords

        except Exception as e:
            print(f"  ⚠️ Could not expand keywords with LLM: {e}")
            return []

    def calculate_relevance_score(
        self, content: str, title: str, filename: str
    ) -> float:
        """Calculate relevance score based on dynamic keywords"""
        content_lower = content.lower()
        title_lower = title.lower()

        score = 5.0  # Base score

        # Count matches for each keyword set
        persona_matches = sum(1 for kw in self.persona_keywords if kw in content_lower)
        task_matches = sum(1 for kw in self.task_keywords if kw in content_lower)
        domain_matches = sum(1 for kw in self.domain_keywords if kw in content_lower)

        # Weight the matches
        score += min(persona_matches * 0.3, 2.0)  # Max +2
        score += min(task_matches * 0.5, 3.0)  # Max +3
        score += min(domain_matches * 0.3, 2.0)  # Max +2

        # Title relevance
        title_keywords = self.persona_keywords + self.task_keywords
        title_matches = sum(1 for kw in title_keywords if kw in title_lower)
        score += min(title_matches * 0.5, 1.0)  # Max +1

        # Add small random variation
        score += random.uniform(-0.2, 0.2)

        return min(max(score, 1), 10)

    def extract_title_from_pdf(self, doc: fitz.Document) -> str:
        """Extract title from PDF metadata or first page using font analysis"""
        if len(doc) == 0:
            return "Untitled Document"

        first_page = doc[0]
        blocks = first_page.get_text("dict")["blocks"]

        # Get text lines and find best title candidate
        text_lines = first_page.get_text().split("\n")
        for line in text_lines[:10]:
            line = line.strip()
            if 5 <= len(line) <= 150 and not line.isdigit():
                # Skip obvious non-titles
                skip_patterns = [
                    line.lower().startswith(("page ", "p.", "fig", "table")),
                    "@" in line,
                    line.count(".") > 3,
                ]
                if not any(skip_patterns):
                    return line

        return "Untitled Document"

    def extract_document_sections(
        self, pdf_path: str, max_sections_per_doc: int = 10
    ) -> Dict[str, Any]:
        """Extract structured content from PDF with sections"""
        try:
            doc = fitz.open(pdf_path)
            filename = os.path.basename(pdf_path)

            # Extract title
            title = self.extract_title_from_pdf(doc)

            # Extract sections with content
            sections = []
            current_section = None

            # Sample pages for speed
            total_pages = min(len(doc), 50)
            sample_rate = max(1, total_pages // 20)  # Process ~20 pages max

            for page_num in range(0, total_pages, sample_rate):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" in block:
                        # Extract text and formatting
                        block_text = ""
                        max_font_size = 0
                        is_bold = False

                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                span_text = span["text"].strip()
                                if span_text:
                                    line_text += span_text + " "
                                    max_font_size = max(max_font_size, span["size"])
                                    if span["flags"] & 16:  # Bold flag
                                        is_bold = True

                            if line_text.strip():
                                block_text += line_text.strip() + " "

                        block_text = block_text.strip()

                        if len(block_text) < 10:  # Skip very short blocks
                            continue

                        # Determine if this is a section header
                        is_header = self.is_section_header(
                            block_text, max_font_size, is_bold
                        )

                        if is_header:
                            # Save previous section if exists
                            if current_section and len(current_section["content"]) > 50:
                                sections.append(current_section)
                                if len(sections) >= max_sections_per_doc:
                                    break

                            # Start new section
                            current_section = {
                                "title": block_text[:200],
                                "content": "",
                                "page": page_num + 1,
                                "font_size": max_font_size,
                            }
                        else:
                            # Add to current section content
                            if current_section:
                                current_section["content"] += block_text + " "
                            else:
                                # Create default section if no header found
                                current_section = {
                                    "title": f"Content from page {page_num + 1}",
                                    "content": block_text + " ",
                                    "page": page_num + 1,
                                    "font_size": max_font_size,
                                }

                if len(sections) >= max_sections_per_doc:
                    break

            # Add final section
            if (
                current_section
                and len(current_section["content"]) > 50
                and len(sections) < max_sections_per_doc
            ):
                sections.append(current_section)

            doc.close()

            # Score sections based on dynamic keywords
            for section in sections:
                section["relevance_score"] = self.calculate_relevance_score(
                    section["content"], section["title"], filename
                )

            return {
                "filename": filename,
                "title": title,
                "sections": sections,
            }

        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                "filename": os.path.basename(pdf_path),
                "title": f"Error processing {os.path.basename(pdf_path)}",
                "sections": [],
            }

    def is_section_header(self, text: str, font_size: float, is_bold: bool) -> bool:
        """Determine if text block is likely a section header"""
        word_count = len(text.split())

        if word_count > 15 or word_count < 1:
            return False

        # Check formatting indicators
        format_score = 0
        if is_bold:
            format_score += 2
        if font_size > 12:
            format_score += 1
        if text.endswith(":"):
            format_score += 2
        if text.isupper() or text.istitle():
            format_score += 1

        # Content indicators
        content_score = 0
        if word_count <= 8:
            content_score += 2
        if not text.startswith(("the ", "this ", "it ", "there ")):
            content_score += 1

        return (format_score + content_score) >= 3

    def get_llm_relevance_score(
        self, section_content: str, persona: str, job_to_be_done: str, doc_name: str
    ) -> float:
        """Get relevance score from LLM for top candidates"""
        try:
            content_preview = (
                section_content[:600] + "..."
                if len(section_content) > 600
                else section_content
            )

            prompt = f"""<|system|>
You are an expert document analyst.

<|user|>
Persona: {persona}
Task: {job_to_be_done}
Document: {doc_name}

Content:
{content_preview}

Rate how relevant this content is for the persona's task (1-10):

<|assistant|>"""

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=800,
                truncation=True,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            numbers = re.findall(r"\d+", generated_text)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 1), 10)

            return 6.0

        except Exception as e:
            print(f"Error in LLM scoring: {e}")
            return 6.0

    def refine_section_text(
        self, content: str, persona: str, job_to_be_done: str
    ) -> str:
        """Extract most relevant sentences based on dynamic keywords"""
        sentences = content.split(". ")
        relevant_sentences = []

        # Use all our keyword sets
        all_keywords = self.persona_keywords + self.task_keywords + self.domain_keywords

        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in all_keywords if keyword in sentence_lower)
            sentence_scores.append((sentence.strip(), score))

        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in sentence_scores[:5] if score > 0]

        if top_sentences:
            refined = ". ".join(top_sentences) + "."
            return refined[:500] + "..." if len(refined) > 500 else refined

        # Fallback to original content
        return content[:500] + "..." if len(content) > 500 else content

    def analyze_documents(
        self, input_data: Dict[str, Any], input_folder: str
    ) -> Dict[str, Any]:
        """Main analysis function - Universal version"""
        start_time = time.time()
        print("🔍 Starting universal document analysis...")

        # Extract input parameters
        documents = input_data["documents"]
        persona = input_data["persona"]["role"]
        job_to_be_done = input_data["job_to_be_done"]["task"]

        print(f"👤 Persona: {persona}")
        print(f"🎯 Task: {job_to_be_done}")
        print(f"📚 Documents: {len(documents)}")

        # Analyze persona and task to build dynamic keywords
        self.analyze_persona_and_task(persona, job_to_be_done)

        # Process each document
        all_sections = []
        processed_docs = []

        # Phase 1: Extract sections from all documents
        for doc_info in documents:
            filename = doc_info["filename"]
            pdf_path = os.path.join(input_folder, filename)

            if os.path.exists(pdf_path):
                print(f"\n📄 Processing: {filename}")

                # Extract with limited sections per doc
                doc_data = self.extract_document_sections(
                    pdf_path, max_sections_per_doc=10
                )
                processed_docs.append(filename)

                # Add scored sections to our list
                for section in doc_data["sections"]:
                    all_sections.append(
                        {
                            "document": filename,
                            "section_title": section["title"],
                            "content": section["content"],
                            "page_number": section["page"],
                            "relevance_score": section["relevance_score"],
                        }
                    )

                print(f"  ✅ Extracted {len(doc_data['sections'])} sections")
            else:
                print(f"  ❌ File not found: {filename}")

        # Phase 2: Sort by keyword scores and get top candidates
        all_sections.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Take top candidates for LLM scoring
        top_candidates = all_sections[:15]

        print(f"\n🎯 Using LLM to verify top {len(top_candidates)} candidates...")

        # Phase 3: Use LLM only on top candidates
        for i, section in enumerate(top_candidates):
            print(
                f"  📊 LLM verification [{i + 1}/{len(top_candidates)}]: {section['section_title'][:40]}...",
                end="",
            )

            llm_score = self.get_llm_relevance_score(
                section["content"], persona, job_to_be_done, section["document"]
            )

            # Combine keyword score and LLM score
            section["relevance_score"] = (
                section["relevance_score"] * 0.4 + llm_score * 0.6
            )
            print(f" Final: {section['relevance_score']:.1f}")

        # Re-sort with final scores
        top_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Select top 5 with diversity
        top_sections = []
        used_docs = set()

        # First pass: best from each document
        for section in top_candidates:
            if section["document"] not in used_docs and len(top_sections) < 5:
                top_sections.append(section)
                used_docs.add(section["document"])

        # Second pass: fill remaining
        for section in top_candidates:
            if len(top_sections) >= 5:
                break
            if section not in top_sections:
                top_sections.append(section)

        print(f"\n🏆 Selected top {len(top_sections)} sections:")
        for i, section in enumerate(top_sections):
            print(
                f"  {i + 1}. [{section['relevance_score']:.1f}] {section['document']}: {section['section_title'][:50]}"
            )

        # Phase 4: Create output
        extracted_sections = []
        for i, section in enumerate(top_sections):
            extracted_sections.append(
                {
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "importance_rank": i + 1,
                    "page_number": section["page_number"],
                }
            )

        # Text refinement
        print("\n✨ Refining text for selected sections...")
        subsection_analysis = []
        for section in top_sections:
            refined_text = self.refine_section_text(
                section["content"], persona, job_to_be_done
            )

            subsection_analysis.append(
                {
                    "document": section["document"],
                    "refined_text": refined_text,
                    "page_number": section["page_number"],
                }
            )

        # Create final output
        output = {
            "metadata": {
                "input_documents": processed_docs,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat(),
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis,
        }

        elapsed_time = time.time() - start_time
        print(f"\n🎉 Analysis complete in {elapsed_time:.1f} seconds!")

        return output


def main():
    """Main function for processing input JSON and generating output"""
    # Docker paths
    input_folder = "/app/input"
    output_folder = "/app/output"

    print("🚀 Intelligent Document Analyzer - Docker Edition")
    print("=" * 55)

    # Check for input JSON file
    input_json_path = None

    # First try specific filenames for backward compatibility
    for filename in ["challenge1b_input.json", "input.json"]:
        test_path = os.path.join(input_folder, filename)
        if os.path.exists(test_path):
            input_json_path = test_path
            break

    # If not found, look for any .json file in the input folder
    if input_json_path is None:
        json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
        if json_files:
            # Use the first JSON file found
            input_json_path = os.path.join(input_folder, json_files[0])
            print(f"📄 Found JSON file: {json_files[0]}")
        else:
            print(f"❌ No JSON files found in input folder: {input_folder}")
            print("Please ensure at least one .json file is in the input folder")
            return

    # Load input configuration
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        print(f"✅ Loaded input configuration")
    except Exception as e:
        print(f"❌ Error loading input JSON: {e}")
        return

    # Initialize analyzer
    try:
        analyzer = IntelligentDocumentAnalyzer()
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return

    # Process documents
    try:
        result = analyzer.analyze_documents(input_data, input_folder)

        # Save output
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "output.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Output saved to: {output_path}")
        print(f"📊 Processed {len(result['metadata']['input_documents'])} documents")
        print(f"🔍 Found {len(result['extracted_sections'])} relevant sections")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
