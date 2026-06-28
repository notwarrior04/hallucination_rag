# EHD-RAG v1.0 Architecture Specification (Frozen)

> Part 1 of 5

## Version
- Project: Explainable Hybrid Document RAG (EHD-RAG)
- Architecture Version: 1.0 (Frozen)

# 1. Vision

EHD-RAG is an explainable Retrieval-Augmented Generation framework that answers user questions from uploaded evidence documents and then performs post-hoc verification to classify every statement as:
- Supported by uploaded evidence
- Synthesized from evidence
- External knowledge
- Unsupported / Hallucinated

# 2. Research Objective

1. Upload one or more evidence documents.
2. Parse, clean, chunk, embed and index them.
3. Retrieve evidence.
4. Generate an answer using an LLM.
5. Verify every factual claim.
6. Render the answer with inline provenance highlighting.

# 3. Architecture Freeze Policy

This document freezes the public architecture.

Internal implementations may improve, but public interfaces remain stable.

# 4. Layered Architecture

User
↓
Evidence Ingestion
↓
Knowledge Base
↓
Retrieval
↓
Generation
↓
Verification
↓
Rendering

# 5. High-Level Pipeline

Upload Documents
↓
DocumentLoader
↓
Parser Registry
  - PDFParser
  - DOCXParser
  - TXTParser
  - Future Parsers
↓
Document
↓
Cleaner
↓
Chunker
↓
KnowledgeBase
↓
Embedding Generator
↓
Embedding Cache
↓
Vector Store
↓
Retriever
↓
Prompt Builder
↓
LLM Interface
↓
Answer
↓
Verification Layer
  - Sentence Splitter
  - Claim Extractor
  - Evidence Mapper
  - NLI Verifier
  - Confidence Engine
  - Provenance Classifier
↓
Renderer
↓
Highlighted Answer

# 6. Folder Structure

```text
src/
├── document/
├── retrieval/
├── generation/
├── verification/
├── rendering/
├── pipelines/
├── configs/
└── utils/
```

# 7. Parser Independence

Parsers are registered through a Parser Registry.

Adding DOCX, TXT or PPTX support should require:
1. Create the parser.
2. Register it.

No other module should change.

# 8. LLM Independence

RAGGenerator
→ LLMInterface
→ Provider
  - Ollama
  - Gemini
  - OpenAI

Changing providers must not affect any other module.

# 9. Core Principles

- Single Responsibility Principle
- Open/Closed Principle
- Dependency Inversion
- Interface Stability
- Provider Independence
- Parser Independence
- Explainability First
- Multi-document Native

# 10. Next Part

Part 2 defines all frozen data models:
- Document
- Page
- Chunk
- KnowledgeBase
- Answer
- Claim
- VerificationResult
- HighlightSegment

# EHD-RAG v1.0 Architecture Specification (Frozen)

> **Part 2 of 5 — Core Data Model Specification**

---

# 1. Purpose

This document defines every core data model used throughout the framework.
These models are **frozen interfaces**. Internal implementations may evolve,
but these public structures should remain stable.

---

# 2. Data Flow

```text
Uploaded File
      │
      ▼
Document
      │
      ▼
Pages
      │
      ▼
Chunks
      │
      ▼
KnowledgeBase
      │
      ▼
Retriever
      │
      ▼
Answer
      │
      ▼
Verification
      │
      ▼
Renderer
```

---

# 3. Document Model

## Responsibility

Represents one uploaded evidence file.

Examples:

- report.pdf
- research.docx
- meeting_notes.txt

A `Document` is **never** responsible for retrieval,
verification, rendering or generation.

### Required Fields

| Field | Description |
|--------|-------------|
| document_id | Unique identifier |
| filename | Original filename |
| filetype | pdf/docx/txt/etc |
| raw_text | Complete extracted text |
| total_pages | Number of pages |
| pages | List of Page objects |
| chunks | List of Chunk objects |
| metadata | Parser metadata |

### Does

- Own extracted text
- Own pages
- Own chunks
- Store metadata

### Does NOT

- Generate embeddings
- Search
- Call LLM
- Verify claims

---

# 4. Page Model

Represents one logical page from a document.

## Required Fields

| Field | Description |
|--------|-------------|
| page_number | Physical page index |
| text | Extracted page text |
| metadata | Page metadata |

Pages are immutable after parsing except for cleaning.

---

# 5. Chunk Model

The Chunk is the smallest searchable evidence unit.

Every retrieval operation returns Chunks.

## Required Fields

| Field | Description |
|--------|-------------|
| chunk_id | Unique chunk identifier |
| document_id | Parent document |
| filename | Human-readable filename |
| page_number | Source page |
| paragraph_number | Paragraph index |
| section | Optional heading |
| text | Chunk text |
| embedding | Dense vector |
| metadata | Additional information |

### Design Rules

A Chunk must always know where it came from.

The renderer must never need to search backwards to determine provenance.

---

# 6. KnowledgeBase

Represents all uploaded evidence in one session.

## Responsibilities

- Store uploaded documents
- Provide all searchable chunks
- Produce a virtual merged document for indexing

## Public Interface

- add_document()
- remove_document()
- get_document()
- list_documents()
- total_documents()
- total_chunks()
- all_chunks()
- clear()
- to_document()

### Extension Rule

Adding additional document types must not modify KnowledgeBase.

---

# 7. Answer Model

Answer is the central object of the second half of the framework.

Every downstream module enriches this object.

## Required Fields

| Field | Description |
|--------|-------------|
| query | Original user question |
| prompt | Prompt sent to LLM |
| text | Generated answer |
| retrieved_chunks | Supporting evidence |
| claims | Extracted claims |
| confidence | Overall confidence |
| retrieval_time | Retrieval latency |
| generation_time | Generation latency |
| verification_time | Verification latency |
| metadata | Miscellaneous metadata |

### Important Rule

Modules modify the same Answer object.

Modules never create replacement Answer objects.

---

# 8. Future Models

These models are intentionally reserved for the verification stage.

## Claim

Represents one atomic factual statement.

Expected fields:

- claim_id
- text
- sentence_index
- evidence
- verification_result

---

## VerificationResult

Stores:

- entailment score
- contradiction score
- confidence
- evidence links

---

## HighlightSegment

Stores rendering information.

Fields:

- start_index
- end_index
- category
- supporting_evidence
- color

---

# 9. Architecture Decision Record (ADR-001)

## Decision

Chunks permanently store provenance.

## Reason

The renderer and verifier require direct access to:

- filename
- page number
- document id

without performing reverse lookups.

---

# 10. Architecture Decision Record (ADR-002)

## Decision

Answer is the only mutable object after retrieval.

## Reason

Generation, verification and rendering become independent modules that enrich
one shared representation instead of creating incompatible outputs.

---

# 11. Frozen Rules

The following models are frozen for Version 1.0:

- Document
- Page
- Chunk
- KnowledgeBase
- Answer

Only additive fields are allowed in future revisions.

---

# Next Part

Part 3 specifies every project module including:

- document/
- retrieval/
- generation/
- verification/
- rendering/
- pipelines/

Each module will define:

- Responsibilities
- Inputs
- Outputs
- Dependencies
- Extension points
- Forbidden responsibilities

# EHD-RAG v1.0 Architecture Specification (Frozen)

> **Part 3 of 5 — Module Specifications and Frozen Interfaces**

# 1. Module Dependency Rules

```
document
   ↓
retrieval
   ↓
generation
   ↓
verification
   ↓
rendering
```

Rules:
- Lower layers never import higher layers.
- Verification never calls retrieval directly.
- Rendering never performs verification.
- Circular imports are prohibited.

---

# 2. document/

## Responsibility
Own the complete document ingestion lifecycle.

### Contains
- document.py
- knowledge_base.py
- base_parser.py
- parser_registry.py
- document_loader.py
- pdf_parser.py
- cleaner.py
- chunker.py

### Does
- Parse uploaded files
- Normalize extracted content
- Produce Document objects

### Does NOT
- Generate embeddings
- Retrieve information
- Call LLMs
- Verify answers

### Extension Point
Adding DOCX/TXT/PPTX requires:
1. Create a parser implementing BaseParser.
2. Register it in ParserRegistry.
No other module changes.

---

# 3. Parser Registry (Frozen)

Purpose:
Resolve parser by file extension.

Public Interface

```python
register(extension: str, parser_cls)

get_parser(extension: str)
```

This interface is frozen.

---

# 4. retrieval/

Files

- embedding_generator.py
- embedding_cache.py
- vector_store.py
- retriever.py

Responsibilities

- Create document embeddings
- Cache embeddings
- Maintain FAISS index
- Return relevant chunks

Public Interface

```python
Retriever.retrieve(
    query: str,
    top_k: int = 5
) -> List[Chunk]
```

Frozen.

Internal retrieval strategy may change without affecting callers.

Future upgrades:
- BM25
- Hybrid Retrieval
- RRF
- Cross-Encoder

---

# 5. generation/

Files

- answer.py
- prompt_builder.py
- llm_interface.py
- rag_generator.py
- providers/

Responsibilities

- Build prompts
- Invoke selected LLM
- Produce Answer objects

Does NOT
- Verify claims
- Highlight evidence
- Score confidence

---

# 6. Provider Architecture

```
LLMInterface
      │
      ▼
BaseProvider
      │
 ┌────┼─────┐
 ▼    ▼     ▼
Ollama Gemini OpenAI
```

Required provider interface

```python
generate(answer: Answer) -> Answer
```

Every provider must implement this contract.

---

# 7. Prompt Builder

Input:
- Query
- Retrieved Chunks

Output:
- Prompt string

Rules:
- Evidence-first prompting
- No hidden state
- Deterministic formatting

---

# 8. RAG Generator

Single public entry point for answer generation.

```python
generate(
    query: str,
    top_k: int = 5
) -> Answer
```

Workflow:
1. Retrieve chunks.
2. Build prompt.
3. Call provider.
4. Return populated Answer.

---

# 9. verification/

Reserved files

- sentence_splitter.py
- claim_extractor.py
- evidence_mapper.py
- nli_verifier.py
- confidence_engine.py
- provenance_classifier.py

Responsibilities

- Analyse generated answer
- Verify claims
- Compute confidence
- Classify provenance

Must never:
- Retrieve documents
- Regenerate answers

---

# 10. rendering/

Renderer receives a verified Answer.

Responsibilities:
- Inline highlighting
- Human-readable evidence display
- Export to CLI/Web/UI

Renderer never performs verification.

---

# 11. pipelines/

rag_pipeline.py

Purpose:
Complete ingestion workflow.

Responsibilities:
- Load
- Clean
- Chunk
- Embed
- Cache
- Index

No generation logic.

---

# 12. Frozen Public Interfaces

```
DocumentLoader.load()

ParserRegistry.register()

Retriever.retrieve()

PromptBuilder.build()

LLMInterface.generate()

RAGGenerator.generate()

Renderer.render()
```

These signatures are frozen for Version 1.0.

---

# 13. ADR-003

Decision:
Use provider abstraction for LLMs.

Reason:
Switch models without changing business logic.

---

# 14. ADR-004

Decision:
Use parser registry.

Reason:
Support new document formats without modifying ingestion.

---

# Next Part

Part 4 defines coding standards, dependency rules, extension strategy, testing methodology, configuration, logging and implementation conventions.

# EHD-RAG v1.0 Architecture Specification (Frozen)

> **Part 4 of 5 — Development Standards, Configuration and Extension Strategy**

# 1. Coding Standards

## Language
- Python 3.10+
- Type hints on all public methods.
- Google-style docstrings.
- Black formatting.
- Meaningful variable names.

## Naming

Classes: PascalCase

Functions: snake_case

Constants: UPPER_CASE

Private helpers: _leading_underscore

---

# 2. Dependency Rules

Allowed dependency direction:

document
→ retrieval
→ generation
→ verification
→ rendering

Never import upward.

No circular imports.

---

# 3. Configuration

All configurable values belong under `configs/`.

Examples:

- embedding model
- chunk size
- overlap
- top_k
- ollama model
- cache location

Never hard-code these values inside business logic.

---

# 4. Error Handling

Every public module must raise meaningful exceptions.

Examples:

- UnsupportedFileTypeError
- ParserError
- EmbeddingError
- RetrievalError
- GenerationError
- VerificationError

Never silently ignore failures.

---

# 5. Logging

Future logging should include:

- upload events
- parsing time
- chunk count
- embedding time
- retrieval latency
- generation latency
- verification latency

Use Python's logging module.

Never use print() outside debugging/tests.

---

# 6. Testing Strategy

Every module must have an independent test.

Examples:

Document Layer
✓ PDF parsing
✓ Cleaning
✓ Chunk creation

Retrieval Layer
✓ Embedding generation
✓ Cache hit/miss
✓ Vector search

Generation Layer
✓ Prompt creation
✓ Provider response

Verification Layer
✓ Claim extraction
✓ NLI labels
✓ Confidence scoring

---

# 7. Performance Targets

- Parser should stream large files when practical.
- Embeddings generated once and cached.
- Vector index reusable.
- Retrieval < 1 second for moderate document collections.
- Verification modular for future parallelization.

---

# 8. Caching Strategy

Cache only expensive computations.

Cache:
- embeddings
- vector index (future)

Do not cache:
- generated answers
- verification results

---

# 9. Extension Strategy

## New Parser

1. Implement BaseParser.
2. Register in ParserRegistry.

No other changes.

## New LLM

1. Implement BaseProvider.
2. Register/select provider.

No other changes.

## New Retrieval Method

Modify Retriever internals only.

Public interface remains unchanged.

## New Verification Technique

Add module inside verification/.

No generation changes.

---

# 10. Security

- Never execute uploaded files.
- Treat uploaded documents as untrusted.
- Keep API keys outside source code.
- Validate file extensions.
- Sanitize extracted text before processing.

---

# 11. Git Workflow

Suggested branches:

main
develop
feature/*
bugfix/*
experiment/*

Architecture changes require explicit approval after v1.0.

---

# 12. ADR-005

Decision:
One central Answer object.

Reason:
Allows retrieval, generation, verification and rendering to enrich the same artifact.

---

# 13. ADR-006

Decision:
Parser independence through registry.

Reason:
Adding formats must require minimal modification.

---

# 14. Definition of Done

A module is complete only if:

- code implemented
- documented
- tested
- type hinted
- integrated
- no architecture violation

---

# 15. Implementation Order (Frozen)

1. Document Layer
2. Retrieval Layer
3. Generation Layer
4. Verification Layer
5. Rendering Layer
6. UI

Do not change this order.

---

# Next Part

Part 5 contains the final roadmap, milestone checklist, research contribution, deployment plan and architecture freeze declaration.

# EHD-RAG v1.0 Architecture Specification (Frozen)

> **Part 5 of 5 — Master Roadmap, Research Contribution and Project Constitution**

# 1. Project Status

## Completed
- Core architecture frozen
- Document ingestion framework
- PDF parser
- Cleaner
- Chunker
- KnowledgeBase
- Embedding generation
- Embedding cache
- FAISS vector store
- Dense retriever
- Prompt builder
- Generation framework design

## In Progress
- Ollama provider integration
- End-to-end RAG execution

## Remaining
- Verification pipeline
- Rendering engine
- User interface
- Evaluation experiments

---

# 2. Final Development Roadmap

Phase 1
✓ Document Processing

Phase 2
✓ Retrieval Infrastructure

Phase 3
✓ Generation

Phase 4
Verification
- Sentence Splitter
- Claim Extractor
- Evidence Mapper
- NLI Verifier
- Confidence Engine
- Provenance Classifier

Phase 5
Rendering
- Inline highlighting
- Evidence visualization
- Confidence display

Phase 6
User Interface
- CLI
- Streamlit (optional)

---

# 3. Research Contribution Mapping

Existing RAG provides:
- Retrieval
- Prompt construction
- LLM generation

EHD-RAG contributes:
- Post-hoc verification
- Claim-level evidence mapping
- Confidence estimation
- Provenance classification
- Inline explainability

---

# 4. Colour Semantics

GREEN
- Directly supported by uploaded evidence.

BLUE
- Synthesized from multiple evidence fragments.

YELLOW
- External factual knowledge (when explicitly allowed).

RED
- Unsupported, hallucinated, estimated or unverifiable content.

---

# 5. Evaluation Plan

Evaluate:

- Retrieval quality
- Generation quality
- Hallucination reduction
- Evidence attribution accuracy
- Verification precision
- Overall confidence calibration

Suggested datasets:
- FEVER
- SQuAD v2
- User supplied documents

---

# 6. Deliverables

Codebase

Architecture document

Research report

Experimental results

Presentation

Demonstration

---

# 7. Success Criteria

The framework is considered complete when a user can:

1. Upload one or more evidence documents.
2. Ask a natural-language question.
3. Receive a complete answer.
4. Inspect inline provenance.
5. Identify unsupported statements.
6. Review confidence values.

---

# 8. Future Extensions (Outside Version 1)

- DOCX parser
- TXT parser
- PPTX parser
- OCR support
- Hybrid retrieval
- BM25
- Cross-encoder reranker
- Table extraction
- Multi-modal evidence
- Distributed vector stores

These extensions must preserve all frozen interfaces.

---

# 9. Project Constitution

The following rules are permanent:

1. Parser independence is mandatory.
2. Provider independence is mandatory.
3. Retrieval is independent of verification.
4. Rendering never performs verification.
5. Public interfaces remain stable.
6. Answer is the central mutable object.
7. New functionality should be added through extension rather than modification.
8. Architecture changes require documented justification.

---

# 10. Architecture Freeze Declaration

Architecture Version: 1.0

Status: FROZEN

No further architectural redesign shall be performed unless:
- a critical defect is identified,
- the research objective cannot be achieved,
- or a change is required for correctness.

All future effort is directed toward implementation, experimentation and evaluation.

---

# End of Specification

This concludes the EHD-RAG v1.0 Architecture Specification.

Recommended repository documents:

- ARCHITECTURE.md
- README.md
- CONTRIBUTING.md
- DEVELOPMENT.md
- TESTING.md

