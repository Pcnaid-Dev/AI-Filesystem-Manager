
Comprehensive Instruction for Building an AI-Powered File Organizer

You are to build a Windows-based AI File Organizer with the following features and constraints:
	1.	Local/offline operation with optional GPU acceleration.
	2.	Robust, modular pipeline that can handle images, videos, audio, and documents.
	3.	Duplicate detection for any file type (exact + near-duplicate checks).
	4.	Semantic & fuzzy search using a vector database and textual search index.
	5.	Automated tagging, classification, and summarization via local LLMs and computer vision models.
	6.	Optional chat interface for natural language queries (“Find photos of John’s birthday 2022,” “Summarize that PDF,” etc.).

1. Directory Monitoring & Workflow
	•	Directory Watcher: Listen to changes in specified folders (local or cloud-sync) via a Python library like watchdog or scheduled tasks.
	•	Indexing/Analysis Pipeline:
	1.	Check for duplicates (general or near-duplicate images) using:
	•	SHA-256 or MD5 (via Python hashlib) for exact duplicates.
	•	imagededup or a perceptual hashing approach for near-duplicate images.
	2.	Metadata extraction with:
	•	Apache Tika (for documents)
	•	ExifTool (for EXIF, IPTC, XMP in images/videos)
	•	FFprobe (for container-level video/audio metadata)
	3.	OCR (if scanned docs or images containing text):
	•	Tesseract or
	•	PaddleOCR
	4.	Visual AI (object detection, scene tagging, face recognition):
	•	YOLOv8 for objects in images/video frames.
	•	InsightFace for face detection, clustering, recognition.
	•	OpenCLIP to generate image embeddings for textual description or scene similarity.
	5.	Text-based embeddings:
	•	Sentence Transformers for semantic text embeddings.
	6.	Content classification / summarization:
	•	LLM calls to local models, e.g.:
	•	Mistral-7B, or
	•	Phi-2 (2.7B), or
	•	TinyLlama-1.1B, or
	•	FLAN-T5
	•	Storage:
	•	ElasticSearch or OpenSearch for keyword/metadata indexing.
	•	Faiss or Chroma for vector embedding storage & similarity search.

2. Detailed Requirements

2.1 Core Functionality
	1.	File Organization & Sorting
	•	Multi-directory, multi-category structure.
	•	Custom sorting rules (by file type, metadata, user-defined criteria).
	•	Automatic/manual tag assignment & smart classification.
	2.	Indexing & Analysis
	•	Metadata extraction (EXIF, doc properties).
	•	OCR for scanned PDFs/images.
	•	Large file/folder classification with context-aware analysis.
	3.	Search & Retrieval
	•	ElasticSearch-based indexing with fuzzy/semantic queries.
	•	Persistent vector DB for advanced context-based search.
	•	Filtering by assigned tags (face, object, event tags, etc.).
	4.	Scheduling & Automation
	•	Automated scanning triggered by file changes or set schedules.
	•	Predefined rules for repeated file patterns.
	•	Caching results to speed up repeated operations.
	5.	Knowledge Base & Chat Integration
	•	User-editable notes/instructions.
	•	Chat UI for searching, summarizing, reorganizing.
	•	System-provided suggestions for knowledge base updates.

2.2 Advanced Capabilities
	1.	Metadata & Tag Intelligence
	•	Automatic or user-refined tag assignment.
	•	Summarize text documents (LLM-based).
	2.	Image & Video Analysis
	•	Object detection, scene detection, face grouping/recognition.
	•	Tagging for recognized objects, actions, and scenes (e.g., “office,” “presentation”).
	3.	Context-Aware File Optimization
	•	Evaluate file importance by usage, size, or age.
	•	Suggest or auto-perform reorganizations, cleanups.
	4.	Multi-Format Support
	•	Documents, images, audio, video, archives, raw images, e-books, etc.
	•	Integrations with MuPDF, LibScan, LibRaw, Libarchive, Antiword, etc., as needed.
	5.	Pluggable AI Models
	•	Modular approach so we can swap OCR engines, summarization models, face recognition backends in the future.

2.3 Newly Added Features
	1.	Lightweight Mobile/Web Companion
	•	Minimal UI accessible remotely or on mobile.
	•	Real-time sync, optional push notifications.
	2.	Cloud/Network Drive Support
	•	Index and organize remote directories (OneDrive, Google Drive, network shares).
	•	Handle offline caching, partial syncing.
	3.	Facial Recognition & Object Tagging
	•	Automatically cluster similar faces, identify recognized individuals.
	•	Tag objects/actions in images or video frames.
	•	UI for manual overrides, re-labeling, or tag removal.

2.4 Technical Architecture
	1.	Scanner/Watcher
	•	Monitors directories, triggers indexing.
	2.	Indexer & Classification Engine
	•	Gathers metadata, extracts text/embeddings, calls YOLOv8 for object detection, InsightFace for faces, etc.
	•	Updates both the search indexes and the vector database.
	3.	Search Engine
	•	ElasticSearch for traditional text retrieval.
	•	Faiss/Chroma for vector-based semantic or image similarity.
	•	Combined or “hybrid” queries for robust fuzzy searching.
	4.	Chat/Orchestration Layer
	•	LLM-based conversation.
	•	Interprets user requests, queries indexes, and returns relevant results or suggested actions.
	5.	AI/ML Integration
	•	Pluggable pipeline design.
	•	Config UI to swap or upgrade models easily.
	6.	User Interface
	•	Dashboard for tasks, duplicates, reorganizations.
	•	Explorer & Search Panel with advanced filtering (faces, objects).
	•	Chat UI for summarization, queries, reorganization proposals.
	•	Lightweight companion for web/mobile.

2.5 Implementation Phases
	1.	Phase 1 – Basic Indexing & Metadata Extraction (Tika, ExifTool, minimal UI)
	2.	Phase 2 – Classification & Tagging (LLM-based doc classification, YOLOv8 for images, basic face detection)
	3.	Phase 3 – Advanced Search & Summarization (Vector search, OCR improvements, LLM summarizer)
	4.	Phase 4 – Scheduling & Automation (Watcher, triggers, caching, advanced rules)
	5.	Phase 5 – Chat & Knowledge Base (In-app conversation, user-editable KB)
	6.	Phase 6 – Additional File Formats & Extended Libraries (archives, e-books, raw images)
	7.	Phase 7 – Pluggable AI/ML Models & Cloud Drive Support (registry-like approach to AI components, remote indexing)
	8.	Phase 8 – Face Grouping, Facial Recognition & Media Tagging (InsightFace, YOLOv8-based tagging, UI overrides)
	9.	Phase 9 – Lightweight Mobile/Web Companion (sync, push notifications, quick UI)
	10.	Phase 10 – Optimization & Maintenance (tuning, security, logs, user access controls)

⸻

3. Models & Tools URLs

Below are the direct references for each major component:
	•	Duplicate Check:
	•	Hashing (builtin hashlib) → https://docs.python.org/3/library/hashlib.html
	•	dupeGuru → https://github.com/arsenetar/dupeguru
	•	imagededup → https://github.com/idealo/imagededup
	•	Apache Tika → https://tika.apache.org/
	•	ExifTool → https://exiftool.org/
	•	FFmpeg / FFprobe → https://ffmpeg.org/
	•	Tesseract → https://github.com/tesseract-ocr/tesseract
	•	PaddleOCR → https://github.com/PaddlePaddle/PaddleOCR
	•	YOLOv8 → https://github.com/ultralytics/ultralytics
	•	InsightFace → https://github.com/deepinsight/insightface
	•	Faiss → https://github.com/facebookresearch/faiss
	•	Chroma → https://github.com/chroma-core/chroma
	•	ElasticSearch → https://www.elastic.co/elasticsearch/
	•	OpenCLIP → https://github.com/mlfoundations/open_clip
	•	Sentence Transformers
	•	MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
	•	LLMs (Mistral, Phi-2, TinyLlama, FLAN-T5)
	•	Mistral-7B → https://huggingface.co/mistralai/Mistral-7B
	•	Phi-2 → https://huggingface.co/microsoft/phi-2
	•	TinyLlama-1.1B → https://github.com/jzhang38/TinyLlama
	•	FLAN-T5 → https://huggingface.co/google/flan-t5-large
	•	watchdog → https://pypi.org/project/watchdog/
	•	Gradio → https://gradio.app/
	•	Haystack → https://haystack.deepset.ai/

⸻

4. Environment & Dependencies

4.1 Python Libraries (PyPI)
	1.	Python 3.9+ (official python.org)
	2.	torch / torchvision (install with CUDA if using GPU acceleration)
	3.	transformers (Hugging Face)
	4.	sentence_transformers
	5.	faiss-gpu or faiss-cpu
	6.	chroma (for a local vector store)
	7.	elasticsearch or opensearch-py (for search indexing)
	8.	pytesseract or paddleocr (OCR)
	9.	watchdog (file monitoring)
	10.	gradio (web-based UI) or pyqt5 / pyside2 (desktop GUI)
	11.	requests, fastapi, uvicorn (if building an API server)
	12.	ultralytics (for YOLOv8)
	13.	insightface (face recognition)

4.2 External Dependencies
	•	Tesseract (native C++) → install from official releases or package manager.
	•	FFmpeg → https://www.gyan.dev/ffmpeg/builds/
	•	Java (JRE) for Tika server usage (optional).
	•	Perl for ExifTool (if not using a compiled binary).

⸻

5. Final Instructions to Implement

1. Set Up Environment
	•	Create a Python 3.9+ virtual environment.
	•	Install libraries (PyTorch w/ CUDA, transformers, watchers, etc.).
	•	Install Tesseract, FFmpeg, Java, and ExifTool as needed.

2. Build the Pipeline
	•	File Scanner: Use watchdog or scheduling.
	•	Indexing: On new/changed file → gather metadata with ExifTool/Tika → check duplicates → run OCR if needed → run YOLOv8/CLIP if it’s an image or video.
	•	Embeddings: Generate embeddings (text or image) → store in Faiss/Chroma.
	•	Tagging: Create or update tags from classification results, face recognition, object detection.
	•	Search: Insert textual metadata into ElasticSearch for keyword queries; also store embeddings for semantic/fuzzy lookups.

3. GUI / Interaction
	•	Provide a Dashboard (duplicates found, large files, recommended reorganizations).
	•	Provide an Explorer with filtering by tags, file type, date, recognized faces.
	•	Provide a Chat interface that uses an LLM to handle advanced queries or reorganization instructions.

4. Testing & Optimization
	•	Confirm each model (OCR, YOLOv8, face recognition) works offline.
	•	Optimize concurrency or batch processing for large directories.
	•	Evaluate GPU memory usage (choose smaller models if needed).

5. Deployment
	•	Optionally bundle everything with a Windows installer (e.g., PyInstaller or a .NET shell).
	•	Provide a minimal web UI with Gradio for remote or mobile access.
