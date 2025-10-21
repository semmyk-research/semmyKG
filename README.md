---
# metadata
title: semmyKG - Knowledge Graph visualiser toolkit (builder from markdown)
emoji: 🕸️
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.44.1
python_version: 3.12
#command: python app_gradio_lightrag.py
app_file: app.py    #app_gradio_lightrag.py
hf_oauth: true
oauth_scopes: [read-access]
hf_oauth_scopes: [inference-api]
license: mit
pinned: true
short_description: semmyKG - Knowledge Graph toolkit |
#models: [meta-llama/Llama-4-Maverick-17B-128E-Instruct, openai/gpt-oss-120b, openai/gpt-oss-20b, ]
models: 
  - meta-llama/Llama-4-Maverick-17B-128E-Instruct
  - openai/gpt-oss-120b, openai/gpt-oss-20b
tags: [knowledge graph, markdown, RAG, domain]
#preload_from_hub: [https://huggingface.co/datalab-to/surya_layout, https://huggingface.co/datalab-to/surya_tablerec, huggingface.co/datalab-to/line_detector0, https://huggingface.co/tarun-menta/ocr_error_detection/blob/main/config.json]
owner: research-semmyk
#---
#[Project]
#---

#short_description: PDF & HTML parser to markdown
version: 0.2.0
readme: README.md
requires-python: ">=3.12"
#dependencies: []
#---
---

# semmyKG[lightrag] - LightRAG-based Knowledge Graph Toolkit

A modular, sophisticated Gradio application for Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) using the [LightRAG][1] framework.

##   Overview

semmyKG gears towards a comprehensive solution that combines the power of LightRAG with modern web interfaces to create, query, and visualise knowledge graphs from markdown documents.
 The toolkit enables intelligent document processing, semantic search, and interactive knowledge graph visualisation with support for multiple LLM backends. It supports OpenAI and Ollama LLM backends.

##  ✨ Key Features

###  🔍 Intelligent Document processing and RAG Capabilities
- **Dual-level KG-RAG**: Combines traditional RAG with knowledge graph reasoning (powered by LightRAG)
- **Multi-modal LLM Support**: OpenAI, Ollama, and Google GenAI backends. Full GenAI support coming soon.
- **Semantic Search**: Vector-based document retrieval with embedding models (powered by LightRAG)
- **Multi-format Support**: Markdown ingestion with ParserPDF ([GitHub][3] | [HF Space][4]) integration for PDF, Word, and HTML conversion. Full integration coming soon.
- **Markdown Ingestion**: Process and index markdown files from specified directories
- **Knowledge Graph Construction**: Automatically builds entity-relationship graphs after indexing
- **Interactive Visualisation**: Real-time KG exploration

###  ️ Technical Excellence
- **Modular Architecture**: Clean, maintainable code structure
- **Async Operations**: Efficient handling of large document collections
- **Robust Error Handling**: Comprehensive logging and exception management

##  ️ Installation & Setup

### Method 1: Using UV (Recommended)
```bash
git clone https://github.com/semmyk-research/semmyKG
cd semmyKG

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate  # Linux/MacOS
# .venv\Scripts\activate on Windows

# Sync dependencies
uv pip sync
```

### Method 2: Traditional Python Setup
```bash
git clone https://github.com/semmyk-research/semmyKG
cd semmyKG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

##  🔧 Configuration

### Environment Variables Setup
Copy `.env.example` to `.env` and configure your settings:

```env
# API Configuration
OPENAI_API_KEY=your-openai-api-key

# Model Selection (format: provider/model-identifier)
LLM_MODEL=openai/gpt-oss-120b

# LLM Inference Endpoints
OPENAI_API_BASE=your-llm-provider-endpoint
# For local inference servers: http://localhost:1234/v1

# Embedding Configuration
OPENAI_API_EMBED_BASE=your-embedding-provider-endpoint
# Note: For local embedding services, do not include /embedding in URL
LLM_MODEL_EMBED=your-embedding-model

# Ollama/Local hosting Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_API_KEY=your-ollama-api-key-if-required
#[For LMStudio] OLLAMA_API_KEY=lmstudio

## Alternative: Direct Web UI Configuration
# If .env is not set, you can enter credentials directly in the web interface
```

##   Quick Start

### 1. Initialise the Application
```bash
python app.py
```

### 2. Web Interface Workflow
1. **Select Data Folder**: Choose your markdown documents directory (default: `dataset/data/docs`)
2. **Configure Settings**: 
- **Choose LLM Backend**: Select between OpenAI, Ollama, or GenAI
- Select or input other configuration in the Settings pane, 
3. **Activate**: Activate the lightRAG constructor
4. **Process Documents**: Click 'Index Documents' to process your files
5. **Query the System**: Enter your questions and select query mode
6. **Visualise Results**: Click 'Show Knowledge Graph' to finalise building Knowledge Graph and for interactive exploration

##  📁 Project Structure

```
semmyKG/
├── app_gradio_lightrag.py    # Central Gradio coordinating processing
├── app.py                    # Main Gradio app entry point
├── requirements.txt          # Project dependencies
├── .env.example              # Environment template
├── dataset/
│    └── data/
│        └── docs/            # Default document directory
├── utils/
│   ├── utils.py              # Utility functions
│   ├── file_utils.py          # File operations
│   ├── logger.py              # Logging configuration
└── logs/                     # Application logs
```

##   Deployment Options

### Local Development
```bash
python app.py
```

### HuggingFace Spaces
- **Requirements**: Ensure all dependencies in `requirements.txt`
- **Environment**: Configure via web UI or Space secrets

### Google Colab
- **Quick Setup**: Install requirements and configure tokens in 'Secret'
-  **Run**: Copy to `Files`, following folder structure and run app cells as approriate

###  📋 System Requirements

- **Python**: 3.12+
- **Memory**: 8GB+ vRAM recommended for large document sets
- **Storage**: Sufficient space for document collections and vector databases

###  🔌 Supported LLM Backends

#### OpenAI Compatible and Google GenAI
- **Models**: Frontline providers (Openai, Deepseek ...) and custom models
- **Gemini Models**: Access to Google's latest AI models
- **Endpoints**: Local inference servers (LMStudio, Jan.ai, ollama ...)
- **Embedding Models**: Multiple sentence transformer models and inference providers

#### Ollama Integration
- **Local Models**: Access to Ollama's model ecosystem
- **Self-hosted**: Complete data privacy and control


### Document Ingestion
- **Format Support**: Markdown files only (use ParserPDF for other formats)
```python
# The system automatically processes markdown files from:
# - dataset/data/docs/ (default)
```

### Query Modes
- **Semantic Search**: Vector-based similarity matching
- **KG-enhanced RAG**: Combines traditional RAG with graph reasoning

### Interactive Visualisation
- **Real-time Exploration**: Dynamic graph manipulation
- **Entity Highlighting**: Focus on specific nodes and relationships

###  📈 Performance Optimisation: Batch Processing
- **Parallel Insertion**: Configurable batch sizes
- **Rate Limiting**: Built-in delays to prevent API throttling

###  📊  Custom System Prompts: Domain-Specific Expertise
- **Domain Adaptation**: Modify prompts for specific use cases and customised NER (Named Entity Recognition)domain-specific entities rules
- **Specialised Processing**: Tailored entity recognition for security domains
- **Legislation Awareness**: Built-in understanding of legal frameworks


##  🔍 Troubleshooting

### Common Issues
- **Module Import Errors**: Ensure all dependencies are installed
- **API Connection Issues**: Verify endpoint URLs and API keys
- **Memory Management**: Monitor resource usage during large-scale indexing

### Notes
- All user-facing text are in UK English
- For advanced configuration, see LightRAG documemntation
Pending full integration, use our ParserPDF tool ([GitHub][3] | [HF Space][4]) to generate markdown from documents (PDF, Word, html)

##  🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 🛣️ Roadmap (no defined timeline)
- Integrate Huggingface log in (in progress)
- [ParserPDF][3] integration
- Pre and post processing document viewer
- Modal platform support
- Conected UX refactoring

##  📄 License

This project is licensed under the [MIT License][2].

##  🔗 References

- [LightRAG Framework][1]
- [ParserPDF Tool][3] for document conversion
- [HuggingFace Space][4] for ParserPDF


[1]: https://github.com/HKUDS/LightRAG "LightRAG GitHub Repository"
[2]: https://opensource.org/license/mit "MIT License"
[3]: https://github.com/semmyk-research/parserPDF "ParserPDF GitHub Repository"
[4]: https://huggingface.co/spaces/semmyk/parserPDF "ParserPDF HuggingFace Space"