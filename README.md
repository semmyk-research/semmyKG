---
# metadata
title: semmyKG - Knowledge Graph visualiser builder toolkit (from markdown)
emoji: 🕸️
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.44.1
python_version: 3.12
command: python app_gradio_lightrag.py
app_file: app_gradio_lightrag.py
hf_oauth: true
oauth_scopes: [read-access]
hf_oauth_scopes: [inference-api]
license: mit
pinned: true
short_description: semmyKG - Knowledge Graph builder toolkit (from markdown) | (Use ParserPDF for PDF, Word & HTML parser to markdown)
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
version: 0.1.0
readme: README.md
requires-python: ">=3.12"
#dependencies: []
#---
---

# LightRAG Gradio App

A modern, modular Gradio app for knowledge graph-based Retrieval-Augmented Generation (RAG) using [LightRAG][1]. Supports OpenAI and Ollama LLM backends, markdown document ingestion, and interactive knowledge graph visualisation. Our ParserPDF ([GitHub]][3] | [HF Space][4]) pipeline generate markdown from documents (pdf, Word, html).

## Features
- LightRAG for Dual-level RAG and knowledge graph (KG)
- Ingest markdown files from a folder (default: `dataset/data/docs`). 
- Query with OpenAI or Ollama backend (user-selectable)
- Visualise KG interactively in-browser
- Deployable to venv, Colab, or HuggingFace Spaces
- Robust, pythonic, modular code (UK English)

## Setup

### 1. Clone and create venv
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure environment
Copy `.env.example` to `.env` and fill in your keys:
```markdown
OPENAI_API_KEY=your-openai-api-key
LLM_MODEL=your-LLM-model-Name 
    ##(in the format: provider/model-identifier)
OPENAI_API_BASE=your-LLM-inference-provider-endpoint 
    ##(for locally hosted llm inference server like LMStudio or Jan.ai, follow ollama host adding /v1: http://localhost:1234/v1)
OPENAI_API_EMBED_BASE=your-embedding-provider-endpoint 
    ##(for locally hosted, do not include /embedding)
LLM_MODEL_EMBED=your-embedding-model  ##(in the format: provider/embedding-name)
OLLAMA_HOST=http://localhost:11434
OLLAMA_API_KEY=  ##(include if required)
```  
If .env is not set, you can enter into the web UI directly. <br>
Ditto, override .env by inputting directly in web UI.

### 3. Run the app
```bash
python app_gradio_lightrag.py
```  
For 'faster' development 'debug'

```python
##SMY: assist: https://www.gradio.app/guides/developing-faster-with-reload-mode
gradio app_gradio_lightrag.py --demo-name=gradio_ui
```

### 4. Colab/Spaces
- For HuggingFace Spaces: ensure all dependencies are in `requirements.txt` and `.env` is set via the web UI or Space secret.
- For Colab: install requirements and run the app cell.

## Usage
- Select your data folder (default: `dataset/data/docs`)
- Choose LLM backend (OpenAI or Ollama)
- Enter your query and select query mode
- Click 'Index Documents' to build the KG
- Click 'Query' to get answers
- Click 'Show Knowledge Graph' to visualise the KG

## Notes
- Only markdown files are supported for ingestion (images in `/images` subfolder are ignored for now). <br>NB: other formats will be enabled later: pdf, txt, html...
- To generate markdown from documents (PDf, Word, html), use our ParserPDF tool [GitHub]][3] | [HF Space][4].
- All user-facing text is in UK English
- For advanced configuration, see LightRAG documentation

## Roadmap (no defined timeline)
- HuggingFace log in
- [ParserPDF][3] integration

## License
[MIT][2] 

[1]: https://github.com/HKUDS/LightRAG "LightRAG GitHub"
[2]: https://opensource.org/license/mit "MIT License"
[3]: https://github.com/semmyk-research/parserPDF "ParserPDF (GitHub)"
[4]: https://huggingface.co/spaces/semmyk/parserPDF "ParserPDF (HF Space)"