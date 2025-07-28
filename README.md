# Intelligent Document Analyzer 🚀

An AI-powered document analysis system that uses TinyLlama LLM to intelligently extract and prioritize the most relevant sections from PDF documents based on any persona and their specific task.

## 🌟 Features

- **Universal Persona Support**: Automatically adapts to any persona (Travel Planner, PhD Researcher, Investment Analyst, etc.)
- **Dynamic Keyword Extraction**: Builds relevant keywords based on persona and task
- **Fast Processing**: Completes analysis in under 60 seconds
- **Docker Ready**: Fully containerized with pre-downloaded models
- **Offline Operation**: No internet required during runtime
- **Multi-Document Support**: Processes 3-10 PDFs simultaneously
- **Smart Ranking**: Combines keyword analysis with LLM verification

## 📋 Requirements

- Docker and Docker Compose
- 4GB RAM minimum
- 3GB disk space for Docker image
- PDF documents for analysis

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/alamayaz/pdf-analyser
cd pdf-analyser
```

### 2. Project Structure

Create the following structure:
```
intelligent-document-analyzer/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── download_model.py
├── intelligent_document_analyzer.py
├── .dockerignore
├── README.md
├── input/                    # Your input files go here
│   ├── input.json           # Configuration file
│   └── *.pdf                # PDF documents
└── output/                   # Results will appear here
```

### 3. Prepare Input Files

Create `input/input.json` with your configuration:

```json
{
    "challenge_info": {
        "challenge_id": "your_challenge_id",
        "test_case_name": "your_test_case",
        "description": "Analysis description"
    },
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Document 1 Title"
        },
        {
            "filename": "document2.pdf",
            "title": "Document 2 Title"
        }
    ],
    "persona": {
        "role": "Your Persona (e.g., Travel Planner, PhD Researcher, etc.)"
    },
    "job_to_be_done": {
        "task": "Specific task description"
    }
}
```

### 4. Build and Run with Docker

#### Option A: Using Docker Compose (Recommended)
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

#### Option B: Using Docker CLI
```bash
# Build image
docker build -t intelligent-document-analyzer .

# Run container
docker run -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" final_document-analyzer 

### 5. Check Results

The analysis results will be in `output/output.json`:

```json
{
    "metadata": {
        "input_documents": ["doc1.pdf", "doc2.pdf"],
        "persona": "Travel Planner",
        "job_to_be_done": "Plan a 4-day trip...",
        "processing_timestamp": "2025-01-27T..."
    },
    "extracted_sections": [
        {
            "document": "document1.pdf",
            "section_title": "Best Activities",
            "importance_rank": 1,
            "page_number": 3
        }
    ],
    "subsection_analysis": [
        {
            "document": "document1.pdf",
            "refined_text": "Refined content...",
            "page_number": 3
        }
    ]
}
```

## 🎯 Example Use Cases

### Travel Planning
```json
{
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a 4-day trip for a group of 10 college friends"
    }
}
```

### Academic Research
```json
{
    "persona": {
        "role": "PhD Researcher in Machine Learning"
    },
    "job_to_be_done": {
        "task": "Conduct literature review on graph neural networks"
    }
}
```

### Business Analysis
```json
{
    "persona": {
        "role": "Investment Analyst"
    },
    "job_to_be_done": {
        "task": "Analyze market trends and growth opportunities"
    }
}
```

### Medical Education
```json
{
    "persona": {
        "role": "Medical Student"
    },
    "job_to_be_done": {
        "task": "Study cardiovascular diseases for upcoming exam"
    }
}
```

## 🏗️ Architecture

### Processing Pipeline

1. **Persona Analysis**: Extracts keywords from persona and task
2. **Document Processing**: Extracts sections from PDFs (max 10 per doc)
3. **Keyword Scoring**: Scores all sections based on relevance keywords
4. **LLM Verification**: Uses TinyLlama to verify top 15 candidates
5. **Final Selection**: Selects top 5 sections with diversity
6. **Text Refinement**: Extracts most relevant sentences

### Performance Optimization

- **Hybrid Scoring**: Combines fast keyword matching with LLM verification
- **Limited LLM Calls**: Only 15 LLM calls instead of 280+
- **Page Sampling**: Processes every nth page for large documents
- **Pre-downloaded Model**: Model included in Docker image

## 🔧 Configuration

### Environment Variables

```bash
TRANSFORMERS_CACHE=/app/.cache      # Model cache directory
HF_HOME=/app/.cache                 # Hugging Face home
TOKENIZERS_PARALLELISM=false        # Disable tokenizer warnings
PYTHONUNBUFFERED=1                  # Real-time logging
```

### Docker Resource Limits

Modify in `docker-compose.yml`:
```yaml
mem_limit: 4g    # Memory limit
cpus: 2          # CPU cores
```

## 📊 Performance Metrics

- **Model Loading**: 10-15 seconds
- **PDF Processing**: ~2 seconds per document
- **Keyword Scoring**: <1 second total
- **LLM Scoring**: ~2 seconds × 15 candidates
- **Total Time**: 45-55 seconds typical

## 🛠️ Troubleshooting

### Out of Memory
```bash
# Increase memory limit in docker-compose.yml
mem_limit: 6g
```

### Slow Processing
```bash
# Increase CPU allocation
cpus: 4
```

### Permission Issues
```bash
# Fix output directory permissions
chmod 777 output/
```

### Model Download Failures
```bash
# Rebuild with no cache
docker-compose build --no-cache
```

## 📝 Development

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run locally (update paths in code)
python intelligent_document_analyzer.py
```

### Customization

1. **Change Model**: Update `model_name` in `__init__`
2. **Adjust Scoring**: Modify weights in `calculate_relevance_score`
3. **Add Keywords**: Extend `expand_keywords_with_llm`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Test with various personas
4. Submit a pull request

## 📄 License

This project is designed for educational and research purposes.

## 🙏 Acknowledgments

- TinyLlama team for the efficient language model
- PyMuPDF for PDF processing capabilities
- Hugging Face for the transformers library

---

**Note**: First run will take longer due to model download during Docker build. Subsequent runs will be much faster as the model is cached in the image.
