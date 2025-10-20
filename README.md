# 🔬 Multi-Agent Research Intelligence Platform

An autonomous AI-powered research system that discovers, validates, synthesizes, and monitors academic research across multiple sources using a multi-agent architecture.

## 🌟 Features

- **🔍 Multi-Source Discovery**: Searches ArXiv, Semantic Scholar, and the web simultaneously
- **✅ Smart Validation**: AI-powered credibility scoring and relevance checking
- **🧬 Knowledge Synthesis**: Builds knowledge graphs, identifies consensus, gaps, and contradictions
- **🤖 ML Analysis**: Topic modeling (LDA), paper clustering (K-means), citation prediction
- **📊 Comprehensive Reporting**: Generates detailed research reports with visualizations
- **📡 Trend Monitoring**: Analyzes emerging topics and tracks research trends
- **🎨 Interactive UI**: Beautiful Streamlit web interface for easy interaction
- **🔄 Production Ready**: Apache Airflow integration for workflow orchestration

## 🏗️ Architecture

```
Multi-Agent System (7 Specialized Agents)
├── Discovery Agent: Multi-source search
├── Validation Agent: Quality scoring
├── RAG Agent: Vector embeddings (ChromaDB)
├── Synthesis Agent: Knowledge graph building
├── ML Agent: Machine learning analysis
├── Reporter Agent: Report generation
└── Monitoring Agent: Trend analysis

Powered by: LangGraph + Groq LLM + ChromaDB
```

## 📁 Project Structure

```
datascienceproj/
├── 1_Documentation/          # Problem statements, docs
├── 2_Data/                   # Raw and processed data
│   ├── raw/                  # Raw research data
│   └── processed/            # Validated sources
├── 3_Notebooks/              # Jupyter notebooks for testing
├── 4_Src_Code/               # Main source code
│   ├── agents/               # Individual agent modules
│   └── agentic_ai_pipeline.py  # Main pipeline
├── 5_Pipeline/               # Airflow orchestration
│   └── airflow_dag.py        # Production DAG
├── 6_Models/                 # ML models and vector stores
│   └── vectorstore/          # ChromaDB storage
├── 7_Results/                # Generated outputs
│   ├── knowledge_graph.png
│   └── research_reports/
├── 8_Demo/                   # Demo materials
│   └── screenshots/
├── 9_Deployment/             # Deployment files
│   └── app.py                # Streamlit UI
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd datascienceproj

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with your API keys:

**Step 1: Copy the example file**
```bash
# Windows
rename env.example .env

# Mac/Linux
mv env.example .env
```

**Step 2: Edit `.env` and add your keys**
```env
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_KEY=your_serpapi_key_here  # Optional, for web search
```

Get your API keys:
- **Groq API Key** (Required): https://console.groq.com/
- **SERPAPI Key** (Optional): https://serpapi.com/

### 3. Run Options

#### Option A: Streamlit Web UI (Recommended)

```bash
streamlit run 9_Deployment/app.py
```

Then open your browser to `http://localhost:8501`

#### Option B: Command Line

```bash
python 4_Src_Code/agentic_ai_pipeline.py
```

#### Option C: Airflow Orchestration (Production)

```bash
# See 5_Pipeline/README.md for Airflow setup
docker-compose -f docker-compose-airflow.yml up -d
```

## 📖 Usage Example

### Web UI
1. Open the Streamlit app
2. Enter your API keys in the sidebar
3. Type your research query (e.g., "Transformer models for computer vision")
4. Select research depth (quick/standard/deep)
5. Click "Start Research"
6. View results, download reports, and explore knowledge graphs

### Python API
```python
from agentic_ai_pipeline import run_research_pipeline

result = run_research_pipeline(
    query="Large Language Models for code generation",
    research_depth="standard"
)

print(result['executive_summary'])
print(f"Found {len(result['validated_sources'])} validated sources")
```

## 🧠 How It Works

1. **Discovery Phase**: Searches multiple sources (ArXiv, Semantic Scholar, Web)
2. **Validation Phase**: Scores each source for credibility and relevance
3. **RAG Integration**: Creates vector embeddings in ChromaDB for semantic search
4. **Synthesis Phase**: Builds knowledge graph, finds consensus and gaps
5. **ML Analysis**: Performs topic modeling, clustering, and predictions
6. **Reporting Phase**: Generates comprehensive research report
7. **Monitoring Phase**: Sets up alerts and trend tracking

## 📊 Output Files

After running a research query, you'll get:

- **Research Report** (`report_*.txt`): Comprehensive analysis
- **Knowledge Graph** (`kg_*.png`): Visual representation of concepts
- **Vector Database** (`chroma_db/`): Semantic search embeddings
- **JSON Data**: Structured research data for further processing

## 🛠️ Technologies

- **LLM Framework**: LangChain, LangGraph
- **LLM Provider**: Groq (llama-3.1-8b-instant)
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace sentence-transformers
- **ML**: scikit-learn (LDA, K-means)
- **Orchestration**: Apache Airflow
- **UI**: Streamlit
- **Visualization**: matplotlib, networkx

## 📚 Documentation

- **Pipeline Documentation**: See `4_Src_Code/README.md`
- **Airflow Setup**: See `5_Pipeline/README.md`
- **Deployment Guide**: See `9_Deployment/README.md`

## 🔧 Configuration

### Research Depth Options

- **Quick** (~5 min): 10-15 sources, basic analysis
- **Standard** (~8 min): 20-25 sources, full analysis
- **Deep** (~12 min): 30+ sources, comprehensive analysis

### Customization

You can customize agent behavior by modifying:
- `4_Src_Code/agentic_ai_pipeline.py`: Agent logic and prompts
- `5_Pipeline/airflow_dag.py`: Workflow orchestration
- `9_Deployment/app.py`: UI appearance and features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **LangChain**: For the amazing agent framework
- **Groq**: For fast LLM inference
- **Streamlit**: For the beautiful UI framework
- **ArXiv & Semantic Scholar**: For open access to research

## 📞 Support

For issues, questions, or feedback:
- Create an issue on GitHub
- Contact the maintainers

---

**Built with ❤️ for the research community** | © 2025

