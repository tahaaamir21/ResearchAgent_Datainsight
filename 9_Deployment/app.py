"""
🔬 Multi-Agent Research Intelligence Platform - Streamlit UI
Interactive web interface for running research queries and viewing results
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import time
from PIL import Image

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "4_Src_Code"))

# Import the research pipeline
from agentic_ai_pipeline import run_research_pipeline

# Page configuration
st.set_page_config(
    page_title="Research Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        color: #1f1f1f;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .source-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
        color: #1f1f1f;
    }
    .concept-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_result' not in st.session_state:
    st.session_state.research_result = None
if 'running' not in st.session_state:
    st.session_state.running = False

# Header
st.markdown('<h1 class="main-header">🔬 Research Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Autonomous Multi-Agent System for Academic Research Discovery & Analysis</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### Environment Setup")
    
    # Automatically load from environment
    groq_key = os.getenv("GROQ_API_KEY", "")
    serpapi_key = os.getenv("SERPAPI_KEY", "")
    
    # Show status
    if groq_key:
        st.success("✅ GROQ API Key: Configured")
    else:
        st.error("❌ GROQ API Key: Not found")
        st.info("💡 Set `GROQ_API_KEY` in your `.env` file or environment variables")
    
    if serpapi_key:
        st.success("✅ SERPAPI Key: Configured (Optional)")
    else:
        st.warning("⚠️ SERPAPI Key: Not configured (Web search disabled)")
        st.info("💡 Set `SERPAPI_KEY` in your `.env` file for web search")
    
    st.markdown("---")
    st.markdown("### System Capabilities")
    st.markdown("""
    - 🔍 **Multi-Source Discovery**  
      ArXiv, Semantic Scholar, Web
    
    - ✅ **Smart Validation**  
      Credibility & relevance scoring
    
    - 🧬 **Synthesis**  
      Knowledge graphs & gap analysis
    
    - 🤖 **ML Analysis**  
      Topic modeling, clustering, predictions
    
    - 📊 **Reporting**  
      Comprehensive research reports
    
    - 📡 **Monitoring**  
      Trend analysis & alerts
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **Version:** 2.0  
    **Agents:** 7 specialized AI agents  
    **Framework:** LangGraph + Groq LLM  
    **RAG:** ChromaDB + HuggingFace
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["🚀 New Research", "📊 Results", "📚 History"])

with tab1:
    st.header("Start New Research")
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Research Query",
            placeholder="e.g., Large Language Models for code generation",
            help="Enter your research question or topic"
        )
    
    with col2:
        depth = st.selectbox(
            "Research Depth",
            options=["quick", "standard", "deep"],
            index=1,
            help="Quick: ~5 min | Standard: ~8 min | Deep: ~12 min"
        )
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - `Transformer models for computer vision`
        - `Quantum computing in drug discovery`
        - `Few-shot learning in natural language processing`
        - `Graph neural networks for molecular property prediction`
        - `Federated learning privacy techniques`
        """)
    
    # Run button
    if st.button("🚀 Start Research", type="primary", disabled=st.session_state.running):
        if not query:
            st.error("Please enter a research query!")
        elif not groq_key:
            st.error("Please provide your GROQ API Key in the sidebar!")
        else:
            st.session_state.running = True
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Agent progress stages
            stages = [
                ("🔍 Discovery Agent", "Searching ArXiv, Semantic Scholar, and Web..."),
                ("✅ Validation Agent", "Scoring source credibility and relevance..."),
                ("🧠 RAG Agent", "Creating vector store and embeddings..."),
                ("🧬 Synthesis Agent", "Building knowledge graph and finding gaps..."),
                ("🤖 ML Agent", "Performing topic modeling and clustering..."),
                ("📊 Reporter Agent", "Generating comprehensive report..."),
                ("📡 Monitoring Agent", "Setting up trend analysis...")
            ]
            
            # Show progress
            for i, (agent, description) in enumerate(stages):
                progress = (i + 1) / len(stages)
                progress_bar.progress(progress)
                status_text.markdown(f"### {agent}\n{description}")
                
                # Run the actual pipeline when reaching the first stage
                if i == 0:
                    try:
                        result = run_research_pipeline(query=query, research_depth=depth)
                        st.session_state.research_result = result
                        
                        # Update progress for remaining stages
                        for j in range(i + 1, len(stages)):
                            time.sleep(0.5)
                            progress = (j + 1) / len(stages)
                            progress_bar.progress(progress)
                            status_text.markdown(f"### {stages[j][0]}\n{stages[j][1]}")
                        
                    except Exception as e:
                        st.error(f"Research failed: {str(e)}")
                        st.session_state.running = False
                        st.stop()
            
            # Complete
            progress_bar.progress(1.0)
            status_text.markdown("### ✅ Research Complete!")
            st.success("Research completed successfully! Switch to the 'Results' tab to view the report.")
            st.session_state.running = False
            st.balloons()

with tab2:
    st.header("Research Results")
    
    if st.session_state.research_result is None:
        st.info("👈 Run a research query in the 'New Research' tab to see results here.")
    else:
        result = st.session_state.research_result
        
        # Executive Summary
        st.markdown("## 🎯 Executive Summary")
        st.markdown(f"""
        <div class="metric-card">
        {result.get('executive_summary', 'No summary available')}
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics
        st.markdown("## 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sources Found",
                result.get('discovery_metadata', {}).get('total_found', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Sources Validated",
                len(result.get('validated_sources', [])),
                delta=None
            )
        
        with col3:
            st.metric(
                "Quality Score",
                f"{result.get('source_quality_avg', 0):.1f}/100",
                delta=None
            )
        
        with col4:
            st.metric(
                "Key Concepts",
                len(result.get('key_concepts', [])),
                delta=None
            )
        
        # Knowledge Graph
        st.markdown("## 🕸️ Knowledge Graph")
        kg_files = list(Path(".").glob("kg_*.png"))
        if kg_files:
            latest_kg = max(kg_files, key=os.path.getctime)
            try:
                image = Image.open(latest_kg)
                st.image(image, caption="Research Knowledge Graph", use_container_width=True)
            except:
                st.warning("Knowledge graph image could not be loaded")
        else:
            st.info("No knowledge graph visualization available")
        
        # Key Concepts
        st.markdown("## 🔑 Key Concepts")
        concepts = result.get('key_concepts', [])[:20]
        if concepts:
            concept_html = "".join([f'<span class="concept-badge">{c}</span>' for c in concepts])
            st.markdown(concept_html, unsafe_allow_html=True)
        else:
            st.info("No concepts extracted")
        
        # Consensus Findings
        st.markdown("## ✅ Consensus Findings")
        findings = result.get('consensus_findings', [])
        if findings:
            for i, finding in enumerate(findings, 1):
                st.markdown(f"{i}. {finding}")
        else:
            st.info("No consensus findings available")
        
        # Research Gaps
        st.markdown("## 🔍 Research Gaps")
        gaps = result.get('research_gaps', [])
        if gaps:
            for i, gap in enumerate(gaps, 1):
                with st.expander(f"Gap {i}: {gap.get('gap', 'Unknown')}", expanded=False):
                    st.markdown(f"**Why it matters:** {gap.get('importance', 'N/A')}")
        else:
            st.info("No research gaps identified")
        
        # ML Analysis
        ml_insights = result.get('ml_insights', {})
        if ml_insights and ml_insights.get('status') != 'unavailable':
            st.markdown("## 🤖 Machine Learning Analysis")
            
            # Topics
            ml_topics = result.get('ml_topics', [])
            if ml_topics:
                st.markdown("### 📌 Discovered Topics (LDA)")
                for topic in ml_topics[:5]:
                    st.markdown(f"**Topic {topic['topic_id']}:** {', '.join(topic['keywords'])}")
            
            # Clusters
            clusters = result.get('paper_clusters', {})
            if clusters and clusters.get('clusters'):
                st.markdown("### 📊 Paper Clusters")
                st.markdown(f"Papers grouped into **{clusters.get('n_clusters', 0)}** thematic clusters")
                
                for cluster_id, theme in clusters.get('cluster_themes', {}).items():
                    size = clusters.get('cluster_sizes', {}).get(cluster_id, 0)
                    with st.expander(f"Cluster {cluster_id} ({size} papers): {', '.join(theme[:3])}"):
                        papers = clusters.get('clusters', {}).get(cluster_id, [])
                        for paper in papers[:5]:
                            st.markdown(f"- {paper['title']}")
            
            # Citation Predictions
            predictions = result.get('citation_predictions', [])
            if predictions:
                st.markdown("### 🔮 Citation Predictions")
                for pred in predictions[:5]:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"**{pred['title']}**")
                    with col2:
                        st.metric("Current", pred['current_citations'])
                    with col3:
                        st.metric("Predicted", pred['predicted_citations_1yr'], 
                                delta=f"{pred['predicted_growth_rate']}%")
        
        # Top Sources
        st.markdown("## 📚 Top Sources")
        citation_map = result.get('citation_map', {})
        top_cited = citation_map.get('top_cited', [])[:10]
        
        if top_cited:
            for i, (title, citations) in enumerate(top_cited, 1):
                st.markdown(f"""
                <div class="source-card">
                <strong>{i}. {title[:100]}</strong><br>
                <small>Citations: {citations}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No source citation data available")
        
        # Download Options
        st.markdown("## 💾 Download")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download full report
            report_files = list(Path(".").glob("report_*.txt"))
            if report_files:
                latest_report = max(report_files, key=os.path.getctime)
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report_content,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            # Download as JSON
            json_data = json.dumps({
                "query": result.get('query'),
                "executive_summary": result.get('executive_summary'),
                "key_concepts": result.get('key_concepts'),
                "consensus_findings": result.get('consensus_findings'),
                "research_gaps": result.get('research_gaps'),
                "metadata": {
                    "sources_found": result.get('discovery_metadata', {}).get('total_found'),
                    "sources_validated": len(result.get('validated_sources', [])),
                    "quality_score": result.get('source_quality_avg'),
                    "timestamp": result.get('completed_at')
                }
            }, indent=2)
            
            st.download_button(
                label="📦 Download Data (JSON)",
                data=json_data,
                file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Download knowledge graph
            if kg_files:
                latest_kg = max(kg_files, key=os.path.getctime)
                with open(latest_kg, 'rb') as f:
                    st.download_button(
                        label="🖼️ Download Graph (PNG)",
                        data=f,
                        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )

with tab3:
    st.header("Research History")
    st.info("🚧 Research history feature coming soon! Your past research sessions will be saved and accessible here.")
    
    # Show recent report files
    report_files = sorted(Path(".").glob("report_*.txt"), key=os.path.getctime, reverse=True)
    kg_files = sorted(Path(".").glob("kg_*.png"), key=os.path.getctime, reverse=True)
    
    if report_files or kg_files:
        st.markdown("### Recent Files")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Reports:**")
            for report in report_files[:5]:
                st.markdown(f"- `{report.name}` ({report.stat().st_size // 1024} KB)")
        
        with col2:
            st.markdown("**Knowledge Graphs:**")
            for kg in kg_files[:5]:
                st.markdown(f"- `{kg.name}` ({kg.stat().st_size // 1024} KB)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Multi-Agent Research Intelligence Platform | Powered by LangGraph, Groq LLM & ChromaDB</p>
    <p>Built with ❤️ using Streamlit | © 2025</p>
</div>
""", unsafe_allow_html=True)

