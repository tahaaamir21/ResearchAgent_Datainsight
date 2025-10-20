# ============================================================================
# COMPLETE MULTI-AGENT RESEARCH INTELLIGENCE PLATFORM
# All Agents: Orchestrator, Discovery, Validation, Synthesis, ML, Reporter, Monitoring
# With RAG Integration + ML Analysis + Knowledge Graphs + Airflow Orchestration
# Using: Groq LLM + LangGraph + ChromaDB + Scikit-learn + Apache Airflow + arXiv + SerpAPI + Semantic Scholar
#
# DEPLOYMENT OPTIONS:
# 1. Standalone: python main.py (interactive mode)
# 2. Airflow DAG: See airflow_research_dag.py for production orchestration
# ============================================================================

"""
SETUP INSTRUCTIONS:
1. Install dependencies:
   pip install langgraph langchain langchain-groq langchain-huggingface langchain-chroma
   pip install arxiv google-search-results chromadb sentence-transformers
   pip install requests beautifulsoup4 python-dotenv networkx matplotlib
   pip install plotly scikit-learn nltk gensim

2. Create a .env file:
   GROQ_API_KEY=your_groq_key_here
   SERPAPI_KEY=your_serpapi_key_here (optional - for web search)
   
3. Run:
   python main.py
"""

import os
import json
from typing import List, Dict, TypedDict, Annotated, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import arxiv
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None
    print("Warning: serpapi not installed. Web search will be disabled.")
    print("Install with: pip install google-search-results")
import requests
from collections import Counter, defaultdict
import re
import hashlib

# LangChain & LangGraph imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Visualization
import networkx as nx
try:
    import matplotlib.pyplot as plt
except:
    print("matplotlib not available for visualization")

# Machine Learning imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  Warning: scikit-learn not installed. ML features will be disabled.")
    print("   Install with: pip install scikit-learn numpy")

# Load environment variables
load_dotenv()

# ============================================================================
# PART 1: COMPLETE STATE DEFINITION
# ============================================================================

class ResearchState(TypedDict):
    """Complete state with all agents"""
    # User Input
    query: str
    research_depth: str
    user_id: Optional[str]  # For monitoring
    
    # Discovery Phase
    raw_sources: List[Dict]
    discovery_metadata: Dict
    
    # Validation Phase
    validated_sources: List[Dict]
    validation_scores: List[Dict]
    credibility_report: Dict
    
    # Synthesis Phase
    knowledge_graph: Dict
    research_gaps: List[Dict]
    consensus_findings: List[str]
    contradictions: List[Dict]
    key_concepts: List[str]
    
    # RAG Phase
    vector_store_id: str
    embeddings_created: bool
    rag_ready: bool
    
    # Reporter Phase
    executive_summary: str
    detailed_report: str
    citation_map: Dict
    visualizations: List[str]
    
    # Monitoring Phase
    monitoring_enabled: bool
    alert_triggers: List[Dict]
    trend_analysis: Dict
    
    # ML Analysis Phase
    ml_topics: List[Dict]
    paper_clusters: Dict
    citation_predictions: List[Dict]
    ml_quality_scores: List[Dict]
    ml_insights: Dict
    
    # Quality Metrics
    source_quality_avg: float
    citation_counts: Dict
    conflicts_detected: List[Dict]
    
    # Orchestration
    current_agent: str
    workflow_status: str
    errors: List[str]
    synthesis_ready: bool
    
    # Timestamps
    started_at: str
    completed_at: str


# ============================================================================
# PART 2: DISCOVERY AGENT (Enhanced)
# ============================================================================

class DiscoveryAgent:
    """Autonomous multi-source search agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.sources_searched = []
        
    def search_arxiv(self, query: str, max_results: int = 8) -> List[Dict]:
        """Search arXiv"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                results.append({
                    "id": f"arxiv_{paper.entry_id.split('/')[-1]}",
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "summary": paper.summary,
                    "full_text": paper.summary,  # For RAG
                    "url": paper.pdf_url,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "categories": paper.categories,
                    "source_type": "arxiv",
                    "metadata": {
                        "arxiv_id": paper.entry_id.split('/')[-1],
                        "primary_category": paper.primary_category
                    }
                })
            
            self.sources_searched.append("arxiv")
            print(f"  ✓ arXiv: {len(results)} papers")
            return results
        except Exception as e:
            print(f"  ✗ arXiv failed: {e}")
            return []
    
    def search_web(self, query: str, num_results: int = 8) -> List[Dict]:
        """Search web"""
        try:
            if GoogleSearch is None:
                return []
            
            api_key = os.getenv("SERPAPI_KEY")
            if not api_key:
                return []
            
            search = GoogleSearch({
                "q": query,
                "api_key": api_key,
                "num": num_results
            })
            
            results_dict = search.get_dict()
            organic_results = results_dict.get("organic_results", [])
            
            results = []
            for r in organic_results:
                results.append({
                    "id": f"web_{hashlib.md5(r.get('link', '').encode()).hexdigest()[:8]}",
                    "title": r.get("title", "No title"),
                    "url": r.get("link", ""),
                    "summary": r.get("snippet", ""),
                    "full_text": r.get("snippet", ""),
                    "source_type": "web",
                    "metadata": {
                        "position": r.get("position", 999)
                    }
                })
            
            self.sources_searched.append("web")
            print(f"  ✓ Web: {len(results)} results")
            return results
        except Exception as e:
            print(f"  ✗ Web failed: {e}")
            return []
    
    def search_semantic_scholar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Semantic Scholar"""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,authors,abstract,year,citationCount,url,venue,publicationDate"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            papers = data.get("data", [])
            
            results = []
            for paper in papers:
                paper_id = hashlib.md5(paper.get("title", "").encode()).hexdigest()[:8]
                results.append({
                    "id": f"scholar_{paper_id}",
                    "title": paper.get("title", "Unknown"),
                    "authors": [a.get("name", "Unknown") for a in paper.get("authors", [])],
                    "summary": paper.get("abstract", "")[:500],
                    "full_text": paper.get("abstract", ""),
                    "url": paper.get("url", ""),
                    "published": paper.get("publicationDate", paper.get("year", "Unknown")),
                    "citation_count": paper.get("citationCount", 0),
                    "source_type": "semantic_scholar",
                    "metadata": {
                        "venue": paper.get("venue", "Unknown"),
                        "citations": paper.get("citationCount", 0)
                    }
                })
            
            self.sources_searched.append("semantic_scholar")
            print(f"  ✓ Semantic Scholar: {len(results)} papers")
            return results
        except Exception as e:
            print(f"  ✗ Semantic Scholar failed: {e}")
            return []
    
    def reformulate_query(self, original_query: str) -> List[str]:
        """Generate alternative queries"""
        try:
            prompt = f"""Generate 2 alternative search queries for: "{original_query}"

Keep core meaning, use different keywords. Return ONLY queries, one per line."""

            response = self.llm.invoke(prompt)
            alternatives = [q.strip() for q in response.content.split('\n') if q.strip()]
            return alternatives[:2]
        except:
            return []
    
    def deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """Remove duplicates"""
        if not sources:
            return []
        
        unique = []
        seen_titles = set()
        
        for source in sources:
            title = source.get("title", "").lower().strip()
            title_key = re.sub(r'[^\w\s]', '', title)[:50]
            
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(source)
        
        return unique
    
    def discover(self, state: ResearchState) -> Dict:
        """Main discovery workflow"""
        print("\n🔍 DISCOVERY AGENT")
        print("="*70)
        
        query = state["query"]
        depth = state.get("research_depth", "standard")
        
        all_sources = []
        
        print("\n[Phase 1] Primary Search...")
        all_sources.extend(self.search_arxiv(query, max_results=8))
        all_sources.extend(self.search_web(query, num_results=8))
        all_sources.extend(self.search_semantic_scholar(query, limit=5))
        
        if depth in ["standard", "deep"] and len(all_sources) < 15:
            print("\n[Phase 2] Reformulated Search...")
            alt_queries = self.reformulate_query(query)
            for alt_query in alt_queries[:1]:
                print(f"  → {alt_query}")
                all_sources.extend(self.search_arxiv(alt_query, max_results=3))
                all_sources.extend(self.search_semantic_scholar(alt_query, limit=3))
        
        print("\n[Phase 3] Deduplication...")
        unique_sources = self.deduplicate_sources(all_sources)
        
        print(f"\n✓ Discovery: {len(unique_sources)} unique sources")
        
        return {
            "raw_sources": unique_sources,
            "discovery_metadata": {
                "total_found": len(all_sources),
                "unique_sources": len(unique_sources),
                "sources_searched": list(set(self.sources_searched)),
                "timestamp": datetime.now().isoformat()
            }
        }


# ============================================================================
# PART 3: VALIDATION AGENT
# ============================================================================

class ValidationAgent:
    """Source validation and quality assessment"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def calculate_source_score(self, source: Dict) -> Dict:
        """Calculate credibility score"""
        score = 0
        factors = []
        
        # Source type (30 points)
        source_type = source.get("source_type", "")
        if source_type == "semantic_scholar":
            score += 28
            factors.append("Peer-reviewed (+28)")
        elif source_type == "arxiv":
            score += 25
            factors.append("arXiv preprint (+25)")
        else:
            score += 15
            factors.append("Web source (+15)")
        
        # Citations (25 points)
        citations = source.get("citation_count", 0)
        if citations > 100:
            citation_score = 25
        elif citations > 50:
            citation_score = 20
        elif citations > 10:
            citation_score = 15
        else:
            citation_score = 5
        score += citation_score
        factors.append(f"Citations: {citations} (+{citation_score})")
        
        # Recency (20 points)
        try:
            pub_date = source.get("published", "")
            if pub_date:
                year = int(str(pub_date).split('-')[0])
                age = datetime.now().year - year
                if age <= 1:
                    recency_score = 20
                elif age <= 3:
                    recency_score = 15
                elif age <= 5:
                    recency_score = 10
                else:
                    recency_score = 5
                score += recency_score
                factors.append(f"Age: {age}y (+{recency_score})")
        except:
            score += 10
        
        # Content quality (25 points)
        summary = source.get("summary", "")
        if len(summary) > 200:
            score += 20
            factors.append("Substantial (+20)")
        elif len(summary) > 100:
            score += 15
            factors.append("Moderate (+15)")
        else:
            score += 5
        
        final_score = min(score, 100)
        
        return {
            "score": final_score,
            "factors": factors,
            "grade": self._score_to_grade(final_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 85: return "A - Excellent"
        elif score >= 70: return "B - Good"
        elif score >= 55: return "C - Fair"
        elif score >= 40: return "D - Poor"
        else: return "F - Very Poor"
    
    def check_relevance(self, source: Dict, query: str) -> Dict:
        """Check relevance with LLM"""
        try:
            prompt = f"""Assess relevance to query.

Query: {query}
Title: {source.get('title', 'Unknown')}
Summary: {source.get('summary', '')[:300]}

Format:
RELEVANT: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASON: [One sentence]"""

            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            is_relevant = "YES" in content.split('\n')[0].upper()
            confidence = "MEDIUM"
            reason = "No reason"
            
            for line in content.split('\n'):
                if "CONFIDENCE:" in line:
                    confidence = line.split(':')[1].strip()
                if "REASON:" in line:
                    reason = line.split(':', 1)[1].strip()
            
            return {
                "is_relevant": is_relevant,
                "confidence": confidence,
                "reason": reason
            }
        except:
            return {"is_relevant": True, "confidence": "LOW", "reason": "Error"}
    
    def validate(self, state: ResearchState) -> Dict:
        """Main validation workflow"""
        print("\n✅ VALIDATION AGENT")
        print("="*70)
        
        raw_sources = state["raw_sources"]
        query = state["query"]
        
        validated = []
        scores = []
        
        print(f"\n[Phase 1] Scoring {len(raw_sources)} sources...")
        
        for idx, source in enumerate(raw_sources, 1):
            score_result = self.calculate_source_score(source)
            relevance = self.check_relevance(source, query)
            
            if relevance["is_relevant"] and score_result["score"] >= 40:
                validated.append(source)
                scores.append({
                    "source_id": source.get("id", ""),
                    "source_title": source.get("title", ""),
                    "credibility_score": score_result["score"],
                    "grade": score_result["grade"],
                    "factors": score_result["factors"],
                    "relevance": relevance
                })
                print(f"  ✓ Source {idx}: {score_result['grade']}")
            else:
                print(f"  ✗ Source {idx}: Rejected")
        
        avg_score = sum(s["credibility_score"] for s in scores) / len(scores) if scores else 0
        
        credibility_report = {
            "total_validated": len(validated),
            "average_quality_score": round(avg_score, 2),
            "score_distribution": {
                "excellent": len([s for s in scores if s["credibility_score"] >= 85]),
                "good": len([s for s in scores if 70 <= s["credibility_score"] < 85]),
                "fair": len([s for s in scores if 55 <= s["credibility_score"] < 70])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n✓ Validation: {len(validated)}/{len(raw_sources)} passed")
        print(f"  Avg Quality: {avg_score:.1f}/100")
        
        return {
            "validated_sources": validated,
            "validation_scores": scores,
            "credibility_report": credibility_report,
            "source_quality_avg": avg_score
        }


# ============================================================================
# PART 4: RAG INTEGRATION AGENT
# ============================================================================

class RAGAgent:
    """RAG integration with ChromaDB"""
    
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def create_vector_store(self, sources: List[Dict], query: str) -> Chroma:
        """Create ChromaDB vector store from sources"""
        print("\n🧠 RAG AGENT - Creating Vector Store")
        print("="*70)
        
        documents = []
        
        for source in sources:
            # Create document for each source
            text = f"""Title: {source.get('title', '')}
Authors: {', '.join(source.get('authors', [])[:3])}
Summary: {source.get('full_text', source.get('summary', ''))}
Published: {source.get('published', 'Unknown')}
Source: {source.get('source_type', 'unknown')}"""
            
            doc = Document(
                page_content=text,
                metadata={
                    "source_id": source.get("id", ""),
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "source_type": source.get("source_type", ""),
                    "citation_count": source.get("citation_count", 0)
                }
            )
            documents.append(doc)
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        print(f"  → Created {len(split_docs)} chunks from {len(sources)} sources")
        
        # Create vector store
        collection_name = f"research_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        
        print(f"  ✓ Vector store created: {collection_name}")
        
        return vectorstore, collection_name
    
    def query_rag(self, vectorstore: Chroma, query: str, k: int = 5) -> List[Dict]:
        """Query the RAG system"""
        results = vectorstore.similarity_search(query, k=k)
        
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return formatted_results
    
    def integrate(self, state: ResearchState) -> Dict:
        """Main RAG integration"""
        validated = state["validated_sources"]
        query = state["query"]
        
        if not validated:
            print("⚠ No sources to embed")
            return {
                "embeddings_created": False,
                "rag_ready": False,
                "vector_store_id": ""
            }
        
        vectorstore, collection_id = self.create_vector_store(validated, query)
        
        # Test query
        print("\n[Testing RAG]")
        test_results = self.query_rag(vectorstore, query, k=3)
        print(f"  ✓ Retrieved {len(test_results)} relevant chunks")
        
        return {
            "embeddings_created": True,
            "rag_ready": True,
            "vector_store_id": collection_id,
            "_vectorstore": vectorstore  # Store for later use
        }


# ============================================================================
# PART 5: SYNTHESIS AGENT (NEW!)
# ============================================================================

class SynthesisAgent:
    """Synthesize findings, detect gaps, build knowledge graph"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_key_concepts(self, sources: List[Dict]) -> List[str]:
        """Extract main concepts from all sources"""
        try:
            # Combine all titles and summaries
            combined_text = "\n".join([
                f"{s.get('title', '')} {s.get('summary', '')[:200]}"
                for s in sources[:10]
            ])
            
            prompt = f"""Extract 10-15 key concepts/terms from this research:

{combined_text}

Return ONLY a comma-separated list of concepts. Be specific and technical."""

            response = self.llm.invoke(prompt)
            concepts = [c.strip() for c in response.content.split(',')]
            
            print(f"  ✓ Extracted {len(concepts)} key concepts")
            return concepts[:15]
        except:
            return []
    
    def find_consensus(self, sources: List[Dict], query: str) -> List[str]:
        """Identify consensus findings across sources"""
        try:
            summaries = "\n\n".join([
                f"Source {i+1}: {s.get('title', '')}\n{s.get('summary', '')[:300]}"
                for i, s in enumerate(sources[:8])
            ])
            
            prompt = f"""Identify consensus findings across these sources about: {query}

{summaries}

List 5-7 findings that MULTIPLE sources agree on. Format:
- Finding 1
- Finding 2
etc."""

            response = self.llm.invoke(prompt)
            findings = [line.strip('- ').strip() for line in response.content.split('\n') 
                       if line.strip().startswith('-')]
            
            print(f"  ✓ Found {len(findings)} consensus points")
            return findings
        except:
            return []
    
    def detect_contradictions(self, sources: List[Dict]) -> List[Dict]:
        """Detect contradicting findings"""
        contradictions = []
        
        # Simple keyword-based detection
        positive_terms = ['improve', 'enhance', 'effective', 'successful', 'increase']
        negative_terms = ['not', 'no', 'without', 'fails', 'decrease', 'poor']
        
        positive_sources = []
        negative_sources = []
        
        for source in sources:
            title_lower = source.get('title', '').lower()
            summary_lower = source.get('summary', '').lower()
            text = title_lower + " " + summary_lower
            
            if any(term in text for term in positive_terms):
                positive_sources.append(source.get('title', ''))
            if any(term in text for term in negative_terms):
                negative_sources.append(source.get('title', ''))
        
        if len(positive_sources) > 0 and len(negative_sources) > 0:
            contradictions.append({
                "type": "effectiveness_debate",
                "description": f"Mixed results: {len(positive_sources)} positive vs {len(negative_sources)} negative findings",
                "severity": "medium",
                "sources_positive": positive_sources[:3],
                "sources_negative": negative_sources[:3]
            })
        
        return contradictions
    
    def identify_research_gaps(self, sources: List[Dict], concepts: List[str]) -> List[Dict]:
        """Identify under-researched areas"""
        try:
            concepts_str = ", ".join(concepts[:10])
            
            prompt = f"""Based on these concepts: {concepts_str}

Identify 3-5 research gaps or under-explored areas. Format:
GAP: [gap name]
WHY: [why it matters]
---
GAP: [gap name]
WHY: [why it matters]"""

            response = self.llm.invoke(prompt)
            
            gaps = []
            current_gap = {}
            
            for line in response.content.split('\n'):
                line = line.strip()
                if line.startswith('GAP:'):
                    if current_gap:
                        gaps.append(current_gap)
                    current_gap = {"gap": line.replace('GAP:', '').strip()}
                elif line.startswith('WHY:'):
                    current_gap["importance"] = line.replace('WHY:', '').strip()
                elif line == '---' and current_gap:
                    gaps.append(current_gap)
                    current_gap = {}
            
            if current_gap:
                gaps.append(current_gap)
            
            print(f"  ✓ Identified {len(gaps)} research gaps")
            return gaps
        except:
            return []
    
    def build_knowledge_graph(self, sources: List[Dict], concepts: List[str]) -> Dict:
        """Build knowledge graph structure with better connectivity"""
        print("  → Building knowledge graph...")
        
        graph = {
            "nodes": [],
            "edges": [],
            "metadata": {}
        }
        
        # Add concept nodes (ensure we have meaningful concepts)
        valid_concepts = [c for c in concepts[:15] if len(c.strip()) > 2]
        for concept in valid_concepts:
            concept_id = concept.lower().replace(' ', '_').replace('-', '_')
            graph["nodes"].append({
                "id": concept_id,
                "label": concept,
                "type": "concept"
            })
        
        # Add source nodes
        for source in sources[:10]:
            graph["nodes"].append({
                "id": source.get("id", ""),
                "label": source.get("title", "")[:60],
                "type": "source",
                "url": source.get("url", "")
            })
        
        # Create edges: sources -> concepts (co-occurrence)
        for source in sources[:10]:
            text = (source.get('title', '') + ' ' + source.get('summary', '')).lower()
            edges_added = 0
            for concept in valid_concepts:
                # More flexible matching
                concept_lower = concept.lower()
                concept_words = concept_lower.split()
                
                # Check if concept or any major word from it appears in text
                if (concept_lower in text or 
                    any(word in text for word in concept_words if len(word) > 3)):
                    concept_id = concept_lower.replace(' ', '_').replace('-', '_')
                    graph["edges"].append({
                        "source": source.get("id", ""),
                        "target": concept_id,
                        "relation": "discusses"
                    })
                    edges_added += 1
            
            # If no edges added, connect to most common concept
            if edges_added == 0 and valid_concepts:
                concept_id = valid_concepts[0].lower().replace(' ', '_').replace('-', '_')
                graph["edges"].append({
                    "source": source.get("id", ""),
                    "target": concept_id,
                    "relation": "relates_to"
                })
        
        # Create concept-to-concept edges (co-occurrence in sources)
        concept_cooccurrence = defaultdict(set)
        for source in sources[:10]:
            text = (source.get('title', '') + ' ' + source.get('summary', '')).lower()
            source_concepts = []
            for concept in valid_concepts:
                if concept.lower() in text:
                    source_concepts.append(concept)
            
            # Link concepts that appear together
            for i, c1 in enumerate(source_concepts):
                for c2 in source_concepts[i+1:]:
                    c1_id = c1.lower().replace(' ', '_').replace('-', '_')
                    c2_id = c2.lower().replace(' ', '_').replace('-', '_')
                    edge_key = tuple(sorted([c1_id, c2_id]))
                    concept_cooccurrence[edge_key].add(source.get("id", ""))
        
        # Add concept-concept edges if they co-occur in multiple sources
        for (c1_id, c2_id), source_ids in concept_cooccurrence.items():
            if len(source_ids) >= 2:  # At least 2 sources mention both
                graph["edges"].append({
                    "source": c1_id,
                    "target": c2_id,
                    "relation": "related_to"
                    })
        
        graph["metadata"] = {
            "total_nodes": len(graph["nodes"]),
            "total_edges": len(graph["edges"]),
            "concept_nodes": len([n for n in graph["nodes"] if n["type"] == "concept"]),
            "source_nodes": len([n for n in graph["nodes"] if n["type"] == "source"]),
            "created_at": datetime.now().isoformat()
        }
        
        print(f"  ✓ Graph: {len(graph['nodes'])} nodes ({graph['metadata']['concept_nodes']} concepts, {graph['metadata']['source_nodes']} sources)")
        print(f"  ✓ Edges: {len(graph['edges'])} connections")
        return graph
    
    def synthesize(self, state: ResearchState) -> Dict:
        """Main synthesis workflow"""
        print("\n🧬 SYNTHESIS AGENT")
        print("="*70)
        
        sources = state["validated_sources"]
        query = state["query"]
        
        if not sources:
            return {
                "knowledge_graph": {},
                "research_gaps": [],
                "consensus_findings": [],
                "contradictions": [],
                "key_concepts": []
            }
        
        print("\n[Phase 1] Extracting concepts...")
        concepts = self.extract_key_concepts(sources)
        
        print("\n[Phase 2] Finding consensus...")
        consensus = self.find_consensus(sources, query)
        
        print("\n[Phase 3] Detecting contradictions...")
        contradictions = self.detect_contradictions(sources)
        if contradictions:
            print(f"  ⚠ Found {len(contradictions)} contradictions")
        
        print("\n[Phase 4] Identifying gaps...")
        gaps = self.identify_research_gaps(sources, concepts)
        
        print("\n[Phase 5] Building knowledge graph...")
        kg = self.build_knowledge_graph(sources, concepts)
        
        print(f"\n✓ Synthesis complete")
        
        return {
            "key_concepts": concepts,
            "consensus_findings": consensus,
            "contradictions": contradictions,
            "research_gaps": gaps,
            "knowledge_graph": kg,
            "conflicts_detected": contradictions
        }


# ============================================================================
# PART 6: REPORTER AGENT (NEW!)
# ============================================================================

class ReporterAgent:
    """Generate comprehensive reports with citations"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_executive_summary(self, state: ResearchState) -> str:
        """Create 3-4 sentence executive summary"""
        try:
            consensus = state.get("consensus_findings", [])[:5]
            gaps = state.get("research_gaps", [])[:3]
            
            prompt = f"""Create a 3-4 sentence executive summary.

Query: {state['query']}
Key Findings: {', '.join(consensus)}
Research Gaps: {', '.join([g.get('gap', '') for g in gaps])}

Be concise and impactful."""

            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            return "Executive summary generation failed."
    
    def generate_detailed_report(self, state: ResearchState) -> str:
        """Generate comprehensive report"""
        try:
            sources = state["validated_sources"][:10]
            consensus = state.get("consensus_findings", [])
            gaps = state.get("research_gaps", [])
            contradictions = state.get("contradictions", [])
            concepts = state.get("key_concepts", [])
            
            # Build context
            sources_text = "\n".join([
                f"[{i+1}] {s.get('title', '')} - {s.get('summary', '')[:200]}"
                for i, s in enumerate(sources)
            ])
            
            prompt = f"""Generate a comprehensive research report.

QUERY: {state['query']}

SOURCES:
{sources_text}

CONSENSUS FINDINGS:
{chr(10).join(['- ' + f for f in consensus])}

RESEARCH GAPS:
{chr(10).join(['- ' + g.get('gap', '') for g in gaps])}

KEY CONCEPTS: {', '.join(concepts[:10])}

Create a structured report with:
1. **Executive Summary**
2. **Key Findings** (synthesize consensus)
3. **Notable Insights** (important details)
4. **Research Gaps & Future Directions**
5. **Conflicting Evidence** (if any)
6. **Conclusion**

Use professional academic language. Be analytical."""

            response = self.llm.invoke(prompt)
            
            # Add source citations
            report = response.content
            
            # Add citations section
            citations_section = "\n\n## SOURCES\n\n"
            for i, s in enumerate(sources, 1):
                citations_section += f"[{i}] {s.get('title', 'Unknown')}\n"
                if s.get('authors'):
                    citations_section += f"    Authors: {', '.join(s.get('authors', [])[:3])}\n"
                citations_section += f"    URL: {s.get('url', 'N/A')}\n"
                citations_section += f"    Source: {s.get('source_type', 'Unknown')}\n\n"
            
            report += citations_section
            
            return report
        except Exception as e:
            return f"Detailed report generation failed: {e}"
    
    def create_citation_map(self, sources: List[Dict]) -> Dict:
        """Build citation relationship map"""
        citation_map = {
            "total_sources": len(sources),
            "by_type": defaultdict(int),
            "by_year": defaultdict(int),
            "top_cited": [],
            "citation_network": []
        }
        
        for source in sources:
            source_type = source.get("source_type", "unknown")
            citation_map["by_type"][source_type] += 1
            
            # Extract year
            try:
                pub_date = str(source.get("published", ""))
                year = pub_date.split('-')[0]
                citation_map["by_year"][year] += 1
            except:
                pass
        
        # Top cited sources
        cited_sources = [(s.get("title", ""), s.get("citation_count", 0)) 
                        for s in sources if s.get("citation_count", 0) > 0]
        cited_sources.sort(key=lambda x: x[1], reverse=True)
        citation_map["top_cited"] = cited_sources[:10]
        
        return dict(citation_map)
    
    def generate_visualizations(self, state: ResearchState) -> List[str]:
        """Generate visualization metadata"""
        visualizations = []
        
        # Knowledge graph viz
        kg = state.get("knowledge_graph", {})
        if kg.get("nodes"):
            visualizations.append({
                "type": "knowledge_graph",
                "title": "Research Knowledge Graph",
                "description": f"{len(kg.get('nodes', []))} concepts interconnected",
                "data": kg
            })
        
        # Citation map viz
        citation_map = state.get("citation_map", {})
        if citation_map:
            visualizations.append({
                "type": "citation_distribution",
                "title": "Source Distribution",
                "data": citation_map
            })
        
        # Quality scores viz
        scores = state.get("validation_scores", [])
        if scores:
            visualizations.append({
                "type": "quality_scores",
                "title": "Source Quality Scores",
                "data": [{"source": s["source_title"][:40], "score": s["credibility_score"]} 
                        for s in scores[:10]]
            })
        
        return visualizations
    
    def report(self, state: ResearchState) -> Dict:
        """Main reporting workflow"""
        print("\n📊 REPORTER AGENT")
        print("="*70)
        
        print("\n[Phase 1] Executive Summary...")
        exec_summary = self.generate_executive_summary(state)
        
        print("\n[Phase 2] Detailed Report...")
        detailed_report = self.generate_detailed_report(state)
        
        print("\n[Phase 3] Citation Mapping...")
        citation_map = self.create_citation_map(state["validated_sources"])
        
        print("\n[Phase 4] Visualizations...")
        visualizations = self.generate_visualizations(state)
        
        print(f"\n✓ Report generated: {len(detailed_report)} chars")
        print(f"  Citations: {citation_map['total_sources']} sources")
        print(f"  Visualizations: {len(visualizations)} prepared")
        
        return {
            "executive_summary": exec_summary,
            "detailed_report": detailed_report,
            "citation_map": citation_map,
            "visualizations": visualizations
        }


# ============================================================================
# PART 7: MONITORING AGENT (NEW!)
# ============================================================================

class MonitoringAgent:
    """Monitor for new research and emerging trends"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_alert_triggers(self, query: str, concepts: List[str]) -> List[Dict]:
        """Define monitoring triggers"""
        triggers = []
        
        # Keyword-based triggers
        for concept in concepts[:8]:
            triggers.append({
                "type": "keyword_match",
                "keyword": concept,
                "priority": "high" if len(concept.split()) > 1 else "medium",
                "action": "notify"
            })
        
        # Citation threshold trigger
        triggers.append({
            "type": "citation_threshold",
            "threshold": 50,
            "priority": "high",
            "action": "alert"
        })
        
        # Recency trigger
        triggers.append({
            "type": "new_publication",
            "timeframe_days": 7,
            "priority": "medium",
            "action": "daily_digest"
        })
        
        return triggers
    
    def analyze_trends(self, sources: List[Dict], concepts: List[str]) -> Dict:
        """Analyze research trends"""
        trend_analysis = {
            "emerging_topics": [],
            "publication_velocity": {},
            "citation_trends": {},
            "author_networks": []
        }
        
        # Emerging topics (concepts appearing frequently)
        concept_counts = Counter()
        for source in sources:
            text = (source.get('title', '') + ' ' + source.get('summary', '')).lower()
            for concept in concepts:
                if concept.lower() in text:
                    concept_counts[concept] += 1
        
        trend_analysis["emerging_topics"] = [
            {"topic": topic, "frequency": count, "trend": "rising"}
            for topic, count in concept_counts.most_common(8)
        ]
        
        # Publication velocity (by year)
        year_counts = defaultdict(int)
        for source in sources:
            try:
                pub_date = str(source.get("published", ""))
                year = int(pub_date.split('-')[0])
                year_counts[year] += 1
            except:
                pass
        
        trend_analysis["publication_velocity"] = dict(year_counts)
        
        # Citation trends
        avg_citations_by_year = defaultdict(list)
        for source in sources:
            try:
                year = int(str(source.get("published", "")).split('-')[0])
                citations = source.get("citation_count", 0)
                avg_citations_by_year[year].append(citations)
            except:
                pass
        
        for year, citations in avg_citations_by_year.items():
            trend_analysis["citation_trends"][year] = sum(citations) / len(citations) if citations else 0
        
        # Top author networks
        author_freq = Counter()
        for source in sources:
            for author in source.get("authors", [])[:2]:
                author_freq[author] += 1
        
        trend_analysis["author_networks"] = [
            {"author": author, "papers": count}
            for author, count in author_freq.most_common(10)
        ]
        
        return trend_analysis
    
    def monitor(self, state: ResearchState) -> Dict:
        """Main monitoring workflow"""
        print("\n📡 MONITORING AGENT")
        print("="*70)
        
        query = state["query"]
        concepts = state.get("key_concepts", [])
        sources = state["validated_sources"]
        
        print("\n[Phase 1] Creating alert triggers...")
        triggers = self.create_alert_triggers(query, concepts)
        print(f"  ✓ {len(triggers)} triggers configured")
        
        print("\n[Phase 2] Analyzing trends...")
        trends = self.analyze_trends(sources, concepts)
        print(f"  ✓ Identified {len(trends['emerging_topics'])} emerging topics")
        print(f"  ✓ Tracked {len(trends['author_networks'])} key authors")
        
        print("\n[Phase 3] Setting up monitoring...")
        monitoring_config = {
            "enabled": True,
            "check_frequency": "daily",
            "notification_email": None,  # User would configure
            "last_check": datetime.now().isoformat()
        }
        
        print(f"\n✓ Monitoring configured")
        
        return {
            "monitoring_enabled": True,
            "alert_triggers": triggers,
            "trend_analysis": trends
        }


# ============================================================================
# PART 7B: MACHINE LEARNING AGENT (NEW!)
# ============================================================================

class MLAgent:
    """Machine Learning analysis using scikit-learn"""
    
    def __init__(self, llm):
        self.llm = llm
        self.vectorizer = None
        self.cluster_model = None
    
    def topic_modeling(self, sources: List[Dict], n_topics: int = 5) -> List[Dict]:
        """Discover hidden topics using LDA"""
        if not ML_AVAILABLE:
            return []
        
        try:
            # Prepare documents
            documents = [
                f"{s.get('title', '')} {s.get('summary', '')}"
                for s in sources
            ]
            
            if len(documents) < 3:
                return []
            
            # Vectorize
            vectorizer = CountVectorizer(max_features=100, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(documents)
            
            # LDA topic modeling
            n_topics = min(n_topics, len(documents))
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-8:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                topics.append({
                    "topic_id": topic_idx + 1,
                    "keywords": top_words[:5],
                    "all_keywords": top_words,
                    "weight": float(topic.sum())
                })
            
            print(f"  ✓ Discovered {len(topics)} topics using LDA")
            return topics
        except Exception as e:
            print(f"  ✗ Topic modeling failed: {e}")
            return []
    
    def cluster_papers(self, sources: List[Dict], n_clusters: int = 3) -> Dict:
        """Cluster papers using K-means"""
        if not ML_AVAILABLE:
            return {}
        
        try:
            # Prepare documents
            documents = [
                f"{s.get('title', '')} {s.get('summary', '')}"
                for s in sources
            ]
            
            if len(documents) < n_clusters:
                n_clusters = max(2, len(documents) // 2)
            
            # TF-IDF vectorization
            self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # K-means clustering
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self.cluster_model.fit_predict(tfidf_matrix)
            
            # Organize by clusters
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[int(label)].append({
                    "title": sources[idx].get("title", "")[:60],
                    "source_id": sources[idx].get("id", ""),
                    "summary": sources[idx].get("summary", "")[:100]
                })
            
            # Get cluster themes
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_themes = {}
            
            for cluster_id in range(n_clusters):
                center = self.cluster_model.cluster_centers_[cluster_id]
                top_terms_idx = center.argsort()[-5:][::-1]
                theme_words = [feature_names[i] for i in top_terms_idx]
                cluster_themes[cluster_id] = theme_words
            
            result = {
                "n_clusters": n_clusters,
                "clusters": dict(clusters),
                "cluster_themes": cluster_themes,
                "cluster_sizes": {k: len(v) for k, v in clusters.items()}
            }
            
            print(f"  ✓ Clustered papers into {n_clusters} groups")
            return result
        except Exception as e:
            print(f"  ✗ Clustering failed: {e}")
            return {}
    
    def predict_citations(self, sources: List[Dict]) -> List[Dict]:
        """Predict future citation potential using ML"""
        if not ML_AVAILABLE:
            return []
        
        try:
            predictions = []
            
            for source in sources:
                # Features for prediction
                current_citations = source.get("citation_count", 0)
                
                # Simple heuristic-based prediction
                try:
                    year = int(str(source.get("published", "2023")).split('-')[0])
                    age = datetime.now().year - year
                except:
                    age = 1
                
                # Title/summary quality indicators
                title_length = len(source.get("title", ""))
                summary_length = len(source.get("summary", ""))
                
                # Prediction formula (heuristic)
                # More citations + recent + good content = higher prediction
                recency_factor = max(1, 5 - age)
                quality_factor = min(2.0, (title_length + summary_length) / 500)
                
                predicted_citations = int(current_citations * (1 + 0.2 * recency_factor * quality_factor))
                growth_rate = ((predicted_citations - current_citations) / max(1, current_citations)) * 100
                
                predictions.append({
                    "title": source.get("title", "")[:60],
                    "current_citations": current_citations,
                    "predicted_citations_1yr": predicted_citations,
                    "predicted_growth_rate": round(growth_rate, 1),
                    "impact_category": "high" if predicted_citations > 100 else "medium" if predicted_citations > 20 else "low"
                })
            
            # Sort by predicted impact
            predictions.sort(key=lambda x: x["predicted_citations_1yr"], reverse=True)
            
            print(f"  ✓ Generated citation predictions for {len(predictions)} papers")
            return predictions[:10]
        except Exception as e:
            print(f"  ✗ Citation prediction failed: {e}")
            return []
    
    def ml_quality_scoring(self, sources: List[Dict]) -> List[Dict]:
        """Alternative quality scoring using ML features"""
        if not ML_AVAILABLE:
            return []
        
        try:
            scores = []
            
            for source in sources:
                # Extract features
                features = {
                    "title_length": len(source.get("title", "")),
                    "summary_length": len(source.get("summary", "")),
                    "has_citations": 1 if source.get("citation_count", 0) > 0 else 0,
                    "citation_log": np.log1p(source.get("citation_count", 0)),
                    "num_authors": len(source.get("authors", [])),
                    "is_recent": 1 if "2023" in str(source.get("published", "")) or "2024" in str(source.get("published", "")) else 0
                }
                
                # Compute ML-based quality score (0-100)
                ml_score = (
                    min(20, features["title_length"] / 5) +
                    min(30, features["summary_length"] / 50) +
                    min(30, features["citation_log"] * 5) +
                    min(10, features["num_authors"] * 2) +
                    features["is_recent"] * 10
                )
                
                scores.append({
                    "title": source.get("title", "")[:60],
                    "ml_score": round(ml_score, 1),
                    "features": features,
                    "grade": "A" if ml_score >= 80 else "B" if ml_score >= 65 else "C" if ml_score >= 50 else "D"
                })
            
            scores.sort(key=lambda x: x["ml_score"], reverse=True)
            
            print(f"  ✓ Computed ML quality scores for {len(scores)} papers")
            return scores
        except Exception as e:
            print(f"  ✗ ML quality scoring failed: {e}")
            return []
    
    def analyze(self, state: ResearchState) -> Dict:
        """Main ML analysis workflow"""
        print("\n🤖 MACHINE LEARNING AGENT")
        print("="*70)
        
        sources = state["validated_sources"]
        
        if not sources:
            return {
                "ml_topics": [],
                "paper_clusters": {},
                "citation_predictions": [],
                "ml_quality_scores": [],
                "ml_insights": {}
            }
        
        if not ML_AVAILABLE:
            print("  ⚠️  ML libraries not available. Skipping ML analysis.")
            return {
                "ml_topics": [],
                "paper_clusters": {},
                "citation_predictions": [],
                "ml_quality_scores": [],
                "ml_insights": {"status": "unavailable"}
            }
        
        print("\n[Phase 1] Topic Modeling (LDA)...")
        topics = self.topic_modeling(sources, n_topics=5)
        
        print("\n[Phase 2] Paper Clustering (K-means)...")
        clusters = self.cluster_papers(sources, n_clusters=3)
        
        print("\n[Phase 3] Citation Prediction...")
        predictions = self.predict_citations(sources)
        
        print("\n[Phase 4] ML Quality Scoring...")
        ml_scores = self.ml_quality_scoring(sources)
        
        # Generate insights
        insights = {
            "top_predicted_paper": predictions[0] if predictions else None,
            "dominant_cluster": max(clusters.get("cluster_sizes", {}).items(), key=lambda x: x[1])[0] if clusters.get("cluster_sizes") else None,
            "top_topic": topics[0] if topics else None,
            "average_ml_score": round(sum(s["ml_score"] for s in ml_scores) / len(ml_scores), 1) if ml_scores else 0
        }
        
        print(f"\n✓ ML Analysis complete")
        
        return {
            "ml_topics": topics,
            "paper_clusters": clusters,
            "citation_predictions": predictions,
            "ml_quality_scores": ml_scores,
            "ml_insights": insights
        }


# ============================================================================
# PART 8: ORCHESTRATOR (LangGraph)
# ============================================================================

def create_research_workflow(llm):
    """Create LangGraph workflow orchestrating all agents"""
    
    # Initialize all agents
    discovery_agent = DiscoveryAgent(llm)
    validation_agent = ValidationAgent(llm)
    rag_agent = RAGAgent(llm)
    synthesis_agent = SynthesisAgent(llm)
    ml_agent = MLAgent(llm)
    reporter_agent = ReporterAgent(llm)
    monitoring_agent = MonitoringAgent(llm)
    
    # Define workflow nodes
    def discovery_node(state: ResearchState) -> ResearchState:
        """Discovery phase"""
        state["current_agent"] = "discovery"
        result = discovery_agent.discover(state)
        state.update(result)
        return state
    
    def validation_node(state: ResearchState) -> ResearchState:
        """Validation phase"""
        state["current_agent"] = "validation"
        result = validation_agent.validate(state)
        state.update(result)
        return state
    
    def rag_node(state: ResearchState) -> ResearchState:
        """RAG integration phase"""
        state["current_agent"] = "rag"
        result = rag_agent.integrate(state)
        state.update(result)
        return state
    
    def synthesis_node(state: ResearchState) -> ResearchState:
        """Synthesis phase"""
        state["current_agent"] = "synthesis"
        result = synthesis_agent.synthesize(state)
        state.update(result)
        state["synthesis_ready"] = True
        return state
    
    def ml_node(state: ResearchState) -> ResearchState:
        """Machine Learning analysis phase"""
        state["current_agent"] = "ml"
        result = ml_agent.analyze(state)
        state.update(result)
        return state
    
    def reporter_node(state: ResearchState) -> ResearchState:
        """Reporting phase"""
        state["current_agent"] = "reporter"
        result = reporter_agent.report(state)
        state.update(result)
        return state
    
    def monitoring_node(state: ResearchState) -> ResearchState:
        """Monitoring setup phase"""
        state["current_agent"] = "monitoring"
        result = monitoring_agent.monitor(state)
        state.update(result)
        state["workflow_status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        return state
    
    # Build graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("discovery", discovery_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("ml", ml_node)
    workflow.add_node("reporter", reporter_node)
    workflow.add_node("monitoring", monitoring_node)
    
    # Define edges (workflow sequence)
    workflow.set_entry_point("discovery")
    workflow.add_edge("discovery", "validation")
    workflow.add_edge("validation", "rag")
    workflow.add_edge("rag", "synthesis")
    workflow.add_edge("synthesis", "ml")
    workflow.add_edge("ml", "reporter")
    workflow.add_edge("reporter", "monitoring")
    workflow.add_edge("monitoring", END)
    
    return workflow.compile()


# ============================================================================
# PART 9: VISUALIZATION HELPERS
# ============================================================================

def visualize_knowledge_graph(kg_data: Dict, output_file: str = "knowledge_graph.png"):
    """Visualize knowledge graph using networkx"""
    try:
        G = nx.Graph()
        
        # Add nodes
        for node in kg_data.get("nodes", []):
            node_type = node.get("type", "unknown")
            G.add_node(node["id"], label=node["label"], type=node_type)
        
        # Add edges
        edges_added = 0
        for edge in kg_data.get("edges", []):
            if edge["source"] in G and edge["target"] in G:
                G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))
                edges_added += 1
        
        print(f"  → Visualizing {len(G.nodes())} nodes and {edges_added} edges...")
        
        if len(G.nodes()) == 0:
            print("  ⚠️  No nodes to visualize")
            return None
        
        # Draw
        plt.figure(figsize=(20, 14))
        
        # Use spring layout with better parameters for connected graphs
        if edges_added > 0:
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        else:
            # If no edges, use circular layout
            pos = nx.circular_layout(G)
        
        # Separate node types
        concept_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'concept']
        source_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'source']
        
        # Draw concept nodes (larger, blue)
        if concept_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=concept_nodes, 
                                  node_color='#4CAF50', node_size=1500, 
                                  alpha=0.8, label='Concepts')
        
        # Draw source nodes (smaller, coral)
        if source_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, 
                                  node_color='#FF6B6B', node_size=1000, 
                                  alpha=0.7, label='Sources')
        
        # Draw edges with different styles
        concept_edges = [(u, v) for u, v, d in G.edges(data=True) 
                        if G.nodes[u].get('type') == 'concept' and G.nodes[v].get('type') == 'concept']
        source_concept_edges = [(u, v) for u, v in G.edges() 
                               if (u, v) not in concept_edges]
        
        if concept_edges:
            nx.draw_networkx_edges(G, pos, edgelist=concept_edges, 
                                 alpha=0.4, width=2, edge_color='#4CAF50', style='dashed')
        if source_concept_edges:
            nx.draw_networkx_edges(G, pos, edgelist=source_concept_edges, 
                                 alpha=0.3, width=1.5, edge_color='#666')
        
        # Labels
        labels = {}
        for n in G.nodes():
            label_text = G.nodes[n].get('label', '')[:40]
            # Add line break for long labels
            if len(label_text) > 20:
                words = label_text.split()
                mid = len(words) // 2
                label_text = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            labels[n] = label_text
        
        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold')
        
        plt.title("Research Knowledge Graph", fontsize=20, fontweight='bold', pad=20)
        plt.legend(loc='upper right', fontsize=12)
        
        # Add statistics text
        stats_text = f"Nodes: {len(G.nodes())} | Edges: {edges_added} | Concepts: {len(concept_nodes)} | Sources: {len(source_nodes)}"
        plt.text(0.5, 0.02, stats_text, ha='center', transform=plt.gcf().transFigure, 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Knowledge graph saved: {output_file}")
        print(f"    Stats: {len(concept_nodes)} concepts, {len(source_nodes)} sources, {edges_added} connections")
        return output_file
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_report(state: ResearchState):
    """Pretty print the final report"""
    print("\n" + "="*80)
    print("📋 FINAL RESEARCH REPORT")
    print("="*80)
    
    print("\n🎯 EXECUTIVE SUMMARY")
    print("-" * 80)
    print(state.get("executive_summary", "Not available"))
    
    print("\n\n📊 KEY METRICS")
    print("-" * 80)
    print(f"Sources Discovered: {state['discovery_metadata'].get('total_found', 0)}")
    print(f"Sources Validated: {len(state['validated_sources'])}")
    print(f"Average Quality Score: {state.get('source_quality_avg', 0):.1f}/100")
    print(f"Key Concepts: {len(state.get('key_concepts', []))}")
    print(f"Consensus Findings: {len(state.get('consensus_findings', []))}")
    print(f"Research Gaps: {len(state.get('research_gaps', []))}")
    print(f"Contradictions Detected: {len(state.get('contradictions', []))}")
    
    # ML Metrics
    ml_insights = state.get("ml_insights", {})
    if ml_insights and ml_insights.get("status") != "unavailable":
        print(f"ML Topics Discovered: {len(state.get('ml_topics', []))}")
        print(f"Paper Clusters: {state.get('paper_clusters', {}).get('n_clusters', 0)}")
        print(f"Average ML Score: {ml_insights.get('average_ml_score', 0)}/100")
    
    print("\n\n🔑 KEY CONCEPTS")
    print("-" * 80)
    concepts = state.get("key_concepts", [])[:15]
    for i, concept in enumerate(concepts, 1):
        print(f"{i}. {concept}")
    
    print("\n\n✅ CONSENSUS FINDINGS")
    print("-" * 80)
    for i, finding in enumerate(state.get("consensus_findings", []), 1):
        print(f"{i}. {finding}")
    
    if state.get("contradictions"):
        print("\n\n⚠️  CONFLICTING EVIDENCE")
        print("-" * 80)
        for contradiction in state.get("contradictions", []):
            print(f"• {contradiction.get('description', '')}")
    
    print("\n\n🔍 RESEARCH GAPS")
    print("-" * 80)
    for i, gap in enumerate(state.get("research_gaps", []), 1):
        print(f"{i}. {gap.get('gap', 'Unknown')}")
        print(f"   Why it matters: {gap.get('importance', 'N/A')}")
        print()
    
    print("\n\n📈 TREND ANALYSIS")
    print("-" * 80)
    trends = state.get("trend_analysis", {})
    emerging = trends.get("emerging_topics", [])[:5]
    for topic_info in emerging:
        print(f"• {topic_info['topic']}: {topic_info['frequency']} mentions ({topic_info['trend']})")
    
    # ML Analysis Section
    ml_insights = state.get("ml_insights", {})
    if ml_insights and ml_insights.get("status") != "unavailable":
        print("\n\n🤖 MACHINE LEARNING ANALYSIS")
        print("-" * 80)
        
        # ML Topics (LDA)
        ml_topics = state.get("ml_topics", [])
        if ml_topics:
            print("\n📌 Discovered Topics (LDA):")
            for topic in ml_topics[:3]:
                print(f"  Topic {topic['topic_id']}: {', '.join(topic['keywords'])}")
        
        # Paper Clusters
        clusters = state.get("paper_clusters", {})
        if clusters and clusters.get("clusters"):
            print(f"\n📊 Paper Clusters (K-means): {clusters.get('n_clusters', 0)} clusters")
            for cluster_id, theme in clusters.get("cluster_themes", {}).items():
                size = clusters.get("cluster_sizes", {}).get(cluster_id, 0)
                print(f"  Cluster {cluster_id} ({size} papers): {', '.join(theme)}")
        
        # Citation Predictions
        predictions = state.get("citation_predictions", [])
        if predictions:
            print(f"\n🔮 Top Citation Predictions:")
            for pred in predictions[:3]:
                print(f"  • {pred['title']}")
                print(f"    Current: {pred['current_citations']} → Predicted: {pred['predicted_citations_1yr']} citations")
                print(f"    Growth: {pred['predicted_growth_rate']}% | Impact: {pred['impact_category']}")
    
    print("\n\n📚 TOP SOURCES")
    print("-" * 80)
    citation_map = state.get("citation_map", {})
    for i, (title, citations) in enumerate(citation_map.get("top_cited", [])[:5], 1):
        print(f"{i}. {title[:70]}")
        print(f"   Citations: {citations}")
    
    print("\n\n🔗 SOURCE DISTRIBUTION")
    print("-" * 80)
    by_type = citation_map.get("by_type", {})
    for source_type, count in by_type.items():
        print(f"• {source_type}: {count} sources")
    
    print("\n\n" + "="*80)
    print("✅ WORKFLOW COMPLETE")
    print("="*80)
    print(f"Started: {state.get('started_at', 'N/A')}")
    print(f"Completed: {state.get('completed_at', 'N/A')}")
    print(f"Status: {state.get('workflow_status', 'unknown')}")


# ============================================================================
# PART 10: MAIN EXECUTION
# ============================================================================

def initialize_state(query: str, research_depth: str = "standard") -> ResearchState:
    """Initialize empty state"""
    return {
        "query": query,
        "research_depth": research_depth,
        "user_id": None,
        "raw_sources": [],
        "discovery_metadata": {},
        "validated_sources": [],
        "validation_scores": [],
        "credibility_report": {},
        "knowledge_graph": {},
        "research_gaps": [],
        "consensus_findings": [],
        "contradictions": [],
        "key_concepts": [],
        "vector_store_id": "",
        "embeddings_created": False,
        "rag_ready": False,
        "executive_summary": "",
        "detailed_report": "",
        "citation_map": {},
        "visualizations": [],
        "monitoring_enabled": False,
        "alert_triggers": [],
        "trend_analysis": {},
        "ml_topics": [],
        "paper_clusters": {},
        "citation_predictions": [],
        "ml_quality_scores": [],
        "ml_insights": {},
        "source_quality_avg": 0.0,
        "citation_counts": {},
        "conflicts_detected": [],
        "current_agent": "orchestrator",
        "workflow_status": "initialized",
        "errors": [],
        "synthesis_ready": False,
        "started_at": datetime.now().isoformat(),
        "completed_at": ""
    }


def run_research_pipeline(query: str, research_depth: str = "standard"):
    """
    Main entry point for the research intelligence platform
    
    Args:
        query: Research question or topic
        research_depth: 'quick' | 'standard' | 'deep'
    """
    print("\n" + "="*80)
    print("🚀 MULTI-AGENT RESEARCH INTELLIGENCE PLATFORM")
    print("="*80)
    print(f"\n📝 Query: {query}")
    print(f"🔬 Depth: {research_depth}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("\n❌ Error: GROQ_API_KEY not found in environment")
        print("Please set it in your .env file or environment variables")
        return None
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.3
    )
    
    # Create workflow
    print("\n🏗️  Building workflow graph...")
    workflow = create_research_workflow(llm)
    
    # Initialize state
    initial_state = initialize_state(query, research_depth)
    
    # Execute workflow
    print("\n🎬 Executing multi-agent workflow...")
    print("="*80)
    
    try:
        final_state = workflow.invoke(initial_state)
        
        # Print comprehensive report
        print_report(final_state)
        
        # Generate visualizations
        if final_state.get("knowledge_graph", {}).get("nodes"):
            print("\n📊 Generating visualizations...")
            visualize_knowledge_graph(
                final_state["knowledge_graph"], 
                output_file=f"kg_{hashlib.md5(query.encode()).hexdigest()[:8]}.png"
            )
        
        # Save detailed report to file
        report_filename = f"report_{hashlib.md5(query.encode()).hexdigest()[:8]}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED RESEARCH REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Query: {query}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(final_state.get("detailed_report", ""))
        
        print(f"\n💾 Detailed report saved: {report_filename}")
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# INTERACTIVE MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n" + "="*80)
    print("WELCOME TO MULTI-AGENT RESEARCH INTELLIGENCE PLATFORM")
    print("="*80)
    print("\nThis system will autonomously:")
    print("  - Search ArXiv, Semantic Scholar, and the web")
    print("  - Validate and score source credibility")
    print("  - Build a knowledge graph of concepts")
    print("  - Perform ML analysis (topic modeling, clustering, predictions)")
    print("  - Generate a comprehensive research report")
    print("  - Set up monitoring for new research")
    print("\n" + "-"*80)
    
    # Get user query
    print("\nEnter your research question or topic:")
    print("   Examples:")
    print("   - Large Language Models for code generation")
    print("   - Quantum computing in drug discovery")
    print("   - Transformer models for computer vision")
    print()
    query = input("Your query: ").strip()
    
    if not query:
        print("\nError: Query cannot be empty!")
        exit(1)
    
    # Get research depth
    print("\nSelect research depth:")
    print("   1. Quick   - Faster, fewer sources (~5 min)")
    print("   2. Standard - Balanced coverage (~8 min)")
    print("   3. Deep    - Comprehensive, more sources (~12 min)")
    print()
    depth_choice = input("Enter choice (1/2/3) [default: 2]: ").strip()
    
    depth_map = {
        "1": "quick",
        "2": "standard",
        "3": "deep",
        "": "standard"
    }
    
    research_depth = depth_map.get(depth_choice, "standard")
    
    # Confirm and run
    print("\n" + "-"*80)
    print(f"Query: {query}")
    print(f"Depth: {research_depth}")
    print("-"*80)
    
    confirm = input("\nProceed? (y/n) [default: y]: ").strip().lower()
    
    if confirm in ['n', 'no']:
        print("\nResearch cancelled.")
        exit(0)
    
    # Run the research pipeline
    result = run_research_pipeline(
        query=query,
        research_depth=research_depth
    )
    
    if result:
        print("\n" + "="*80)
        print("RESEARCH COMPLETE!")
        print("="*80)
        print("\nFiles generated:")
        print(f"   - Report: report_{hashlib.md5(query.encode()).hexdigest()[:8]}.txt")
        print(f"   - Knowledge Graph: kg_{hashlib.md5(query.encode()).hexdigest()[:8]}.png")
        print(f"   - Vector DB: ./chroma_db/")
        print("\nYou can now review the report and knowledge graph!")
    else:
        print("\nResearch failed. Please check errors above.")