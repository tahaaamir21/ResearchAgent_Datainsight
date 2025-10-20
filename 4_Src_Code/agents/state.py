"""
Shared State Definition for Multi-Agent Research System
"""
from typing import List, Dict, TypedDict, Optional

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

