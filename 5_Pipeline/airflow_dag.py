# ============================================================================
# APACHE AIRFLOW DAG - RESEARCH INTELLIGENCE PIPELINE
# Orchestrates the multi-agent research workflow
# ============================================================================

"""
Apache Airflow DAG for Research Intelligence Platform

This DAG orchestrates the entire research workflow:
1. Discovery Agent
2. Validation Agent  
3. RAG Integration
4. Synthesis Agent
5. ML Agent
6. Reporter Agent
7. Monitoring Agent

Schedule: Daily at 2 AM
Retries: 2
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import datetime, timedelta
import json
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our main research system
from main import (
    DiscoveryAgent,
    ValidationAgent,
    RAGAgent,
    SynthesisAgent,
    MLAgent,
    ReporterAgent,
    MonitoringAgent,
    initialize_state,
    visualize_knowledge_graph
)
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default arguments for all tasks
default_args = {
    'owner': 'research_team',
    'depends_on_past': False,
    'email': ['research@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# DAG configuration
dag_config = {
    'dag_id': 'research_intelligence_pipeline',
    'default_args': default_args,
    'description': 'Multi-agent research intelligence system with ML analysis',
    'schedule_interval': '0 2 * * *',  # Daily at 2 AM
    'start_date': days_ago(1),
    'catchup': False,
    'tags': ['research', 'ai', 'ml', 'rag'],
    'max_active_runs': 1,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_llm():
    """Initialize LLM"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment")
    
    return ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

def load_state(**context):
    """Load or initialize state from XCom"""
    ti = context['ti']
    
    # Try to get query from Airflow Variables
    query = Variable.get("research_query", default_var="Latest advances in AI research")
    depth = Variable.get("research_depth", default_var="standard")
    
    state = initialize_state(query, depth)
    
    # Store in XCom for downstream tasks
    ti.xcom_push(key='research_state', value=state)
    
    print(f"✓ Initialized state for query: {query}")
    return state

def save_state(state, **context):
    """Save state to XCom"""
    ti = context['ti']
    ti.xcom_push(key='research_state', value=state)
    return state

# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def run_discovery(**context):
    """Task 1: Discovery Agent"""
    ti = context['ti']
    state = ti.xcom_pull(key='research_state', task_ids='initialize_state')
    
    print("\n" + "="*70)
    print("AIRFLOW TASK: Discovery Agent")
    print("="*70)
    
    llm = get_llm()
    agent = DiscoveryAgent(llm)
    result = agent.discover(state)
    
    state.update(result)
    save_state(state, **context)
    
    # Push metrics
    ti.xcom_push(key='sources_found', value=len(result['raw_sources']))
    
    print(f"✓ Discovery complete: {len(result['raw_sources'])} sources found")
    return len(result['raw_sources'])

def run_validation(**context):
    """Task 2: Validation Agent"""
    ti = context['ti']
    state = ti.xcom_pull(key='research_state', task_ids='initialize_state')
    
    print("\n" + "="*70)
    print("AIRFLOW TASK: Validation Agent")
    print("="*70)
    
    llm = get_llm()
    agent = ValidationAgent(llm)
    result = agent.validate(state)
    
    state.update(result)
    save_state(state, **context)
    
    # Push metrics
    ti.xcom_push(key='sources_validated', value=len(result['validated_sources']))
    ti.xcom_push(key='avg_quality', value=result['source_quality_avg'])
    
    print(f"✓ Validation complete: {len(result['validated_sources'])} sources validated")
    return len(result['validated_sources'])

def run_rag(**context):
    """Task 3: RAG Integration"""
    ti = context['ti']
    state = ti.xcom_pull(key='research_state', task_ids='initialize_state')
    
    print("\n" + "="*70)
    print("AIRFLOW TASK: RAG Agent")
    print("="*70)
    
    llm = get_llm()
    agent = RAGAgent(llm)
    result = agent.integrate(state)
    
    state.update(result)
    save_state(state, **context)
    
    # Push metrics
    ti.xcom_push(key='rag_ready', value=result['rag_ready'])
    ti.xcom_push(key='vector_store_id', value=result['vector_store_id'])
    
    print(f"✓ RAG complete: Vector store {result['vector_store_id']}")
    return result['rag_ready']

def run_synthesis(**context):
    """Task 4: Synthesis Agent"""
    ti = context['ti']
    state = ti.xcom_pull(key='research_state', task_ids='initialize_state')
    
    print("\n" + "="*70)
    print("AIRFLOW TASK: Synthesis Agent")
    print("="*70)
    
    llm = get_llm()
    agent = SynthesisAgent(llm)
    result = agent.synthesize(state)
    
    state.update(result)
    save_state(state, **context)
    
    # Push metrics
    ti.xcom_push(key='concepts_found', value=len(result['key_concepts']))
    ti.xcom_push(key='consensus_findings', value=len(result['consensus_findings']))
    ti.xcom_push(key='research_gaps', value=len(result['research_gaps']))
    
    print(f"✓ Synthesis complete: {len(result['key_concepts'])} concepts identified")
    return len(result['key_concepts'])

def run_ml_analysis(**context):
    """Task 5: ML Agent"""
    ti = context['ti']
    state = ti.xcom_pull(key='research_state', task_ids='initialize_state')
    
    print("\n" + "="*70)
    print("AIRFLOW TASK: ML Agent")
    print("="*70)
    
    llm = get_llm()
    agent = MLAgent(llm)
    result = agent.analyze(state)
    
    state.update(result)
    save_state(state, **context)
    
    # Push metrics
    ti.xcom_push(key='ml_topics', value=len(result['ml_topics']))
    ti.xcom_push(key='clusters', value=result['paper_clusters'].get('n_clusters', 0))
    
    print(f"✓ ML Analysis complete: {len(result['ml_topics'])} topics discovered")
    return len(result['ml_topics'])

def run_reporter(**context):
    """Task 6: Reporter Agent"""
    ti = context['ti']
    state = ti.xcom_pull(key='research_state', task_ids='initialize_state')
    
    print("\n" + "="*70)
    print("AIRFLOW TASK: Reporter Agent")
    print("="*70)
    
    llm = get_llm()
    agent = ReporterAgent(llm)
    result = agent.report(state)
    
    state.update(result)
    save_state(state, **context)
    
    # Save report to file
    import hashlib
    query = state['query']
    report_filename = f"/tmp/airflow_report_{hashlib.md5(query.encode()).hexdigest()[:8]}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AIRFLOW ORCHESTRATED RESEARCH REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Query: {query}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution Date: {context['ds']}\n\n")
        f.write(result['detailed_report'])
    
    # Visualize knowledge graph
    if state.get('knowledge_graph', {}).get('nodes'):
        kg_filename = f"/tmp/airflow_kg_{hashlib.md5(query.encode()).hexdigest()[:8]}.png"
        visualize_knowledge_graph(state['knowledge_graph'], output_file=kg_filename)
        ti.xcom_push(key='kg_file', value=kg_filename)
    
    ti.xcom_push(key='report_file', value=report_filename)
    
    print(f"✓ Report saved: {report_filename}")
    return report_filename

def run_monitoring(**context):
    """Task 7: Monitoring Agent"""
    ti = context['ti']
    state = ti.xcom_pull(key='research_state', task_ids='initialize_state')
    
    print("\n" + "="*70)
    print("AIRFLOW TASK: Monitoring Agent")
    print("="*70)
    
    llm = get_llm()
    agent = MonitoringAgent(llm)
    result = agent.monitor(state)
    
    state.update(result)
    state['workflow_status'] = 'completed'
    state['completed_at'] = datetime.now().isoformat()
    
    save_state(state, **context)
    
    # Push metrics
    ti.xcom_push(key='alert_triggers', value=len(result['alert_triggers']))
    ti.xcom_push(key='trends_found', value=len(result['trend_analysis'].get('emerging_topics', [])))
    
    print(f"✓ Monitoring configured: {len(result['alert_triggers'])} triggers set")
    return len(result['alert_triggers'])

def generate_summary_email(**context):
    """Generate email summary of the pipeline run"""
    ti = context['ti']
    
    # Collect metrics from all tasks
    metrics = {
        'sources_found': ti.xcom_pull(key='sources_found', task_ids='discovery'),
        'sources_validated': ti.xcom_pull(key='sources_validated', task_ids='validation'),
        'avg_quality': ti.xcom_pull(key='avg_quality', task_ids='validation'),
        'concepts_found': ti.xcom_pull(key='concepts_found', task_ids='synthesis'),
        'ml_topics': ti.xcom_pull(key='ml_topics', task_ids='ml_analysis'),
        'clusters': ti.xcom_pull(key='clusters', task_ids='ml_analysis'),
        'report_file': ti.xcom_pull(key='report_file', task_ids='reporter'),
        'kg_file': ti.xcom_pull(key='kg_file', task_ids='reporter'),
    }
    
    # Create summary
    summary = f"""
    Research Intelligence Pipeline Summary
    =======================================
    
    Execution Date: {context['ds']}
    Query: {Variable.get("research_query", default_var="N/A")}
    
    Metrics:
    - Sources Discovered: {metrics['sources_found']}
    - Sources Validated: {metrics['sources_validated']}
    - Average Quality Score: {metrics['avg_quality']:.1f}/100
    - Concepts Identified: {metrics['concepts_found']}
    - ML Topics Discovered: {metrics['ml_topics']}
    - Paper Clusters: {metrics['clusters']}
    
    Output Files:
    - Report: {metrics['report_file']}
    - Knowledge Graph: {metrics.get('kg_file', 'N/A')}
    
    Status: ✓ COMPLETED
    
    This is an automated email from the Research Intelligence Platform.
    """
    
    ti.xcom_push(key='email_body', value=summary)
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(summary)
    
    return summary

# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(**dag_config) as dag:
    
    # Task 0: Initialize state
    initialize = PythonOperator(
        task_id='initialize_state',
        python_callable=load_state,
        provide_context=True,
    )
    
    # Task 1: Discovery
    discovery = PythonOperator(
        task_id='discovery',
        python_callable=run_discovery,
        provide_context=True,
    )
    
    # Task 2: Validation
    validation = PythonOperator(
        task_id='validation',
        python_callable=run_validation,
        provide_context=True,
    )
    
    # Task 3: RAG Integration
    rag = PythonOperator(
        task_id='rag',
        python_callable=run_rag,
        provide_context=True,
    )
    
    # Task 4: Synthesis
    synthesis = PythonOperator(
        task_id='synthesis',
        python_callable=run_synthesis,
        provide_context=True,
    )
    
    # Task 5: ML Analysis
    ml_analysis = PythonOperator(
        task_id='ml_analysis',
        python_callable=run_ml_analysis,
        provide_context=True,
    )
    
    # Task 6: Reporter
    reporter = PythonOperator(
        task_id='reporter',
        python_callable=run_reporter,
        provide_context=True,
    )
    
    # Task 7: Monitoring
    monitoring = PythonOperator(
        task_id='monitoring',
        python_callable=run_monitoring,
        provide_context=True,
    )
    
    # Task 8: Generate summary
    summary = PythonOperator(
        task_id='generate_summary',
        python_callable=generate_summary_email,
        provide_context=True,
    )
    
    # Task 9: Cleanup (optional)
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command='echo "Pipeline completed successfully at $(date)"',
    )
    
    # Define task dependencies (DAG flow)
    initialize >> discovery >> validation >> rag >> synthesis >> ml_analysis >> reporter >> monitoring >> summary >> cleanup

# ============================================================================
# PARALLEL WORKFLOW (Alternative - for faster execution)
# ============================================================================

with DAG(
    dag_id='research_intelligence_pipeline_parallel',
    default_args=default_args,
    description='Parallel execution variant of research pipeline',
    schedule_interval='0 3 * * *',  # Daily at 3 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['research', 'ai', 'parallel'],
) as parallel_dag:
    
    init_parallel = PythonOperator(
        task_id='initialize',
        python_callable=load_state,
    )
    
    discover_parallel = PythonOperator(
        task_id='discovery',
        python_callable=run_discovery,
    )
    
    validate_parallel = PythonOperator(
        task_id='validation',
        python_callable=run_validation,
    )
    
    # These can run in parallel after validation
    rag_parallel = PythonOperator(
        task_id='rag',
        python_callable=run_rag,
    )
    
    synthesis_parallel = PythonOperator(
        task_id='synthesis',
        python_callable=run_synthesis,
    )
    
    # ML runs after synthesis
    ml_parallel = PythonOperator(
        task_id='ml_analysis',
        python_callable=run_ml_analysis,
    )
    
    # Reporter needs all upstream tasks
    report_parallel = PythonOperator(
        task_id='reporter',
        python_callable=run_reporter,
    )
    
    monitor_parallel = PythonOperator(
        task_id='monitoring',
        python_callable=run_monitoring,
    )
    
    # Parallel execution pattern
    init_parallel >> discover_parallel >> validate_parallel
    validate_parallel >> [rag_parallel, synthesis_parallel]
    [rag_parallel, synthesis_parallel] >> ml_parallel
    ml_parallel >> report_parallel >> monitor_parallel

