# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from kavutiri_engine import Task, SimulationConfig, KavutiriSimulator

# --- CONFIGURATION ---
ST_TITLE = "🌱 Kavutiri Garden Project Planner (MC-PERT)"
st.set_page_config(page_title=ST_TITLE, layout="wide")

# --- SESSION STATE MANAGEMENT (Mock Multi-Project) ---
# Each key in st.session_state['projects'] is a project name
DEFAULT_PROJECT_NAME = "My First Garden"

def init_session_state():
    """Initialize all necessary session state variables."""
    if 'projects' not in st.session_state:
        st.session_state.projects: Dict[str, Dict[str, Any]] = {}
        
    if 'current_project_name' not in st.session_state:
        st.session_state.current_project_name = DEFAULT_PROJECT_NAME
    
    if st.session_state.current_project_name not in st.session_state.projects:
        load_default_project(st.session_state.current_project_name)

def load_default_project(project_name: str):
    """Load default tasks and config into a new project state."""
    default_tasks = [
        Task("A", "Clear garden area", [], 1.0, 2.0, 4.0, 2),
        Task("B", "Remove big rocks", ["A"], 0.5, 1.0, 2.5, 1),
        Task("C", "Till soil & compost", ["A"], 0.5, 1.0, 2.0, 1),
        Task("D", "Plant seeds", ["B", "C"], 0.3, 0.6, 1.2, 2),
        Task("E", "Water & mulch", ["D"], 0.5, 1.0, 2.0, 1),
    ]
    
    st.session_state.projects[project_name] = {
        'tasks': default_tasks,
        'config': SimulationConfig(),
        'results': None
    }

def get_current_project():
    """Returns the state dictionary for the currently selected project."""
    return st.session_state.projects[st.session_state.current_project_name]

# --- UI COMPONENTS ---

def display_task_table():
    """Display and allow editing of the current project's tasks."""
    st.subheader("📝 Project Tasks & Time Estimates (in Days/Units)")
    
    project = get_current_project()
    tasks_df = pd.DataFrame([t.to_dict() for t in project['tasks']])
    
    # Simple editor: uses Streamlit's data editor for quick changes
    edited_df = st.data_editor(
        tasks_df[['id', 'name', 'optimistic', 'most_likely', 'pessimistic', 'labor_required', 'predecessors']],
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "predecessors": st.column_config.ListColumn("Must Finish First (IDs)"),
            "labor_required": st.column_config.NumberColumn("Team Mates Needed", min_value=1)
        },
        hide_index=True,
        num_rows="dynamic", # Allows adding/deleting rows
    )
    
    if st.button("💾 Save Tasks & Update Project"):
        try:
            new_tasks = []
            for _, row in edited_df.iterrows():
                # Convert list-like predecessors string if necessary
                predecessors_list = row['predecessors']
                if isinstance(predecessors_list, str):
                    predecessors_list = [p.strip() for p in predecessors_list.split(",") if p.strip()]

                task = Task(
                    id=str(row['id']).strip(),
                    name=str(row['name']).strip(),
                    predecessors=predecessors_list,
                    optimistic=float(row['optimistic']),
                    most_likely=float(row['most_likely']),
                    pessimistic=float(row['pessimistic']),
                    labor_required=int(row['labor_required']),
                )
                task.validate()
                new_tasks.append(task)
            
            project['tasks'] = new_tasks
            # Reset results on task change
            project['results'] = None 
            st.success("Tasks updated successfully!")
            st.rerun() # Rerun to refresh the display
            
        except Exception as e:
            st.error(f"Error saving tasks: {e}")

def display_simulation_config():
    """Display and allow editing of the current project's config."""
    project = get_current_project()
    config = project['config']
    
    st.subheader("💡 Simulation & Resource Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 🧑‍🤝‍🧑 Team Settings")
        team_size = st.slider("Number of Team Mates (Labor Capacity):", 
                              min_value=1, max_value=10, 
                              value=config.team_size, key="team_size_slider")
        config.team_size = team_size

    with col2:
        st.write("#### ⚙️ Scenario Settings")
        
        n_iter = st.number_input("Scenarios to Test (Iterations):", 
                                 min_value=100, max_value=50000, step=1000, 
                                 value=config.n_iterations)
        config.n_iterations = n_iter
        
        # Simplified: Stick to Beta-PERT for this demo
        # distribution = st.selectbox("Uncertainty Type (Distribution):", 
        #                             ["beta-pert", "triangular", "normal"], 
        #                             index=["beta-pert", "triangular", "normal"].index(config.distribution))
        # config.distribution = distribution
        
        # lambda_param = st.number_input("Beta-PERT 'Weight' (Lambda):", value=config.lambda_param)
        # config.lambda_param = lambda_param

    st.markdown("---")
    if st.button("🚀 Run Simulation!", type="primary"):
        run_simulation_logic()

@st.cache_data(show_spinner="Running Monte Carlo Simulation...")
def run_simulation_cached(tasks_list: List[Task], config: SimulationConfig) -> Dict:
    """Wraps the simulation run in a Streamlit cache."""
    st.info(f"Running {config.n_iterations} scenarios with {config.team_size} team mates...")
    
    # Convert list of Task objects to a cache-hashable tuple of dicts
    tasks_for_hash = tuple(t.to_dict() for t in tasks_list) 
    
    try:
        simulator = KavutiriSimulator(tasks_list, config)
        summary = simulator.run_simulation()
        
        # Return all necessary data to be cached
        return {
            'summary': summary,
            'durations': simulator.results['durations'],
            'sensitivity_df': simulator.calculate_task_sensitivity().to_json(orient='split')
        }
    except Exception as e:
        st.error(f"Simulation Failed: {e}")
        return {}

def run_simulation_logic():
    """Logic to trigger the simulation and update state."""
    project = get_current_project()
    
    # Use the cached function to run the simulation
    results_cache = run_simulation_cached(project['tasks'], project['config'])
    
    if results_cache:
        project['results'] = results_cache
        st.success("Simulation complete! Check the results below.")
        # Clear the cache function call to ensure next run re-executes if inputs change
        # run_simulation_cached.clear() # Not strictly necessary if inputs change

def display_results():
    """Display the cached simulation results."""
    project = get_current_project()
    results = project['results']
    
    if not results:
        st.warning("Run the simulation first to see results.")
        return

    summary = results['summary']
    durations = results['durations']
    sensitivity_df = pd.read_json(results['sensitivity_df'], orient='split')
    
    st.header("📊 Simulation Results")
    
    # --- 1. Key Statistics (Column 1) ---
    st.subheader("📋 Key Timeline Statistics")
    
    stats_df = pd.DataFrame({
        "Metric": ["Average Time (Mean)", "Most Likely (P50)", "90% Safe Target (P90)"],
        "Time (Days)": [f"{summary['mean']:.2f}", f"{summary['p50']:.2f}", f"{summary['p90']:.2f}"]
    })
    st.dataframe(stats_df, hide_index=True)
    
    st.markdown(f"**Critical Path:** `{' → '.join(summary.get('most_common_critical_path', ['N/A']))}` (Longest in {summary.get('critical_path_frequency', 0):.1%})")

    st.markdown("---")
    
    # --- 2. Risk Chart (Column 2) ---
    st.subheader("📈 Timeline Risk Chart")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(durations, bins=30, density=True, alpha=0.7, color='green', edgecolor='white')
    
    # Cumulative distribution line (Risk curve)
    sorted_dur = np.sort(durations)
    cumulative = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur)
    ax2 = ax.twinx()
    ax2.plot(sorted_dur, cumulative * 100, linewidth=2, color='orange', label='P(finish by this date)')
    ax2.set_ylabel('Probability of Finishing (%)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0, 100)

    # Markers
    ax.axvline(summary['mean'], color='blue', linestyle='--', label=f"Avg: {summary['mean']:.2f}")
    ax.axvline(summary['p90'], color='red', linestyle='-', label=f"P90: {summary['p90']:.2f}")
    
    ax.set_xlabel('Project Duration (Days/Time Units)')
    ax.set_ylabel('Frequency of Outcome')
    ax.set_title('Project Timeline Risk Profile')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.5)
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # --- 3. Sensitivity Analysis ---
    st.subheader("⚠️ Task Risk Analysis (Sensitivity)")
    st.info("Tasks with high 'Crit. Index' or 'Impact' are the most critical to watch.")
    st.dataframe(sensitivity_df.style.format({
        'Crit. Index (%)': "{:.1f}%",
        'Correlation': "{:.3f}",
        'Impact': "{:.3f}"
    }), use_container_width=True)


# --- MAIN APP LOGIC ---
def main():
    """The main Streamlit application logic."""
    init_session_state()
    st.title(ST_TITLE)

    # --- Multi-Project Selector (Multi-User Mock) ---
    st.sidebar.header("📂 Project Selector")
    
    project_options = list(st.session_state.projects.keys())
    # Allow user to switch projects
    selected_project = st.sidebar.selectbox(
        "Select/Switch Project:",
        options=project_options,
        index=project_options.index(st.session_state.current_project_name) if st.session_state.current_project_name in project_options else 0
    )
    st.session_state.current_project_name = selected_project
    
    # New Project button
    new_name = st.sidebar.text_input("New Project Name:", "New Project")
    if st.sidebar.button("➕ Create New Project"):
        if new_name not in st.session_state.projects and new_name.strip():
            load_default_project(new_name)
            st.session_state.current_project_name = new_name
            st.rerun()
        else:
            st.sidebar.error("Invalid or existing project name.")
            
    st.sidebar.markdown("---")

    # --- Main Content Tabs ---
    tab1, tab2, tab3 = st.tabs(["1. Plan Tasks", "2. Run Simulation", "3. View Results"])

    with tab1:
        display_task_table()
    
    with tab2:
        display_simulation_config()
        
    with tab3:
        display_results()

if __name__ == '__main__':
    main()
