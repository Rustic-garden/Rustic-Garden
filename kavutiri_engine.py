#!/usr/bin/env python3
"""
Kavutiri Project Garden Planner & Simulator
A simplified Monte Carlo PERT tool for teaching project management to kids (11-16).
Focuses on "what-if" scenarios with team size and resource constraints.
Backend based on pert_montecarlo.py and pert_ascii.py logic.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import stats # Keep scipy.stats for potential future use or better stats calcs

# --- Configuration for a more playful, kid-friendly look (Windows 10 compatible) ---
FONT_FAMILY = "Segoe UI" # Common Windows font
HEADLINE_FONT = (FONT_FAMILY, 14, "bold")
NORMAL_FONT = (FONT_FAMILY, 10)
COLOR_PRIMARY = "#5cb85c" # Garden Green
COLOR_ACCENT = "#f0ad4e" # Sunny Orange
COLOR_BACKGROUND = "#f0f0f0"

# ----------------------------
# DATA MODELS (From pert_montecarlo.py - simplified for focus)
# ----------------------------
@dataclass
class Task:
    """Task definition with PERT estimates and basic resource needs"""
    id: str
    name: str
    predecessors: List[str]
    optimistic: float
    most_likely: float
    pessimistic: float
    labor_required: int = 1 # Number of 'team mates' needed
    fixed_cost: float = 0.0 # Keep for potential cost tracking

    def to_dict(self):
        return asdict(self)

    def validate(self):
        """Validate task parameters"""
        if not (self.optimistic <= self.most_likely <= self.pessimistic):
            raise ValueError(f"Task {self.id}: Must have O ≤ M ≤ P")
        if self.optimistic < 0:
            raise ValueError(f"Task {self.id}: Durations must be positive")
        return True

@dataclass
class SimulationConfig:
    """Simulation configuration"""
    n_iterations: int = 5000 # Reduced default for speed
    distribution: str = "beta-pert"
    lambda_param: float = 4.0
    random_seed: Optional[int] = 42
    team_size: int = 2 # Max simultaneous labor units (team mates)

# ----------------------------
# SIMULATION ENGINE (Adapted from PERTSimulator)
# ----------------------------
class KavutiriSimulator:
    """Core PERT simulation engine with resource constraint approximation"""

    def __init__(self, tasks: List[Task], config: SimulationConfig):
        self.tasks = tasks
        self.config = config
        self.task_dict = {t.id: t for t in tasks}
        self.results = None
        self.critical_path_frequency = {}

    def _beta_pert_sample(self, o: float, m: float, p: float, rng: np.random.Generator) -> float:
        """Sample from Beta-PERT distribution"""
        if p <= o: return o
        lamb = self.config.lambda_param
        alpha = 1 + lamb * (m - o) / (p - o)
        beta = 1 + lamb * (p - m) / (p - o)
        u = rng.beta(alpha, beta)
        return o + u * (p - o)

    def _sample_duration(self, task: Task, rng: np.random.Generator) -> float:
        """Sample task duration based on configured distribution"""
        o, m, p = task.optimistic, task.most_likely, task.pessimistic
        dist = self.config.distribution

        if dist == "beta-pert":
            return self._beta_pert_sample(o, m, p, rng)
        elif dist == "triangular":
            return rng.triangular(o, m, p)
        elif dist == "normal":
            mu = m
            sigma = (p - o) / 6 if p > o else 0.001
            sample = rng.normal(mu, sigma)
            return max(o, min(p, sample))
        else:
            return self._beta_pert_sample(o, m, p, rng)

    def _topological_order(self) -> List[Task]:
        """Return tasks in topological order; raise on cycles."""
        id_map = self.task_dict
        indeg = {t.id: 0 for t in self.tasks}
        adj = {t.id: [] for t in self.tasks}
        for t in self.tasks:
            for p in t.predecessors:
                if p in id_map:
                    adj[p].append(t.id)
                    indeg[t.id] += 1
        queue = [tid for tid, d in indeg.items() if d == 0]
        ordered_ids = []
        while queue:
            v = queue.pop(0)
            ordered_ids.append(v)
            for nbr in adj.get(v, []):
                indeg[nbr] -= 1
                if indeg[nbr] == 0:
                    queue.append(nbr)
        if len(ordered_ids) != len(self.tasks):
            raise ValueError("Cycle detected in task dependencies or missing predecessor IDs.")
        return [id_map[tid] for tid in ordered_ids]

    def _compute_schedule_cpm(self, durations: Dict[str, float], ordered_tasks: List[Task]) -> Dict:
        """Compute Early/Late Start/Finish (CPM)"""
        ES, EF = {}, {}
        for task in ordered_tasks:
            pred_finish = [EF[p] for p in task.predecessors if p in EF]
            ES[task.id] = max(pred_finish) if pred_finish else 0
            EF[task.id] = ES[task.id] + durations[task.id]

        project_end = max(EF.values()) if EF else 0.0
        
        LS, LF = {}, {}
        for task in reversed(ordered_tasks):
            successors = [t for t in ordered_tasks if task.id in t.predecessors]
            if successors:
                LF[task.id] = min([LS[s.id] for s in successors]) if successors else project_end
            else:
                LF[task.id] = project_end
            LS[task.id] = LF[task.id] - durations[task.id]

        slack = {tid: LS[tid] - ES[tid] for tid in ES}
        # Critical path tasks are those with near-zero slack
        critical_path = [tid for tid, s in slack.items() if abs(s) < 0.001]

        return {
            'duration': project_end,
            'critical_path': critical_path
        }

    def _simulate_one_run(self, rng: np.random.Generator, ordered_tasks: List[Task]) -> Dict:
        """Run a single Monte Carlo iteration with a simplified Resource-Constrained (RC) check"""
        
        # 1. Sample all durations (PERT)
        sampled_durations = {t.id: self._sample_duration(t, rng) for t in self.tasks}
        
        # 2. Compute CPM schedule (no resource constraint yet)
        cpm_schedule = self._compute_schedule_cpm(sampled_durations, ordered_tasks)
        
        # 3. Apply simplified resource-constrained delay for 'labor' (Team Size)
        # This is a basic heuristic and not a full RCPSP solver.
        
        # Get start/finish times from CPM as a baseline
        current_time = 0.0
        finish_times = {t.id: 0.0 for t in ordered_tasks}
        
        # Dictionary to track current project state: (Task ID, Task End Time)
        pending_tasks = []
        
        while len(finish_times) < len(ordered_tasks) or pending_tasks:
            
            # 3a. Advance time to the earliest finish of a pending task
            if pending_tasks:
                current_time = min(end_time for _, end_time in pending_tasks)
            
            # 3b. Free up resources from finished tasks
            finished_now = [tid for tid, end_time in pending_tasks if end_time <= current_time]
            pending_tasks = [(tid, end_time) for tid, end_time in pending_tasks if end_time > current_time]
            
            # Determine available labor capacity
            labor_in_use = sum(self.task_dict[tid].labor_required for tid, _ in pending_tasks)
            labor_available = self.config.team_size - labor_in_use
            
            # 3c. Find ready tasks (all predecessors finished, not started)
            ready_tasks = [
                t for t in ordered_tasks
                if t.id not in finish_times # Not finished
                and t.id not in [tid for tid, _ in pending_tasks] # Not started
                and all(p in finish_times for p in t.predecessors)
            ]
            
            # Sort ready tasks (e.g., by shortest duration first or criticality) - simple: use optimistic duration
            ready_tasks.sort(key=lambda t: t.optimistic)
            
            # 3d. Schedule ready tasks if resources are available
            for task in ready_tasks:
                required = task.labor_required
                if required <= labor_available:
                    start_time = max(current_time, *[finish_times[p] for p in task.predecessors if p in finish_times])
                    end_time = start_time + sampled_durations[task.id]
                    
                    pending_tasks.append((task.id, end_time))
                    finish_times[task.id] = end_time
                    
                    labor_available -= required
            
            # If no tasks could be started and no tasks are pending, it means we must have finished
            # or there is a gap. The outer while loop should handle project completion.
            if not pending_tasks and len(finish_times) < len(ordered_tasks):
                # Should not happen in a valid project network unless a task is 0 duration
                # or the resource logic is flawed. Break to avoid infinite loop.
                break 

            # Break if all tasks are finished
            if len(finish_times) == len(ordered_tasks):
                break

        rc_project_end = max(finish_times.values()) if finish_times else 0.0
        
        # Re-run CPM on the *sampled durations* and *new finish times* to find the critical path
        # (This is a simplification; the critical path in RCPS is defined differently)
        # For simplicity, we stick to the project end time and the original CPM CP
        
        return {
            'duration': rc_project_end,
            'cpm_duration': cpm_schedule['duration'],
            'critical_path': cpm_schedule['critical_path'] # Use CPM critical path for simplicity
        }

    def run_simulation(self, progress_callback=None):
        """Run Monte Carlo simulation"""
        rng = np.random.default_rng(self.config.random_seed)
        n = self.config.n_iterations
        ordered_tasks = self._topological_order() # Compute once

        durations = np.zeros(n)
        task_durations = {t.id: np.zeros(n) for t in self.tasks}
        critical_paths = []
        
        for i in range(n):
            run_result = self._simulate_one_run(rng, ordered_tasks)
            durations[i] = run_result['duration']
            critical_paths.append(tuple(sorted(run_result['critical_path'])))
            
            if progress_callback and i % 100 == 0:
                progress_callback(i / n * 100)
        
        from collections import Counter
        self.critical_path_frequency = Counter(critical_paths)
        
        self.results = {
            'durations': durations,
            'task_durations': task_durations,
            'critical_paths': critical_paths
        }
        
        return self.get_summary_statistics()

    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics from simulation results"""
        if self.results is None:
            raise ValueError("Run simulation first")
        
        durations = self.results['durations']
        
        stats_dict = {
            'iterations': self.config.n_iterations,
            'mean': float(np.mean(durations)),
            'median': float(np.median(durations)),
            'std_dev': float(np.std(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations)),
            'p10': float(np.percentile(durations, 10)),
            'p50': float(np.percentile(durations, 50)),
            'p90': float(np.percentile(durations, 90)),
        }
        
        most_common_cp = self.critical_path_frequency.most_common(1)
        if most_common_cp:
            cp, freq = most_common_cp[0]
            stats_dict['most_common_critical_path'] = list(cp)
            stats_dict['critical_path_frequency'] = freq / self.config.n_iterations
        
        return stats_dict
    
    def calculate_task_sensitivity(self) -> pd.DataFrame:
        """Calculate task sensitivity (Criticality Index and Correlation)"""
        if self.results is None:
            return pd.DataFrame()
        
        sensitivity_data = []
        project_durations = self.results['durations']
        
        for task in self.tasks:
            task_durs = self.results['task_durations'][task.id]
            # Correlation between task duration and total project duration
            correlation = np.corrcoef(task_durs, project_durations)[0, 1] if np.std(task_durs) > 0 else 0
            
            # Criticality Index: Frequency of task being on the critical path (from CPM model)
            crit_freq = sum(1 for cp in self.results['critical_paths'] if task.id in cp) / len(self.results['critical_paths']) * 100
            
            sensitivity_data.append({
                'Task ID': task.id,
                'Task Name': task.name,
                'Crit. Index (%)': crit_freq,
                'Correlation': correlation,
                'Impact': abs(correlation) # Simple measure of task impact on project total time
            })
        
        # Sort by impact
        return pd.DataFrame(sensitivity_data).sort_values('Impact', ascending=False)

# ----------------------------
# GUI APPLICATION
# ----------------------------
class KavutiriGUI:
    """Main GUI application for kids"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌱 Kavutiri Garden Project Planner")
        self.root.geometry("1000x800")
        self.root.configure(bg=COLOR_BACKGROUND)

        # Set theme and style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=COLOR_BACKGROUND)
        style.configure('TLabel', background=COLOR_BACKGROUND, font=NORMAL_FONT)
        style.configure('TCheckbutton', background=COLOR_BACKGROUND, font=NORMAL_FONT)
        style.configure('TButton', font=NORMAL_FONT, padding=5)
        style.configure("Accent.TButton", foreground=COLOR_BACKGROUND, background=COLOR_PRIMARY, font=(FONT_FAMILY, 10, "bold"))
        
        self.tasks = self.get_default_tasks()
        self.config = SimulationConfig()
        self.simulator = None
        self.results = None
        
        self.setup_ui()
        self.load_tasks_to_table()
        
    def get_default_tasks(self) -> List[Task]:
        """Load default kitchen garden tasks (Simplified)"""
        # Durations are in 'days' or 'time units'
        return [
            Task("A", "Clear garden area", [], 1.0, 2.0, 4.0, 2, 0),
            Task("B", "Remove big rocks & trash", ["A"], 0.5, 1.0, 2.5, 1, 0),
            Task("C", "Till soil & add compost", ["A"], 0.5, 1.0, 2.0, 1, 50),
            Task("D", "Plant seeds & small plants", ["B", "C"], 0.3, 0.6, 1.2, 2, 100),
            Task("E", "Water & add mulch", ["D"], 0.5, 1.0, 2.0, 1, 30),
            Task("F", "Build plant supports", ["C"], 1.0, 1.5, 3.0, 1, 75),
            Task("G", "Check tools & final setup", ["E", "F"], 0.3, 0.5, 1.0, 1, 0),
        ]
    
    def setup_ui(self):
        """Setup the user interface with tabs"""
        
        # Main Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Project Setup (Tasks & Resources)
        self.setup_tab = ttk.Frame(self.notebook, padding="10 10 10 10")
        self.notebook.add(self.setup_tab, text="1. Plan Your Tasks 📝")
        self.setup_tasks_panel(self.setup_tab)
        
        # Tab 2: Simulation Run (Config & Go!)
        self.run_tab = ttk.Frame(self.notebook, padding="10 10 10 10")
        self.notebook.add(self.run_tab, text="2. Run 'What-If' Simulation 💡")
        self.setup_run_panel(self.run_tab)
        
        # Tab 3: Results & Analysis
        self.results_tab = ttk.Frame(self.notebook, padding="10 10 10 10")
        self.notebook.add(self.results_tab, text="3. Check Your Results 📊")
        self.setup_results_panel(self.results_tab)
    
    # ------------------
    # TAB 1: SETUP
    # ------------------
    def setup_tasks_panel(self, parent):
        """Panel for managing tasks and team size"""
        
        # Task Management Frame
        tasks_frame = ttk.LabelFrame(parent, text="Garden Tasks & Times (Times in Days)", padding=10)
        tasks_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview (Task Table)
        tree_scroll = ttk.Scrollbar(tasks_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.task_tree = ttk.Treeview(tasks_frame, yscrollcommand=tree_scroll.set,
                                      columns=("ID", "Name", "O", "M", "P", "Labor", "Pred"),
                                      show="headings", height=12)
        tree_scroll.config(command=self.task_tree.yview)
        
        self.task_tree.heading("ID", text="ID")
        self.task_tree.heading("Name", text="Task Name")
        self.task_tree.heading("O", text="Best Case (O)")
        self.task_tree.heading("M", text="Most Likely (M)")
        self.task_tree.heading("P", text="Worst Case (P)")
        self.task_tree.heading("Labor", text="Team Mates Needed")
        self.task_tree.heading("Pred", text="Must Finish First")
        
        self.task_tree.column("ID", width=30, anchor=tk.CENTER)
        self.task_tree.column("Name", width=200)
        self.task_tree.column("O", width=80, anchor=tk.CENTER)
        self.task_tree.column("M", width=80, anchor=tk.CENTER)
        self.task_tree.column("P", width=80, anchor=tk.CENTER)
        self.task_tree.column("Labor", width=120, anchor=tk.CENTER)
        self.task_tree.column("Pred", width=150)
        
        self.task_tree.pack(fill=tk.BOTH, expand=True)
        
        # Task management buttons
        btn_frame = ttk.Frame(tasks_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="➕ Add Task", command=self.add_task, style="Accent.TButton").pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="✏️ Edit Task", command=self.edit_task).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="❌ Delete Task", command=self.delete_task).pack(side=tk.LEFT, padx=10)
        
        # Configuration/Resource Frame
        config_frame = ttk.Frame(parent, padding=10)
        config_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # Left side: Team Size
        team_frame = ttk.LabelFrame(config_frame, text="Team and Timeline", padding=10)
        team_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Label(team_frame, text="Number of Team Mates (Labor):", font=HEADLINE_FONT).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.team_size_var = tk.IntVar(value=self.config.team_size)
        ttk.Scale(team_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.team_size_var, length=200).grid(row=1, column=0, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(team_frame, textvariable=self.team_size_var, font=(FONT_FAMILY, 12, "bold")).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(team_frame, text="Notes & Reminders:", font=HEADLINE_FONT).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        self.reminders_text = tk.Text(team_frame, wrap=tk.WORD, height=5, width=40, font=NORMAL_FONT)
        self.reminders_text.insert(tk.END, "Example: We need to buy seeds by Day 3. Our target finish date is Day 10.")
        self.reminders_text.grid(row=3, column=0, columnspan=2, sticky=tk.EW)
        
        # Right side: Load/Save
        file_frame = ttk.LabelFrame(config_frame, text="Project Files", padding=10)
        file_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)
        
        ttk.Button(file_frame, text="📂 Load Project", command=self.load_project).pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="💾 Save Project", command=self.save_project).pack(fill=tk.X, pady=5)
        
    def load_tasks_to_table(self):
        """Load tasks into the treeview"""
        self.task_tree.delete(*self.task_tree.get_children())
        for task in self.tasks:
            pred_str = ", ".join(task.predecessors) if task.predecessors else "-"
            self.task_tree.insert("", tk.END, values=(
                task.id, task.name, task.optimistic, 
                task.most_likely, task.pessimistic, task.labor_required, pred_str
            ))

    # ------------------
    # TAB 2: RUN
    # ------------------
    def setup_run_panel(self, parent):
        """Panel for simulation configuration and running the simulation"""
        
        run_frame = ttk.LabelFrame(parent, text="Simulation Settings", padding=20)
        run_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Use a grid for neat alignment
        run_frame.columnconfigure(1, weight=1)
        
        # Iterations
        ttk.Label(run_frame, text="How many 'what-if' scenarios to test (Iterations):").grid(row=0, column=0, sticky=tk.W, pady=5, padx=10)
        self.iter_var = tk.IntVar(value=self.config.n_iterations)
        ttk.Entry(run_frame, textvariable=self.iter_var, width=15).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Distribution
        ttk.Label(run_frame, text="Uncertainty Type (Distribution):").grid(row=1, column=0, sticky=tk.W, pady=5, padx=10)
        self.dist_var = tk.StringVar(value=self.config.distribution)
        dist_combo = ttk.Combobox(run_frame, textvariable=self.dist_var, 
                                   values=["beta-pert", "triangular", "normal"], 
                                   state="readonly", width=15)
        dist_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Lambda (Beta-PERT)
        ttk.Label(run_frame, text="Beta-PERT 'Weight' (Lambda):").grid(row=2, column=0, sticky=tk.W, pady=5, padx=10)
        self.lambda_var = tk.DoubleVar(value=self.config.lambda_param)
        ttk.Entry(run_frame, textvariable=self.lambda_var, width=15).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Random Seed (for reproducibility)
        ttk.Label(run_frame, text="Random Seed (makes results the same each time):").grid(row=3, column=0, sticky=tk.W, pady=5, padx=10)
        self.seed_var = tk.IntVar(value=self.config.random_seed)
        ttk.Entry(run_frame, textvariable=self.seed_var, width=15).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Run button
        ttk.Button(run_frame, text="🚀 START SIMULATION! 🚀", 
                  command=self.run_simulation, style="Accent.TButton", 
                  width=30).grid(row=4, column=0, columnspan=2, pady=20)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(run_frame, variable=self.progress_var, 
                                            maximum=100, mode='determinate')
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(parent, text="What happens when we change the number of team mates or the best/worst case times?").pack(pady=10)

    # ------------------
    # TAB 3: RESULTS
    # ------------------
    def setup_results_panel(self, parent):
        """Panel for displaying simulation results and charts"""
        
        # Notebook for Stats and Sensitivity
        self.results_notebook = ttk.Notebook(parent)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 3a: Distribution Chart
        self.chart_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.chart_frame, text="Timeline Risk Chart 📈")
        
        # Tab 3b: Key Statistics
        self.stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.stats_frame, text="Key Statistics 📋")
        
        # Tab 3c: Sensitivity
        self.sensitivity_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.sensitivity_frame, text="Task Risk Analysis ⚠️")
        
        # Initialize matplotlib figure for Chart Tab
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add basic navigation for the chart
        toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame)
        toolbar.update()
        
        # Statistics text widget for Stats Tab
        stats_scroll = ttk.Scrollbar(self.stats_frame)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_text = tk.Text(self.stats_frame, wrap=tk.WORD, 
                                  yscrollcommand=stats_scroll.set, font=("Consolas", 10),
                                  bg='white', fg='black')
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        stats_scroll.config(command=self.stats_text.yview)

    # ------------------
    # RUN LOGIC
    # ------------------
    def run_simulation(self):
        """Run the Monte Carlo simulation in a separate thread"""
        try:
            # 1. Update configuration from GUI
            self.config.n_iterations = self.iter_var.get()
            self.config.distribution = self.dist_var.get()
            self.config.lambda_param = self.lambda_var.get()
            self.config.random_seed = self.seed_var.get()
            self.config.team_size = self.team_size_var.get()
            
            # 2. Validate tasks
            for task in self.tasks:
                task.validate()
            
            # 3. Create simulator
            self.simulator = KavutiriSimulator(self.tasks, self.config)
            
            # 4. Run in thread to keep GUI responsive
            def run():
                try:
                    self.results = self.simulator.run_simulation(self.update_progress)
                    self.root.after(0, self.display_results)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Simulation Error", str(e)))
                finally:
                    self.root.after(0, lambda: self.progress_var.set(0)) # Reset progress bar
            
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Configuration Error", str(e))
    
    def update_progress(self, value):
        """Update progress bar"""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    # ------------------
    # DISPLAY RESULTS
    # ------------------
    def display_results(self):
        """Display simulation results: plot, stats, sensitivity"""
        if self.results is None or self.simulator is None:
            return
        
        self.notebook.select(self.results_tab) # Switch to results tab
        
        # 1. Plotting (Chart Tab)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        durations = self.simulator.results['durations']
        
        # Histogram with a nice color
        ax.hist(durations, bins=30, density=True, alpha=0.7, color=COLOR_PRIMARY, edgecolor='white')
        
        # Cumulative distribution line (Risk curve)
        sorted_dur = np.sort(durations)
        cumulative = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur)
        ax2 = ax.twinx()
        ax2.plot(sorted_dur, cumulative * 100, linewidth=2, color=COLOR_ACCENT, label='P(finish by this date)')
        ax2.set_ylabel('Probability of Finishing (%)', color=COLOR_ACCENT, fontsize=11)
        ax2.tick_params(axis='y', labelcolor=COLOR_ACCENT)
        ax2.set_ylim(0, 100)

        # Add mean and P90 (90% chance of finishing by this time)
        mean_val = self.results['mean']
        p90 = self.results['p90']
        
        ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Average Time: {mean_val:.2f}')
        ax.axvline(p90, color='red', linestyle='-', linewidth=2, label=f'90% Safe Time: {p90:.2f}')
        
        ax.set_xlabel('Project Duration (Days/Time Units)', fontsize=11)
        ax.set_ylabel('Frequency of Outcome', fontsize=11)
        ax.set_title('Project Timeline Risk Profile', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.5)
        
        self.canvas.draw()
        
        # 2. Update statistics text (Stats Tab)
        self.update_statistics_display()
        
        # 3. Update sensitivity analysis (Sensitivity Tab)
        self.update_sensitivity_display()
        
        messagebox.showinfo("Success", "Simulation finished! Check the 'Results' tabs to see your risk.")
    
    def update_statistics_display(self):
        """Update statistics text widget"""
        self.stats_text.delete(1.0, tk.END)
        
        stats_text = "=" * 70 + "\n"
        stats_text += "GARDEN PROJECT TIMELINE SUMMARY\n"
        stats_text += "=" * 70 + "\n\n"
        
        stats_text += f"Project Setup:\n"
        stats_text += f"  • Team Mates (Labor):    {self.config.team_size}\n"
        stats_text += f"  • Scenarios Tested:      {self.results['iterations']:,}\n"
        stats_text += f"  • Uncertainty Model:     {self.config.distribution}\n\n"
        
        stats_text += f"Key Project Times (in Days):\n"
        stats_text += f"  • Shortest Possible Time: {self.results['min']:.2f}\n"
        stats_text += f"  • Longest Possible Time:  {self.results['max']:.2f}\n"
        stats_text += f"  • Average Time (Mean):    {self.results['mean']:.2f}\n"
        stats_text += f"  • Most Likely Time (P50): {self.results['p50']:.2f}\n"
        stats_text += f"  • SAFE TARGET (P90):      {self.results['p90']:.2f}\n"
        stats_text += f"    -> You have a 90% chance of finishing by this time.\n\n"
        
        if 'most_common_critical_path' in self.results:
            cp = self.results['most_common_critical_path']
            freq = self.results['critical_path_frequency']
            stats_text += f"Most Important Task Chain (Critical Path):\n"
            stats_text += f"  • Task IDs: {' → '.join(cp)}\n"
            stats_text += f"  • Frequency: {freq:.1%}\n"
            stats_text += f"    -> This path was the longest in {freq:.1%} of scenarios.\n\n"
        
        self.stats_text.insert(1.0, stats_text)
    
    def update_sensitivity_display(self):
        """Update sensitivity analysis display"""
        for widget in self.sensitivity_frame.winfo_children():
            widget.destroy()
        
        sensitivity_df = self.simulator.calculate_task_sensitivity()
        
        ttk.Label(self.sensitivity_frame, text="Which Tasks are the Riskiest?", font=HEADLINE_FONT).pack(pady=10)
        ttk.Label(self.sensitivity_frame, text="Tasks with high 'Impact' or 'Crit. Index' should be watched closely.").pack()
        
        # Treeview for sensitivity data
        sens_scroll = ttk.Scrollbar(self.sensitivity_frame)
        sens_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        sens_tree = ttk.Treeview(self.sensitivity_frame, yscrollcommand=sens_scroll.set,
                                 columns=list(sensitivity_df.columns), show="headings", height=10)
        sens_scroll.config(command=sens_tree.yview)
        
        for col in sensitivity_df.columns:
            sens_tree.heading(col, text=col)
            sens_tree.column(col, width=120)
        
        for _, row in sensitivity_df.iterrows():
            values = [f"{v:.1f}%" if col == 'Crit. Index (%)' else 
                      f"{v:.3f}" if isinstance(v, float) else str(v) 
                      for col, v in zip(sensitivity_df.columns, row)]
            sens_tree.insert("", tk.END, values=values)
        
        sens_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(self.sensitivity_frame, text="Export Data 💾", 
                  command=lambda: self.export_dataframe(sensitivity_df, "garden_task_risk.csv")
                  ).pack(pady=10)

    # ------------------
    # TASK MANAGEMENT
    # ------------------
    def add_task(self):
        """Add a new task"""
        dialog = TaskDialog(self.root, "➕ Add New Garden Task")
        if dialog.result:
            # Check for unique ID before adding
            if dialog.result.id in self.simulator.task_dict if self.simulator else [t.id for t in self.tasks]:
                messagebox.showerror("Error", f"Task ID '{dialog.result.id}' already exists!")
                return
            self.tasks.append(dialog.result)
            self.load_tasks_to_table()
    
    def edit_task(self):
        """Edit selected task"""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Selection Required", "Please select a task to edit")
            return
        item = self.task_tree.item(selection[0])
        task_id = item['values'][0]
        task = next((t for t in self.tasks if t.id == task_id), None)
        
        if task:
            dialog = TaskDialog(self.root, f"✏️ Edit Task: {task.name}", task)
            if dialog.result:
                idx = self.tasks.index(task)
                self.tasks[idx] = dialog.result
                self.load_tasks_to_table()
    
    def delete_task(self):
        """Delete selected task"""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Selection Required", "Please select a task to delete")
            return
        item = self.task_tree.item(selection[0])
        task_id = item['values'][0]
        
        dependents = [t.name for t in self.tasks if task_id in t.predecessors]
        if dependents:
            messagebox.showerror("Cannot Delete", 
                               f"Task {task_id} is needed before: {', '.join(dependents)}")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete task {task_id}?"):
            self.tasks = [t for t in self.tasks if t.id != task_id]
            self.load_tasks_to_table()
            
    # ------------------
    # FILE MANAGEMENT
    # ------------------
    def load_project(self):
        """Load project from JSON file"""
        filename = filedialog.askopenfilename(
            title="Load Garden Project", filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                self.tasks = [Task(**t) for t in data['tasks']]
                if 'config' in data:
                    cfg = data['config']
                    self.iter_var.set(cfg.get('n_iterations', 5000))
                    self.dist_var.set(cfg.get('distribution', 'beta-pert'))
                    self.lambda_var.set(cfg.get('lambda_param', 4.0))
                    self.seed_var.set(cfg.get('random_seed', 42))
                    self.team_size_var.set(cfg.get('team_size', 2))
                
                self.reminders_text.delete(1.0, tk.END)
                self.reminders_text.insert(tk.END, data.get('reminders', ''))
                
                self.load_tasks_to_table()
                messagebox.showinfo("Success", "Project loaded! Ready to run.")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load project:\n{str(e)}")
    
    def save_project(self):
        """Save project to JSON file"""
        filename = filedialog.asksaveasfilename(
            title="Save Garden Project", defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                data = {
                    'tasks': [t.to_dict() for t in self.tasks],
                    'config': {
                        'n_iterations': self.iter_var.get(),
                        'distribution': self.dist_var.get(),
                        'lambda_param': self.lambda_var.get(),
                        'random_seed': self.seed_var.get(),
                        'team_size': self.team_size_var.get()
                    },
                    'reminders': self.reminders_text.get(1.0, tk.END)
                }
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Success", "Project saved!")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save project:\n{str(e)}")
    
    def export_dataframe(self, df, default_name):
        """Export a dataframe to CSV"""
        filename = filedialog.asksaveasfilename(
            title="Export Data", defaultextension=".csv", initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")

class TaskDialog:
    """Dialog for adding/editing tasks (Simplified and Focused)"""
    
    def __init__(self, parent, title, task: Optional[Task] = None):
        self.result = None
        self.task = task
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.dialog.update_idletasks()
        self.dialog.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}") # Simple centering
        
        self.setup_ui()
        if task: self.load_task_data()
        self.dialog.wait_window()
    
    def setup_ui(self):
        """Setup dialog UI"""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)
        
        # Labels and Entry Widgets
        ttk.Label(main_frame, text="Task ID (e.g., A, B1):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.id_var = tk.StringVar()
        id_entry = ttk.Entry(main_frame, textvariable=self.id_var, width=30)
        id_entry.grid(row=0, column=1, pady=5, sticky=tk.EW)
        
        ttk.Label(main_frame, text="Task Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.name_var, width=30).grid(row=1, column=1, pady=5, sticky=tk.EW)
        
        ttk.Label(main_frame, text="Best Case Time (O):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.opt_var = tk.DoubleVar(value=1.0)
        ttk.Entry(main_frame, textvariable=self.opt_var, width=30).grid(row=2, column=1, pady=5, sticky=tk.EW)
        
        ttk.Label(main_frame, text="Most Likely Time (M):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.ml_var = tk.DoubleVar(value=2.0)
        ttk.Entry(main_frame, textvariable=self.ml_var, width=30).grid(row=3, column=1, pady=5, sticky=tk.EW)
        
        ttk.Label(main_frame, text="Worst Case Time (P):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.pess_var = tk.DoubleVar(value=4.0)
        ttk.Entry(main_frame, textvariable=self.pess_var, width=30).grid(row=4, column=1, pady=5, sticky=tk.EW)
        
        ttk.Label(main_frame, text="Team Mates Needed (Labor):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.res_var = tk.IntVar(value=1)
        ttk.Entry(main_frame, textvariable=self.res_var, width=30).grid(row=5, column=1, pady=5, sticky=tk.EW)
        
        ttk.Label(main_frame, text="Must Finish First (IDs, comma-separated):").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.pred_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.pred_var, width=30).grid(row=7, column=1, pady=5, sticky=tk.EW)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=8, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="✅ Save Task", command=self.save, style="Accent.TButton").pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Go Back", command=self.dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def load_task_data(self):
        """Load existing task data"""
        self.id_var.set(self.task.id)
        self.name_var.set(self.task.name)
        self.opt_var.set(self.task.optimistic)
        self.ml_var.set(self.task.most_likely)
        self.pess_var.set(self.task.pessimistic)
        self.pred_var.set(", ".join(self.task.predecessors))
        self.res_var.set(self.task.labor_required)
    
    def save(self):
        """Validate and save task"""
        try:
            task_id = self.id_var.get().strip()
            name = self.name_var.get().strip()
            pred_str = self.pred_var.get().strip()
            predecessors = [p.strip() for p in pred_str.split(",") if p.strip()]
            
            self.result = Task(
                id=task_id,
                name=name,
                predecessors=predecessors,
                optimistic=self.opt_var.get(),
                most_likely=self.ml_var.get(),
                pessimistic=self.pess_var.get(),
                labor_required=self.res_var.get(),
                fixed_cost=self.task.fixed_cost if self.task else 0.0 # Preserve or default cost
            )
            
            self.result.validate()
            self.dialog.destroy()
            
        except ValueError as e:
            messagebox.showerror("Oops! Problem Found", str(e), parent=self.dialog)


# ----------------------------
# MAIN APPLICATION ENTRY
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = KavutiriGUI(root)
    root.mainloop()
