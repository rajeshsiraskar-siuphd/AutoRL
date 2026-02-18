# ---------------------------------------------------------------------------------------
# AutoRL: CLI Version - Train and Evaluate Agents
# Author: Rajesh Siraskar
# CLI for training and evaluation of RL agents for Predictive Maintenance
# V.5.0: 17-Feb-2026: Checkpoint mechanism for crash recovery
# V.4.2: 17-Feb-2026: Grid search enabled for LR and Gamma hyperparameters
# Usage: 
# Training: train_agent.py -S SIT -A PPO,A2C,DQN,REINFORCE -E 300 -AM 0
# Grid Search: train_agent.py -S SIT -A PPO -E 200 -LR 0.001,0.0001 -G 0.95,0.99
# Evaluation: train_agent.py -V -S IEEE
# python train_agent.py -S SIT -A PPO,A2C -E 200 -AM 0
#   python train_agent.py -S IEEE -A PPO -E 300 -AM 1
#   python train_agent.py -V -S IEEE  # Evaluate IEEE models
#   python train_agent.py -D -S SIT   # Just list SIT models
#   python train_agent.py  # Uses all defaults: SIT, PPO, 200 episodes, no attention 
# ---------------------------------------------------------------------------------------
print('\n\n--------------------------------------------------------------------------')
print('AutoRL - Train RL agents for Predictive Maintenance')
print('--------------------------------------------------------------------------')
print('Author: Rajesh Siraskar')
print('Version: V.5.0 | 17-Feb-2026 -- Checkpoint mechanism for crash recovery\n\n')
print('CLI version that trains and evaluates RL agents on SIT and IEEE datasets')
print('--------------------------------------------------------------------------')
print('Usage:')
print('Training:   train_agent.py -S SIT -A PPO,A2C,DQN,REINFORCE -E 1e4 -AM 1')
print('Grid Search: train_agent.py -S SIT -A PPO -E 200 -LR 0.001,0.0001 -G 0.95,0.99')
print('Evaluation: train_agent.py -V -S IEEE')
print('--------------------------------------------------------------------------\n\n')

print(' - Loading libraries ...')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import glob
import sqlite3
try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    pass  # Will handle gracefully in plot creation function

# Import RL module
import rl_pdm

print('- Loaded.\n - Starting AutoRL CLI Pipeline ...\n')


# ======================================================================================
# CHECKPOINT DATABASE FUNCTIONS
# ======================================================================================

CHECKPOINT_DB = "checkpoint.db"

def init_checkpoint_db():
    """
    Initialize the checkpoint database and create tables if they don't exist.
    
    Returns:
        Connection object to the database
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    # Create batches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            schema TEXT NOT NULL,
            algorithms TEXT NOT NULL,
            episodes INTEGER NOT NULL,
            learning_rates TEXT NOT NULL,
            gammas TEXT NOT NULL,
            attention_mech INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create tasks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            task_order INTEGER NOT NULL,
            algo TEXT,
            training_file TEXT,
            attention_type TEXT,
            learning_rate REAL,
            gamma REAL,
            model_path TEXT,
            status TEXT NOT NULL,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
        )
    ''')
    
    conn.commit()
    return conn


def create_batch_id():
    """
    Generate a unique batch ID using timestamp.
    
    Returns:
        String batch ID in format: BATCH_YYYYMMDD_HHMMSS
    """
    now = datetime.now()
    return f"BATCH_{now.strftime('%Y%m%d_%H%M%S')}"


def create_batch_record(batch_id: str, schema: str, algos: List[str], 
                       episodes: int, lrs: List[float], gammas: List[float], 
                       attention_mech: int):
    """
    Create a new batch record in the database.
    
    Args:
        batch_id: Unique batch identifier
        schema: 'SIT' or 'IEEE'
        algos: List of algorithm names
        episodes: Number of training episodes
        lrs: List of learning rates
        gammas: List of gamma values
        attention_mech: 1 for attention, 0 for none
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO batches (batch_id, schema, algorithms, episodes, learning_rates, 
                           gammas, attention_mech, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        batch_id,
        schema,
        ','.join(algos),
        episodes,
        ','.join(map(str, lrs)),
        ','.join(map(str, gammas)),
        attention_mech,
        'Queued'
    ))
    
    conn.commit()
    conn.close()


def add_task_to_queue(batch_id: str, task_type: str, task_order: int, 
                     algo: str = None, training_file: str = None, 
                     attention_type: str = None, learning_rate: float = None, 
                     gamma: float = None):
    """
    Add a task to the queue in the database.
    
    Args:
        batch_id: Batch identifier
        task_type: 'TRAIN', 'EVAL', 'HEATMAP', 'REPORT'
        task_order: Execution order
        algo: Algorithm name (for TRAIN tasks)
        training_file: Training data filename
        attention_type: Attention mechanism type
        learning_rate: Learning rate value
        gamma: Gamma value
    
    Returns:
        task_id of the created task
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO tasks (batch_id, task_type, task_order, algo, training_file, 
                         attention_type, learning_rate, gamma, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        batch_id, task_type, task_order, algo, training_file, 
        attention_type, learning_rate, gamma, 'Queued'
    ))
    
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return task_id


def update_task_status(task_id: int, status: str, model_path: str = None, 
                      error_msg: str = None):
    """
    Update the status of a task.
    
    Args:
        task_id: Task identifier
        status: New status ('Queued', 'Training', 'Done', 'Failed')
        model_path: Path to saved model (optional)
        error_msg: Error message if failed (optional)
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE tasks 
        SET status = ?, model_path = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
        WHERE task_id = ?
    ''', (status, model_path, error_msg, task_id))
    
    conn.commit()
    conn.close()


def update_batch_status(batch_id: str, status: str):
    """
    Update the status of a batch.
    
    Args:
        batch_id: Batch identifier
        status: New status ('Queued', 'WIP', 'Done', 'Failed')
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE batches 
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE batch_id = ?
    ''', (status, batch_id))
    
    conn.commit()
    conn.close()


def get_incomplete_batch() -> Optional[Dict]:
    """
    Get the most recent incomplete batch (status = 'WIP' or 'Queued').
    
    Returns:
        Dictionary with batch information or None if no incomplete batch found
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM batches 
        WHERE status IN ('WIP', 'Queued')
        ORDER BY created_at DESC
        LIMIT 1
    ''')
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_pending_tasks(batch_id: str) -> List[Dict]:
    """
    Get all pending tasks for a batch (status != 'Done').
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        List of task dictionaries
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM tasks 
        WHERE batch_id = ? AND status != 'Done'
        ORDER BY task_order
    ''', (batch_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_completed_tasks(batch_id: str, task_type: str = None) -> List[Dict]:
    """
    Get all completed tasks for a batch.
    
    Args:
        batch_id: Batch identifier
        task_type: Optional filter by task type
    
    Returns:
        List of completed task dictionaries
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if task_type:
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE batch_id = ? AND status = 'Done' AND task_type = ?
            ORDER BY task_order
        ''', (batch_id, task_type))
    else:
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE batch_id = ? AND status = 'Done'
            ORDER BY task_order
        ''', (batch_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def mark_batch_complete(batch_id: str):
    """
    Mark a batch as complete if all tasks are done.
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        True if batch was marked complete, False otherwise
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    # Check if all tasks are done
    cursor.execute('''
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN status = 'Done' THEN 1 ELSE 0 END) as done
        FROM tasks
        WHERE batch_id = ?
    ''', (batch_id,))
    
    result = cursor.fetchone()
    total, done = result
    
    if total > 0 and total == done:
        update_batch_status(batch_id, 'Done')
        conn.close()
        return True
    
    conn.close()
    return False



def get_schema_files(schema: str, data_dir: str = "data") -> List[str]:
    """
    Get all CSV files for the specified schema (SIT or IEEE).
    
    Args:
        schema: 'SIT' or 'IEEE'
        data_dir: Path to data directory
    
    Returns:
        List of full file paths for the schema
    """
    # Check if subdirectory exists for strict organizational compliance
    subdir = os.path.join(data_dir, schema)
    if os.path.isdir(subdir):
        files = glob.glob(os.path.join(subdir, "*.csv"))
    else:
        # Fallback to flat directory
        schema_pattern = os.path.join(data_dir, f"{schema}_*.csv")
        files = glob.glob(schema_pattern)
        
    # Exclude test files and tiny files
    files = [f for f in files if not f.endswith(('_TEST.csv', '_tiny.csv', 'temp_sensor_data.csv'))]
    files.sort()
    return files


def train_agents(schema: str, algos: List[str], episodes: int, attention_mech: int, 
                 learning_rates: List[float], gammas: List[float], batch_id: str = None, 
                 resume_mode: bool = False) -> Dict:
    """
    Train agents on all data files of the specified schema with grid search over hyperparameters.
    
    Args:
        schema: 'SIT' or 'IEEE'
        algos: List of algorithm names
        episodes: Number of training episodes
        attention_mech: 1 to use attention mechanism, 0 for none
        learning_rates: List of learning rate values to try
        gammas: List of gamma values to try
        batch_id: Batch ID (required for both new and resumed batches)
        resume_mode: True if resuming an incomplete batch, False for new batch
    
    Returns:
        Dictionary mapping (algo, training_file, attention_type, lr, gamma) -> model_path
    """
    print(f"\n{'='*80}")
    print(f"TRAINING PHASE: Schema={schema}, Algos={algos}, Episodes={episodes}, Attention={attention_mech}")
    print(f"Grid Search: LRs={learning_rates}, Gammas={gammas}")
    if batch_id:
        mode_label = "RECOVERY MODE" if resume_mode else "NEW BATCH"
        print(f"Batch ID: {batch_id} ({mode_label})")
    print(f"{'='*80}\n")
    
    # Set global episodes in rl_pdm
    rl_pdm.EPISODES = episodes
    
    # Get training files
    training_files = get_schema_files(schema)
    if not training_files:
        print(f"ERROR: No training files found for schema {schema}")
        sys.exit(1)
    
    print(f"Training files for {schema}: {[Path(f).stem for f in training_files]}\n")
    
    trained_models = {}
    
    # Mapping from short forms (used in CLI) to rl_pdm attention type names
    attention_short_to_full = {
        'NW': 'NW',           # NadarayaWatson
        'TP': 'Temporal',     # Temporal
        'MH': 'MultiHead',    # MultiHead
        'SA': 'SelfAttn'      # SelfAttn
    }
    
    # Define attention types to train when AM=1
    attention_types = []
    if attention_mech == 1:
        # Train with 4 attention mechanisms: NW, TP, MH, SA (short forms for display)
        attention_types = ['NW', 'TP', 'MH', 'SA']
    else:
        # Train without attention
        attention_types = [None]
    
    if resume_mode:
        # Load pending tasks from database
        print(f"\n{'='*80}")
        print(f"RECOVERY MODE: Loading pending tasks from checkpoint database")
        print(f"{'='*80}\n")
        
        pending_tasks = get_pending_tasks(batch_id)
        completed_tasks = get_completed_tasks(batch_id, task_type='TRAIN')
        
        # Rebuild trained_models from completed tasks
        for task in completed_tasks:
            if task['model_path']:
                key = (task['algo'], task['training_file'], task['attention_type'], 
                      task['learning_rate'], task['gamma'])
                trained_models[key] = task['model_path']
        
        print(f"Found {len(completed_tasks)} completed training tasks")
        print(f"Found {len(pending_tasks)} pending training tasks")
        print(f"Resuming from pending tasks...\n")
        
        # Build training queue from pending tasks
        training_queue = []
        for task in pending_tasks:
            if task['task_type'] == 'TRAIN':
                queue_item = {
                    'task_id': task['task_id'],
                    'training_file': os.path.join('data', schema, task['training_file'] + '.csv') if not task['training_file'].endswith('.csv') else task['training_file'],
                    'training_filename': task['training_file'],
                    'algo': task['algo'],
                    'att_short': task['attention_type'],
                    'lr': task['learning_rate'],
                    'gm': task['gamma'],
                    'display': f"[RESUME] {task['algo']} ({task['attention_type']}) | File: {task['training_file']} | LR={task['learning_rate']} | Gamma={task['gamma']}"
                }
                training_queue.append(queue_item)
                print(f"  {queue_item['display']}")
        
    else:
        # Build training queue - all combinations
        print(f"\n{'='*80}")
        print(f"BUILDING TRAINING QUEUE")
        print(f"{'='*80}\n")
        
        training_queue = []
        task_order = 0
        
        for training_file in training_files:
            training_filename = Path(training_file).stem
            for algo in algos:
                for att_short in attention_types:
                    for lr in learning_rates:
                        for gm in gammas:
                            task_order += 1
                            att_label = f" ({att_short})" if att_short else ""
                            
                            # Add task to database
                            task_id = add_task_to_queue(
                                batch_id=batch_id,
                                task_type='TRAIN',
                                task_order=task_order,
                                algo=algo,
                                training_file=training_filename,
                                attention_type=att_short,
                                learning_rate=lr,
                                gamma=gm
                            )
                            
                            queue_item = {
                                'task_id': task_id,
                                'training_file': training_file,
                                'training_filename': training_filename,
                                'algo': algo,
                                'att_short': att_short,
                                'lr': lr,
                                'gm': gm,
                                'display': f"[{task_order:03d}] {algo}{att_label} | File: {training_filename} | LR={lr} | Gamma={gm}"
                            }
                            training_queue.append(queue_item)
                            print(queue_item['display'])
    
    print(f"\n{'='*80}")
    print(f"TOTAL TRAINING JOBS IN QUEUE: {len(training_queue)}")
    print(f"{'='*80}\n")
    
    # Train combinations - iterate through queue
    for idx, queue_item in enumerate(training_queue, 1):
        training_file = queue_item['training_file']
        training_filename = queue_item['training_filename']
        algo = queue_item['algo']
        att_short = queue_item['att_short']
        lr = queue_item['lr']
        gm = queue_item['gm']
        task_id = queue_item['task_id']
        
        # Map short form to full rl_pdm name
        att_full = attention_short_to_full.get(att_short) if att_short else None
        
        # Format attention label for display
        att_label = f" ({att_short})" if att_short else ""
        print(f"\n>>> [{idx:03d}/{len(training_queue)}] Training {algo} on {training_filename}{att_label} | LR={lr} | Gamma={gm}")
        
        # Update task status to 'Training'
        update_task_status(task_id, 'Training')
        
        try:
            # Train with specific hyperparameters
            result = rl_pdm.train_single_model(
                data_file=training_file,
                algo_name=algo,
                lr=lr,
                gm=gm,
                callback_func=None,  # No callback for CLI
                attention_type=att_full,  # Pass full name to rl_pdm
                data_filename=training_filename
            )
            
            if 'error' in result:
                print(f"  [x] Training failed: {result['error']}")
                update_task_status(task_id, 'Failed', error_msg=result['error'])
            else:
                model_path = result['model_path']
                trained_models[(algo, training_filename, att_short, lr, gm)] = model_path
                print(f"  [+] Model saved: {model_path}")
                print(f"    Weighted Score: {result['Weighted Score']:.4f}")
                
                # Update task status to 'Done' with model path
                update_task_status(task_id, 'Done', model_path=model_path)
                
        except Exception as e:
            error_msg = str(e)
            print(f"  [x] Training error: {error_msg}")
            update_task_status(task_id, 'Failed', error_msg=error_msg)

    
    print(f"\n{'='*80}")
    print(f"Training complete. Total models trained: {len(trained_models)}")
    print(f"{'='*80}\n")
    
    return trained_models


def discover_trained_models(schema: str, models_dir: str = "models") -> Dict:
    """
    Discover already trained models for the specified schema.
    
    Args:
        schema: 'SIT' or 'IEEE'
        models_dir: Directory containing models
    
    Returns:
        Dictionary mapping (algo, training_file, attention_type) -> model_path
    """
    print(f"\n{'='*80}")
    print(f"DISCOVERY PHASE: Looking for existing {schema} models in '{models_dir}'")
    print(f"{'='*80}\n")
    
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory '{models_dir}' not found.")
        return {}

    trained_models = {}
    
    # Mapping from full attention names (in metadata) to short forms (for display/keys)
    attention_full_to_short = {
        'NW': 'NW', 'NadarayaWatson': 'NW',
        'DL': 'DL', 'Simple': 'DL',
        'TP': 'TP', 'Temporal': 'TP', 
        'MH': 'MH', 'MultiHead': 'MH',
        'SA': 'SA', 'SelfAttn': 'SA',
        'HY': 'HY', 'Hybrid': 'HY'
    }

    count = 0
    # Walk through models dir
    try:
        for filename in os.listdir(models_dir):
            if filename.endswith(".zip"):
                try:
                    model_path = os.path.join(models_dir, filename)
                    model_base = os.path.splitext(model_path)[0] # remove .zip
                    metadata_path = model_base + "_metadata.json"
                    
                    algo = "Unknown"
                    training_file = "Unknown"
                    att_short = None
                    
                    # 1. Try Metadata first (Robust)
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                meta = json.load(f)
                                
                            algo = meta.get('algorithm', 'Unknown')
                            training_file = meta.get('training_data', 'Unknown')
                            att_full = meta.get('attention_mechanism')
                            
                            if att_full:
                                att_short = attention_full_to_short.get(att_full, att_full)
                            else:
                                att_short = None
                                
                            # Filter by schema in training_file
                            if schema not in training_file:
                                continue

                        except Exception as e:
                            print(f"  [!] Error reading metadata for {filename}: {e}")
                            continue
                    
                    else:
                        # 2. Fallback: Parse filename
                        # Format: Algo_TrainingData_... (e.g. PPO_IEEE_tiny_...)
                        if schema not in filename:
                            continue
                            
                        parts = filename.split('_')
                        if len(parts) < 3:
                             continue
                        
                        algo = parts[0]
                        # If schema is IEEE, training file starts with IEEE
                        training_file = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else f"{schema}_Unknown"
                        
                        # Check for attention suffix in filename
                        if "_NW" in filename: att_short = "NW"
                        elif "_DL" in filename: att_short = "DL"
                        elif "_TP" in filename: att_short = "TP"
                        elif "_MH" in filename: att_short = "MH"
                        elif "_SA" in filename: att_short = "SA"
                        elif "_HY" in filename: att_short = "HY"
                    
                    # Add to dict
                    key = (algo, training_file, att_short)
                    trained_models[key] = model_path
                    count += 1
                    
                    att_label = f" ({att_short})" if att_short else ""
                    print(f"  Found: {algo} - {training_file}{att_label}")

                except Exception as e:
                     print(f"  [!] Skipped {filename}: {e}")
                     continue

    except Exception as e:
        print(f"ERROR: Failed to list models directory: {e}")
        return {}

    print(f"\nFound {count} models matching schema '{schema}'.")
    return trained_models


def create_evaluation_plot(eval_result: Dict, model_filename: str, test_filename: str, results_dir: str = "results"):
    """
    Create and save an evaluation plot showing tool wear and replacements.
    
    Args:
        eval_result: Dictionary from adjusted_evaluate_model
        model_filename: Filename of the model
        test_filename: Filename of the test data
        results_dir: Directory to save plots
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        os.makedirs(results_dir, exist_ok=True)
        
        timesteps = eval_result['timesteps']
        tool_wear = eval_result['tool_wear']
        actions = eval_result['actions']
        wear_threshold = eval_result['wear_threshold']
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Add tool wear as blue line on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=tool_wear,
                name="Tool Wear",
                line=dict(color='#636EFA', width=3),
                mode='lines'
            ),
            secondary_y=False
        )
        
        # Add wear threshold as dotted line on primary y-axis
        fig.add_hline(
            y=wear_threshold,
            line_dash="dot",
            line_color="gray",
            annotation_text="Threshold",
            annotation_position="right",
            secondary_y=False
        )
        
        # Add IAR bounds if present
        IAR_lower = eval_result.get('IAR_lower', None)
        IAR_upper = eval_result.get('IAR_upper', None)
        if IAR_lower is not None and IAR_upper is not None:
            fig.add_hline(
                y=IAR_lower,
                line_dash="dot",
                line_color="teal",
                opacity=0.5,
                annotation_text="IAR Lower",
                annotation_position="right",
                secondary_y=False
            )
            fig.add_hline(
                y=IAR_upper,
                line_dash="dot",
                line_color="teal",
                opacity=0.5,
                annotation_text="IAR Upper",
                annotation_position="right",
                secondary_y=False
            )
        
        # Get replacement points
        model_override = eval_result.get('model_override', False)
        override_indices = eval_result.get('override_indices', [])
        
        # Find replacements
        replacement_timesteps = [t for t, a in zip(timesteps, actions) if a == 0]
        
        # Separate normal and override replacements
        if model_override and override_indices:
            override_replacements = [t for t in replacement_timesteps if t in override_indices]
            normal_replacements = [t for t in replacement_timesteps if t not in override_indices]
        else:
            normal_replacements = replacement_timesteps
            override_replacements = []
        
        # Add normal replacements as red markers
        if normal_replacements:
            normal_wear_values = [tool_wear[timesteps.index(t)] - 5 for t in normal_replacements]
            fig.add_trace(
                go.Scatter(
                    x=normal_replacements,
                    y=normal_wear_values,
                    name="Tool Replacement",
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='#EF553B',
                        symbol='diamond',
                        opacity=0.7
                    ),
                    showlegend=True
                ),
                secondary_y=False
            )
        
        # Add overridden replacements if any
        if override_replacements:
            override_wear_values = [tool_wear[timesteps.index(t)] - 5 for t in override_replacements]
            fig.add_trace(
                go.Scatter(
                    x=override_replacements,
                    y=override_wear_values,
                    name="Tool-Replacements",
                    mode='markers',
                    marker=dict(
                        size=12,
                        color="#EF553B",
                        symbol='diamond',
                        opacity=0.7
                    ),
                    showlegend=True
                ),
                secondary_y=False
            )
        
        # Update layout
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Tool Wear", secondary_y=False)
        fig.update_yaxes(title_text="Action", secondary_y=True, range=[0.0, 1.5], showticklabels=False, ticks="", showgrid=False)
        
        fig.update_layout(
            title=f"Model Evaluation: {model_filename}_{test_filename}",
            height=500,
            template="plotly_white",
            plot_bgcolor='#f0f2f6',
            hovermode='x unified'
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white', secondary_y=False)
        
        # Save plot
        plot_filename = f"{model_filename}_{test_filename}_Eval.png"
        plot_filepath = os.path.join(results_dir, plot_filename)
        
        fig.write_image(plot_filepath, width=1200, height=600)
        
        return plot_filepath
        
    except Exception as e:
        print(f"  [!] Could not create plot: {str(e)}")
        return None


def evaluate_agents(schema: str, trained_models: Dict) -> pd.DataFrame:
    """
    Evaluate all trained models on all data files of the schema.
    
    Args:
        schema: 'SIT' or 'IEEE'
        trained_models: Dictionary mapping (algo, training_file, attention_type, lr, gamma) -> model_path
    
    Returns:
        DataFrame with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION PHASE: Evaluating {len(trained_models)} model(s)")
    print(f"{'='*80}\n")
    
    # Get test files (same as training files for same schema)
    test_files = get_schema_files(schema)
    
    results = []
    
    for model_key, model_path in trained_models.items():
        # Unpack the model key - handle both old format (3 items) and new format (5 items)
        if len(model_key) == 5:
            algo, train_filename, att_short, lr, gm = model_key
        elif len(model_key) == 3:
            # Old format compatibility (from discovered models without hyperparameters in key)
            algo, train_filename, att_short = model_key
            lr, gm = None, None
        else:
            print(f"  [!] Unexpected model key format: {model_key}, skipping...")
            continue
        
        # Format model description with attention if present
        att_label = f" ({att_short})" if att_short else ""
        hyperparam_label = f" | LR={lr} | G={gm}" if lr is not None else ""
        print(f"\nEvaluating {algo}{att_label}{hyperparam_label} (trained on {train_filename}):")
        
        for test_file in test_files:
            test_filename = Path(test_file).stem
            
            try:
                # Evaluate model
                eval_result = rl_pdm.evaluate_trained_model(model_path, test_file, seed=42)
                
                # Check for errors
                if eval_result.get('error', False):
                    print(f"  [!] {test_filename}: Feature mismatch error")
                    continue
                
                # Determine if self-eval
                self_eval = 'Y' if train_filename == test_filename else 'N'

                # Build result row
                # Extract model filename from path
                model_filename = Path(model_path).stem
                
                # Map attention short form to full name
                attention_map = {
                    'MH': 'Multi-Head',
                    'NW': 'Nadaraya-Watson',
                    'SA': 'Self-Attention',
                    'TP': 'Temporal'
                }
                att_full_name = attention_map.get(att_short, att_short) if att_short else 'None'

                row = {
                    'Algorithm': f"{algo}",
                    'Attention Mechanism': att_full_name,
                    'Learning Rate': lr if lr is not None else "N/A",
                    'Gamma': gm if gm is not None else "N/A",
                    'Model File': model_filename,
                    'Training File': train_filename,
                    'Test File': test_filename,
                    'Self-eval': self_eval,
                    'Tool Usage %': eval_result.get('tool_usage_pct', 0.0) * 100 if eval_result.get('tool_usage_pct') else 0,
                    'Lambda': eval_result.get('lambda', 0),
                    'Threshold Violations': eval_result.get('threshold_violations', 0),
                    'T_wt': eval_result.get('T_wt'),
                    't_FR': eval_result.get('t_FR'),
                     'Model Override': 'Y' if eval_result.get('model_override', False) else 'N',
                    'Eval_Score': calculate_eval_score(eval_result)
                }
                
                results.append(row)
                print(f"  [+] {test_filename}: Lambda={row['Lambda']}, Violations={row['Threshold Violations']}, Score={row['Eval_Score']:.4f}")
                
                # Create and save evaluation plot
                plot_path = create_evaluation_plot(eval_result, model_filename, test_filename)
                if plot_path:
                    print(f"    Plot saved: {plot_path}")
                
            except Exception as e:
                print(f"  [x] {test_filename}: Evaluation error - {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete. Total evaluations: {len(results_df)}")
    print(f"{'='*80}\n")
    
    return results_df


def calculate_eval_score(eval_result: Dict) -> float:
    """
    Calculate weighted evaluation score based on metrics.
    
    Weights:
    - Tool Usage: 50%
    - Lambda: 30%
    - Violations: 20%
    
    Args:
        eval_result: Dictionary from adjusted_evaluate_model
    
    Returns:
        Float score between 0-1 (higher is better)
    """
    tool_usage_pct = eval_result.get('tool_usage_pct', 0) or 0
    lambda_metric = eval_result.get('lambda', 0) or 0
    violations = eval_result.get('threshold_violations', 0)
    
    # Normalize tool usage (0-100% is ideal)
    tool_usage_score = min(1.0, max(0.0, tool_usage_pct))
    
    # Normalize lambda (lower is better, 0 is best)
    # Lambda max is the wear threshold - IAR_LOWER
    IAR_LOWER = rl_pdm.WEAR_THRESHOLD * (1.0 - rl_pdm.IAR_RANGE)
    lambda_score = max(0.0, 1.0 - (abs(lambda_metric) / IAR_LOWER))
    
    # Normalize violations (lower is better, 0 is best)
    violation_score = max(0.0, 1.0 / (1.0 + violations))
    
    # Weighted combination
    eval_score = (
        0.50 * tool_usage_score +
        0.30 * lambda_score +
        0.20 * violation_score
    )
    
    return eval_score


def save_results(results_df: pd.DataFrame, schema: str, attention_mech: int, results_dir: str = "results") -> str:
    """
    Save evaluation results to CSV file.
    
    Args:
        results_df: DataFrame with results
        schema: 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag
        results_dir: Directory to save results
    
    Returns:
        Path to saved CSV file
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    att_label = "AM" if attention_mech else "NoAM"
    filename = f"Evaluation_Results_{schema}_{att_label}_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    
    # Save to CSV
    results_df.to_csv(filepath, index=False)
    print(f"\n✓ Results saved to: {filepath}")
    
    return filepath


def create_heatmaps(results_df: pd.DataFrame, schema: str, attention_mech: int, results_dir: str = "results"):
    """
    Create a single comprehensive heatmap showing evaluation scores for all agents.
    
    Args:
        results_df: DataFrame with evaluation results
        schema: 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag
        results_dir: Directory to save heatmap
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Pivot for heatmap: rows = ALL model files (all agents), cols = test files
    heatmap_data = results_df.pivot_table(
        index='Model File',
        columns='Test File',
        values='Eval_Score',
        aggfunc='first'
    )
    
    # Sort rows alphabetically for consistent ordering
    heatmap_data = heatmap_data.sort_index()
    
    # Create single comprehensive heatmap with larger figsize for all agents
    num_agents = len(heatmap_data)
    figsize_height = max(10, num_agents * 0.5)  # Scale height based on number of agents
    fig, ax = plt.subplots(figsize=(14, figsize_height))
    fig.patch.set_facecolor('#f0f2f6')
    ax.set_facecolor('#f0f2f6')
    
    # Use color scheme: Green (1.0), Yellow (0.5), Red (0.0)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Evaluation Score'},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title(f'Evaluation Score Heatmap\nSchema: {schema}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Test File', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model File', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels (test files) to be horizontal for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
    # Rotate y-axis labels (model files) to be horizontal for readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # Save single heatmap
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    att_label = "AM" if attention_mech else "NoAM"
    filename = f"Heatmap_AllAgents_{schema}_{att_label}_{timestamp}.png"
    filepath = os.path.join(results_dir, filename)
    
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    

def generate_analysis_report(results_df: pd.DataFrame, schema: str, attention_mech: int, results_dir: str = "results"):
    """
    Generate a comprehensive analysis report with statistical plots and error bars.
    
    Creates a detailed 4-panel summary:
    1. Overall Performance Matrix (Heatmap)
    2. Model Performance by Attention (Grouped Bar Chart)
    3. Algorithm Performance (Bar Chart with Std Dev)
    4. Attention Mechanism Performance (Bar Chart with Std Dev)
    
    Args:
        results_df: DataFrame with evaluation results
        schema: 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag
        results_dir: Directory to save report
    """
    print(f"\nGenerating comprehensive analysis report...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if 'Attention Mechanism' exists (it should, based on previous steps)
    if 'Attention Mechanism' not in results_df.columns:
        # Fallback if column missing (e.g. older version)
        results_df['Attention Mechanism'] = 'None'
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Grid: 2 rows, 2 columns
    # Row 1, Col 1: Overall Heatmap
    # Row 1, Col 2: Model Performance (Grouped)
    # Row 2, Col 1: Algo Performance
    # Row 2, Col 2: Attention Performance
    # Increased padding to prevent overlap
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.3)
    
    axs = []
    axs.append(fig.add_subplot(gs[0, 0])) # ax0: Heatmap
    axs.append(fig.add_subplot(gs[0, 1])) # ax1: Model Perf
    axs.append(fig.add_subplot(gs[1, 0])) # ax2: Algo Perf
    axs.append(fig.add_subplot(gs[1, 1])) # ax3: Attn Perf
    
    # Calculate dynamic axis limits
    min_score = results_df['Eval_Score'].min()
    max_score = results_df['Eval_Score'].max()
    
    # Heuristic: Lower bound = min_score - (range * 0.2) or min_score - 0.05
    # Ensure it doesn't go below 0
    # If all scores are high (e.g. > 0.9), we want to zoom in
    padding = max(0.05, (max_score - min_score) * 0.2)
    lower_bound = max(0.0, min_score - padding)
    # Round down to nearest 0.05
    lower_bound = np.floor(lower_bound * 20) / 20.0
    
    upper_bound = 1.0
    
    print(f"  Axis limits: [{lower_bound:.2f}, {upper_bound:.2f}] (Min score: {min_score:.4f})")

    # --- Panel A: Overall Performance (Heatmap) ---
    ax = axs[0]
    try:
        # Aggregation
        pivot_data = results_df.pivot_table(
            index='Model', 
            columns='Attention Mechanism', 
            values='Eval_Score', 
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn', 
            vmin=lower_bound, 
            vmax=upper_bound, 
            cbar_kws={'label': 'Avg Eval Score'}, 
            ax=ax,
            linewidths=1,
            linecolor='white'
        )
        ax.set_title('A. OVERALL PERFORMANCE (Avg Score)', fontsize=14, fontweight='bold', loc='left', pad=15)
        ax.set_xlabel('Attention Mechanism', fontweight='bold')
        ax.set_ylabel('Algorithm', fontweight='bold')
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate heatmap: {e}", ha='center')

    # --- Panel B: Model Performance (Grouped Bar with Error Bars) ---
    ax = axs[1]
    try:
        # Horizontal Grouped Bar Chart
        # y = Attention, x = Score, hue = Algo
        sns.barplot(
            data=results_df,
            x='Eval_Score',
            y='Attention Mechanism',
            hue='Model',
            errorbar='sd',  # Standard Deviation error bars
            palette='viridis',
            ax=ax,
            orient='h',
            capsize=0.1
        )
        ax.set_title('B. MODEL PERFORMANCE (by Attention & Algo)', fontsize=14, fontweight='bold', loc='left', pad=15)
        ax.set_xlabel('Evaluation Score', fontweight='bold')
        ax.set_ylabel('Attention Mechanism', fontweight='bold')
        ax.set_xlim(lower_bound, 1.02) # Little bit of padding on right
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate model plot: {e}", ha='center')

    # --- Panel C: Algorithm Performance (Bar with Error Bars) ---
    ax = axs[2]
    try:
        sns.barplot(
            data=results_df,
            x='Model',
            y='Eval_Score',
            errorbar='sd',
            palette='Blues_d',
            ax=ax,
            capsize=0.1
        )
        ax.set_title('C. ALGORITHM PERFORMANCE (Avg ± Std Dev)', fontsize=14, fontweight='bold', loc='left', pad=20)
        ax.set_xlabel('Algorithm', fontweight='bold')
        ax.set_ylabel('Evaluation Score', fontweight='bold')
        ax.set_ylim(lower_bound, 1.02)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate algo plot: {e}", ha='center')

    # --- Panel D: Attention Mechanism Performance (Bar with Error Bars) ---
    ax = axs[3]
    try:
        sns.barplot(
            data=results_df,
            x='Attention Mechanism',
            y='Eval_Score',
            errorbar='sd',
            palette='Greens_d',
            ax=ax,
            capsize=0.1
        )
        ax.set_title('D. ATTENTION MECHANISM PERFORMANCE (Avg ± Std Dev)', fontsize=14, fontweight='bold', loc='left', pad=15)
        ax.set_xlabel('Attention Mechanism', fontweight='bold')
        ax.set_ylabel('Evaluation Score', fontweight='bold')
        ax.set_ylim(lower_bound, 1.02)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate attention plot: {e}", ha='center')

    # Main Title
    plt.suptitle(f"AutoRL Evaluation Analysis Report | Schema: {schema}", fontsize=20, fontweight='bold', y=0.98)
    
    # Save
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    att_label = "AM" if attention_mech else "NoAM"
    filename = f"Analysis_Report_{schema}_{att_label}_{timestamp}.png"
    filepath = os.path.join(results_dir, filename)
    
    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Analysis report saved: {filepath}")
    except Exception as e:
        print(f"  [!] Failed to save report: {e}")
    finally:
        plt.close(fig)



def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRL: Train and Evaluate RL Agents for Predictive Maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_agent.py -S SIT -A PPO,A2C -E 200 -AM 0
  python train_agent.py -S IEEE -A PPO -E 300 -AM 1
  python train_agent.py -S SIT -A PPO -E 200 -LR 0.001,0.0001 -G 0.95,0.99  # Grid search
  python train_agent.py -V -S IEEE  # Evaluate IEEE models
  python train_agent.py -D -S SIT   # Just list SIT models
  python train_agent.py  # Uses all defaults: SIT, PPO, 200 episodes, no attention
        """
    )
    
    parser.add_argument(
        '-S', '--schema',
        default='SIT',
        choices=['SIT', 'IEEE'],
        help="Data schema: 'SIT' or 'IEEE' (default: SIT)"
    )
    
    parser.add_argument(
        '-A', '--algos',
        default='PPO',
        help="Comma-separated list of algorithms: PPO,A2C,DQN,REINFORCE (default: PPO)"
    )
    
    parser.add_argument(
        '-E', '--episodes',
        type=int,
        default=200,
        help="Number of training episodes (default: 200)"
    )
    
    parser.add_argument(
        '-AM', '--attention-mechanism',
        type=int,
        default=0,
        choices=[0, 1],
        help="Apply attention mechanism: 1 (yes) or 0 (no) (default: 0)"
    )
    
    parser.add_argument(
        '-LR', '--learning-rates',
        type=str,
        default=None,
        help="Comma-separated learning rate values for grid search (e.g., '0.001,0.0001,0.0005'). Default: 0.001"
    )
    
    parser.add_argument(
        '-G', '--gamma',
        type=str,
        default=None,
        help="Comma-separated gamma values for grid search (e.g., '0.95,0.99'). Default: 0.99"
    )
    
    parser.add_argument(
        '-V', '--eval-only',
        action='store_true',
        default=False,
        help="Evaluation only mode: Skip training and evaluate existing models (default: False)"
    )

    parser.add_argument(
        '-D', '--discover',
        action='store_true',
        default=False,
        help="Discover mode: Only list trained models for the schema and exit (default: False)"
    )
    
    args = parser.parse_args()
    
    # Parse algorithms
    algos = [algo.strip().upper() for algo in args.algos.split(',')]
    
    # Validate algorithms
    valid_algos = ['PPO', 'A2C', 'DQN', 'REINFORCE']
    for algo in algos:
        if algo not in valid_algos:
            print(f"ERROR: Algorithm '{algo}' not supported. Choose from: {', '.join(valid_algos)}")
            sys.exit(1)
    
    # Parse learning rates
    if args.learning_rates:
        try:
            learning_rates = [float(lr.strip()) for lr in args.learning_rates.split(',')]
        except ValueError:
            print(f"ERROR: Invalid learning rate format. Use comma-separated float values (e.g., '0.001,0.0001')")
            sys.exit(1)
    else:
        learning_rates = [rl_pdm.LR_DEFAULT]
    
    # Parse gamma values
    if args.gamma:
        try:
            gammas = [float(g.strip()) for g in args.gamma.split(',')]
        except ValueError:
            print(f"ERROR: Invalid gamma format. Use comma-separated float values (e.g., '0.95,0.99')")
            sys.exit(1)
    else:
        gammas = [rl_pdm.GAMMA_DEFAULT]

    
    print(f"\n{'='*80}")
    print(f"AutoRL: RL Agent Training and Evaluation Pipeline")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Data Schema: {args.schema}")
    print(f"  Algorithms: {', '.join(algos)}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Learning Rates: {learning_rates}")
    print(f"  Gamma Values: {gammas}")
    print(f"  Attention Mechanism: {'ON' if args.attention_mechanism else 'OFF'}")
    
    # Calculate total training runs
    total_runs = len(algos) * len(learning_rates) * len(gammas)
    if args.attention_mechanism:
        total_runs *= 4  # 4 attention types
    print(f"  Total training runs (per file): {total_runs}")
    print(f"{'='*80}\n")
    
    
    try:
        # Phase 0: Initialize checkpoint database
        print(f"\n{'='*80}")
        print(f"CHECKPOINT SYSTEM: Initializing...")
        print(f"{'='*80}\n")
        init_checkpoint_db()
        print(f"✓ Checkpoint database initialized: {CHECKPOINT_DB}\n")
        
        # Phase 0.5: Check for incomplete batches (crash recovery)
        batch_id = None
        resume_mode = False
        
        if not args.discover and not args.eval_only:
            incomplete_batch = get_incomplete_batch()
            if incomplete_batch:
                print(f"\n{'='*80}")
                print(f"⚠️  INCOMPLETE BATCH DETECTED")
                print(f"{'='*80}")
                print(f"  Batch ID: {incomplete_batch['batch_id']}")
                print(f"  Schema: {incomplete_batch['schema']}")
                print(f"  Algorithms: {incomplete_batch['algorithms']}")
                print(f"  Episodes: {incomplete_batch['episodes']}")
                print(f"  Status: {incomplete_batch['status']}")
                print(f"  Created: {incomplete_batch['created_at']}")
                print(f"{'='*80}\n")
                
                # Prompt user to resume
                try:
                    response = input("Resume this incomplete batch? (Y/n): ").strip().lower()
                    if response in ['', 'y', 'yes']:
                        batch_id = incomplete_batch['batch_id']
                        resume_mode = True
                        
                        # Load configuration from incomplete batch
                        args.schema = incomplete_batch['schema']
                        algos = incomplete_batch['algorithms'].split(',')
                        args.episodes = incomplete_batch['episodes']
                        learning_rates = [float(x) for x in incomplete_batch['learning_rates'].split(',')]
                        gammas = [float(x) for x in incomplete_batch['gammas'].split(',')]
                        args.attention_mechanism = incomplete_batch['attention_mech']
                        
                        print(f"\n✓ Resuming batch: {batch_id}\n")
                        update_batch_status(batch_id, 'WIP')
                    else:
                        print(f"\n✓ Starting new batch (incomplete batch will remain in database)\n")
                except (EOFError, KeyboardInterrupt):
                    print(f"\n\n✓ Starting new batch\n")
        
        # Phase 0.6: Create new batch if not resuming
        if not resume_mode and not args.discover and not args.eval_only:
            batch_id = create_batch_id()
            create_batch_record(
                batch_id=batch_id,
                schema=args.schema,
                algos=algos,
                episodes=args.episodes,
                lrs=learning_rates,
                gammas=gammas,
                attention_mech=args.attention_mechanism
            )
            update_batch_status(batch_id, 'WIP')
            print(f"✓ Created new batch: {batch_id}\n")
        
        # Phase 1: Discovery Only
        if args.discover:
            discover_trained_models(args.schema)
            sys.exit(0)

        # Phase 2: Training or Discovery
        if args.eval_only:
            trained_models = discover_trained_models(args.schema)
        else:
            trained_models = train_agents(
                schema=args.schema,
                algos=algos,
                episodes=args.episodes,
                attention_mech=args.attention_mechanism,
                learning_rates=learning_rates,
                gammas=gammas,
                batch_id=batch_id,
                resume_mode=resume_mode
            )
        
        if not trained_models:
            print("ERROR: No models were successfully trained.")
            if batch_id:
                update_batch_status(batch_id, 'Failed')
            sys.exit(1)
        
        # Phase 3: Evaluation
        results_df = evaluate_agents(schema=args.schema, trained_models=trained_models)
        
        if results_df.empty:
            print("ERROR: Evaluation produced no results.")
            if batch_id:
                update_batch_status(batch_id, 'Failed')
            sys.exit(1)
        
        # Phase 4: Save results
        results_file = save_results(results_df, args.schema, args.attention_mechanism)
        
        # Phase 5: Create analysis report and heatmaps
        print(f"\nGenerating heatmaps...")
        create_heatmaps(results_df, args.schema, args.attention_mechanism)

        print(f"\nGenerating analysis report...")
        generate_analysis_report(results_df, args.schema, args.attention_mechanism)
        
        # Phase 6: Mark batch as complete
        if batch_id:
            mark_batch_complete(batch_id)
            print(f"\n✓ Batch {batch_id} marked as COMPLETE\n")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Results Summary:")
        print(f"  Total Evaluations: {len(results_df)}")
        print(f"  Models Trained: {len(trained_models)}")
        print(f"  Test Files: {results_df['Test File'].nunique()}")
        print(f"  Average Eval Score: {results_df['Eval_Score'].mean():.4f}")
        if batch_id:
            print(f"  Batch ID: {batch_id}")
        print(f"\nResults saved in: results/")
        print(f"Checkpoint database: {CHECKPOINT_DB}")
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        if batch_id:
            print(f"   Batch {batch_id} status: WIP (can be resumed)")
            print(f"   Run the script again to resume from where you left off.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error:")
        print(f"  {str(e)}")
        if batch_id:
            update_batch_status(batch_id, 'Failed')
            print(f"  Batch {batch_id} marked as FAILED")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
