# LARA
LARA: Long-horizon Action-diffusion Robotic Agent
<img alt="LARA logo" src="https://github.com/EashanVytla/LARA/blob/main/LaraLogo.png" />

## Evaluation Ground Truth Generation

This project includes tools for generating evaluation ground truth data using Claude API to label robot behavior subtasks.

### Setup

1. Install the package:
```bash
pip install -e .
```

Or for development with extra tools:
```bash
pip install -e ".[dev]"
```

2. Set up your Anthropic API key:
   - Get your API key from: https://console.anthropic.com/
   - Copy `.env.example` to `.env`
   - Add your API key to `.env`:
     ```
     ANTHROPIC_API_KEY=your_actual_api_key_here
     ```

3. Configure your data paths in `config/eval_gt_config.yaml`

### Usage

Run the evaluation ground truth generation:

```bash
# Option 1: Using the console script (after pip install -e .)
lara-generate-eval-gt

# Option 2: Running directly
cd llm_evaluator
python generate_eval_gt.py
```

You can override config parameters from the command line:
```bash
lara-generate-eval-gt \
  llm_interface.model=claude-3-haiku-20240307 \
  llm_interface.temperature=0.5 \
  data.batch_size=1 \
  eval.save_images=true

# Or with direct python:
python generate_eval_gt.py \
  llm_interface.model=claude-3-haiku-20240307 \
  llm_interface.temperature=0.5 \
  data.batch_size=1 \
  eval.save_images=true
```

### Configuration

Key configuration files:
- `config/eval_gt_config.yaml` - Main eval configuration
- `config/base_config.yaml` - Base configuration (extended by eval config)

Key parameters:
- `llm_interface.model` - Claude model to use (default: claude-3-haiku-20240307)
- `llm_interface.temperature` - Sampling temperature (default: 0.7)
- `data.history_length` - Number of past states/actions to include (default: 3)
- `data.camera_view` - Camera view to use: "head", "left_wrist", or "right_wrist"
- `eval.use_stuck_detection` - Enable stuck detection (default: true)
- `eval.velocity_threshold` - Threshold for stuck detection (default: 0.01)
- `eval.save_images` - Save images alongside JSONL (default: false)

### Output

The script generates:
- `eval_outputs/eval_gt.jsonl` - JSONL file with ground truth subtasks
- `eval_outputs/metadata.json` - Metadata about the generation run

Each line in the JSONL file contains:
```json
{
  "task_id": "0",
  "episode_id": "123",
  "frame_id": 456,
  "timestamp": 15.2,
  "subtask_gt": "Rotate left 45 degrees to search for the trash can",
  "task_name": "pick_place"
}
```

### Project Structure

```
llm_evaluator/
├── generate_eval_gt.py    # Main script
├── llm_interface.py       # Claude API interface
├── writer.py              # JSONL writer with metadata
└── __init__.py

config/
├── base_config.yaml       # Base configuration
└── eval_gt_config.yaml    # Eval-specific configuration

datas/
├── data_module.py         # PyTorch Lightning data module
└── dataset.py             # Dataset classes
```
