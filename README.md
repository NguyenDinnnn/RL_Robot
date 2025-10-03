
# RL Robot Path Optimization (FastAPI + Console)

This project shows a full pipeline for robot path optimization using Reinforcement Learning (Monte Carlo and Q-learning),
with a **FastAPI server** hosting the environment and **console clients** that train/evaluate via HTTP.

## 🗂 Structure

```
rl_fastapi_robot/
├── app/
│   ├── robot_env.py      # GridWorld environment
│   └── server.py         # FastAPI server
├── clients/
│   ├── random_agent.py   # Baseline (random actions)
│   ├── train_mc.py       # Monte Carlo control training
│   ├── train_qlearning.py# Q-learning training
│   └── eval_policy.py    # Evaluate trained policies
├── models/               # Saved Q tables (.npy)
├── requirements.txt
└── README.md
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.server:app --reload --port 8000