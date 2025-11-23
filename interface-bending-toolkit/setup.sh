# 1. Create & activate a virtualenv (Python 3.10+ recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Put your API key in the environment
# Option A: export directly in shell
export OPENAI_API_KEY="sk-..."

# Option B (optional): use a .env file
export OPENAI_API_KEY="sk-..."> .env


