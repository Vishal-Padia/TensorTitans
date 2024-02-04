# TensorTitans

# Setting up the environment

1. Create a virual environment

```bash
python3 -m venv env
```

2. Activate the virtual environment

```bash
source env/bin/activate # Linux/Mac
.\env\Scripts\activate # Windows
```

3. Install required packages

```bash
pip install -r requirements.txt
```

4. Run FastAPI in sent_analysis

```bash
cd sent_analysis
uvicorn main:app --reload
```
