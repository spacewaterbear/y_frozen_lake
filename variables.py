import os
from dotenv import load_dotenv
load_dotenv()
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
assert WANDB_API_KEY is not None, "WANDB_API_KEY is not defined, please add it in a '.env' file"
