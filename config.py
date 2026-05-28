import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "https://RESOURCE.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")

# Pipeline settings
DQ_SCORE_THRESHOLD       = 75 # Records below this are quarantined
ANOMALY_VOLUME_THRESHOLD = 1.4 # 40% above baseline triggers anomaly
PROMOTION_THRESHOLD      = 80 # Minimum score to promote Silver -> Gold
OUTPUT_DIR               = "outputs"
