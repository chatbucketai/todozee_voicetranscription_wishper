import os

from dotenv import load_dotenv

load_dotenv()

VOXLINGUA_URL = os.environ["VOXLINGUA_URL"]

STT_INDIAN_BASE_URL = os.environ["STT_INDIAN_BASE_URL"]
STT_B2B_TOKEN_KEY = os.environ["STT_B2B_TOKEN_KEY"]

WHISPER_URL = os.environ["WHISPER_URL"]
