"""
Unified secret loading for local dev and Streamlit Cloud.

Local:            keys come from .env (loaded by python-dotenv in analyst.py)
Streamlit Cloud:  app.py injects st.secrets into os.environ at startup,
                  so os.getenv() works the same in both environments.

This module exists as a convenience import; no modules need to change.
"""
import os


def get_secret(key: str, default: str = "") -> str:
    """Return secret by name from os.environ (works local + cloud)."""
    return os.getenv(key, default)


OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
FRED_API_KEY = get_secret("FRED_API_KEY")
