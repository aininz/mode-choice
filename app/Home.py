import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modechoice.config import MODES

st.set_page_config(page_title="ModeChoice - Nested Logit", layout="wide")

st.title("🚆✈️ ModeChoice - Nested Logit with Heterogenous Features App")

st.markdown(
    """
This app uses a trained **Nested Logit** model with **heterogenous features** to:
- Predict **substitution response** when changing one mode's price
- Optimize **population pricing** with capacity constraints + scenarios
- Optimize **user-specific pricing** (optimize objective for a single user context)

Use the left sidebar to navigate pages.
"""
)

st.subheader("Modes")
st.write(MODES)

st.info("Go to a page on the left to run pricing or substitution experiments.")