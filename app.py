# import
import os
import base64
## 3rd party
from PIL import Image
import streamlit as st
## App
from rnatargeting.predict import run_pred

# Init variables
fasta_file = None

# functions
def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return "data:image/png;base64," + encoded_string

# App init
st.set_page_config(
    page_title="RNA Targeting",
    page_icon="img/arc-logo.ico",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Set custom styles
font_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;700&display=swap');
body * {
    font-family: 'IBM Plex Sans', sans-serif !important;
}
</style>
"""
st.markdown(font_css, unsafe_allow_html=True)

# Main
## Title
image_base64 = get_image_as_base64("img/arc-logo-white.png")
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <a href="https://arcinstitute.org/tools/cas13d" target="_blank">
            <img src="{image_base64}" alt="ARC Institute Logo" style="vertical-align: middle; margin-left: 15px; margin-right: 30px;" width="65" height="65">
        </a>
        <span class='font-castoro'>
            <h2>Custom Sequence Cas13d Guide Efficiency Prediction</h2>
        </span>
    </div>
    """, unsafe_allow_html=True
)
## Description
st.markdown(
    """
    <div class='font-ibm-plex-sans'>

    <h5>This interface provides CasRx guide design for custom input sequences.</h5>

    <strong>Guidance:</strong>
    <ul>
      <li>For best results, please input the ENTIRE target sequence to enable local target structure prediction and selection of the best recommended guides.
        <ul>
            <li>If you are only interested in a short region on a target sequence, you can further process our results and pick guides in your region of interest.</li>
        </ul>
      </li>
      <li>We recommend an input sequence of at least 60nt, and ideally >200nt.</li>
      <li>Make sure to check if your gene is available in our main tool before using our custom input model as it does not use information like CDS location, splice variants or relative target positions in the full transcript.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True
)

## File upload
col1, col2 = st.columns([0.75, 0.25])
with col1:
    fasta_file = st.file_uploader('Upload a nucleotide fasta file')
with col2:
    top_n = st.number_input('Top N guides to return', min_value=1, max_value=99999, value=30)
## Predict & display results
if fasta_file is not None:
    with st.spinner("Calculating..."):
        try:
            df_pred = run_pred(fasta_file.getvalue())
            st.dataframe(df_pred.groupby('transcript id').head(top_n))
        except Exception as e:
            st.error(e)
        
        
