# import
import os
## 3rd party
import streamlit as st
## App
from rnatargeting.predict import run_pred

# Init variables
fasta_file = None

# App init
st.set_page_config(
    page_title="",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Styling
font_url_h = "https://fonts.googleapis.com/css2?family=Castoro"
st.markdown(f'<link href="{font_url_h}" rel="stylesheet">', unsafe_allow_html=True)
font_url_c = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;700&display=swap"
st.markdown(f'<link href="{font_url_c}" rel="stylesheet">', unsafe_allow_html=True)
## Custom CSS
st.markdown("""
    <style>
    .font-castoro {
        font-family: 'Castoro', sans-serif;
    }
    .font-ibm-plex-sans {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True
)

# Main
## Description
st.markdown(
    """
    <div class='font-castoro'>
    <h2>Predict custom sequence Cas13d guide efficiency</h2>
    </div>
    """, unsafe_allow_html=True
)
st.markdown(
    """
    <div class='font-ibm-plex-sans'>

    Here, we provide an interface of CasRx guide design for custom input sequences.

    <ul>
      <li>For best results, please input the ENTIRE target sequence to enable local target structure prediction and selection of the recommended best guides. If you are only interested in a short region on a target sequence, you can further process our results and pick guides in your region of interest.</li>
      <li>We recommend an input sequence of at least 60nt, and ideally >200nt.</li>
      <li>If you can find your gene in our precomputed page, we recommend you to use the results there, because the custom input model does not utilize information like CDS location, splice variants or relative target positions in the full transcript.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True
)

## File upload
fasta_file = st.file_uploader('Upload a fasta file')
## Predict & display results
if fasta_file is not None:
    with st.spinner("Calculating..."):
        try:
            df_pred = run_pred(fasta_file.getvalue())
            st.dataframe(df_pred)
        except Exception as e:
            st.error(e)
        
        
