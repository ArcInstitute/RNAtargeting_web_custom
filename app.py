# import
import os
## 3rd party
import streamlit as st
## App
from predict import run_pred

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

# Main
st.markdown(
        """
        ## Predict custom sequence Cas13d guide efficiency

        Here, we provide an interface of CasRx guide design for custom input sequences.

        * For best results, please input the ENTIRE target sequence to enable local target structure prediction and selection of the recommended best guides. If you are only interested in a short region on a target sequence, you can further process our results and pick guides in your region of interest.
        * We recommend an input sequence of at least 60nt, and ideally >200nt.
        * If you can find your gene in our precomputed page, we recommend you to use the results there, because the custom input model does not utilize information like CDS location, splice variants or relative target positions in the full transcript.
        """
    )
fasta_file = st.file_uploader('Upload a fasta file')
if fasta_file is not None:
    with st.spinner("Calculating..."):
        try:
            df_pred = run_pred(fasta_file.getvalue())
            st.dataframe(df_pred)
        except Exception as e:
            st.error(e)
        
        
