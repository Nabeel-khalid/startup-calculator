import streamlit as st
import pandas as pd
import json
import os
import streamlit.components.v1 as components

# Set page configuration - must be the first Streamlit command
st.set_page_config(page_title="Team Cost Calculator", layout="wide")

# Load saved edits if available from local storage
def load_saved_edits():
    if 'saved_edits' in st.session_state:
        return st.session_state['saved_edits']
    return None

# Save edits to local storage
def save_edits_browser(edits):
    st.session_state['saved_edits'] = edits

# Main App
def main():
    st.title("Team Cost Calculator")
    
    # Load saved data
    saved_data = load_saved_edits()
    
    # If saved data exists, load it into the session state
    if saved_data:
        st.session_state['team_data'] = saved_data
    
    # Input fields
    if 'team_data' not in st.session_state:
        st.session_state['team_data'] = []
    
    st.write("### Add Team Member Information")
    name = st.text_input("Name", key="name")
    role = st.text_input("Role", key="role")
    cost = st.number_input("Cost", min_value=0, key="cost")
    
    if st.button("Add Team Member"):
        st.session_state['team_data'].append({
            "Name": name,
            "Role": role,
            "Cost": cost
        })
        st.experimental_rerun()
    
    # Display team members
    if st.session_state['team_data']:
        df = pd.DataFrame(st.session_state['team_data'])
        st.write("### Team Members")
        st.dataframe(df)
    
    # Save draft button
    if st.button("Save Draft to Browser"):
        save_edits_browser(st.session_state['team_data'])
        st.success("Draft saved to browser successfully!")

if __name__ == "__main__":
    main()
