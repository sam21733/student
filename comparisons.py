import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('StudentsPerformance.csv')

df = pd.DataFrame(data)

# Streamlit app
st.title("Comparison of Student Scores Across Different Groups")

# Sidebar for user input
st.sidebar.header("Choose Comparison Type")
comparison_type = st.sidebar.selectbox(
    "Select the type of comparison:",
    ["Gender", "Race/Ethnicity", "Parental Level of Education", "Lunch Type", "Test Preparation Course"]
)

# Function to plot comparison
def plot_comparison(df, comparison_column):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.boxplot(x=comparison_column, y='math score', data=df, ax=ax[0])
    ax[0].set_title('Math Score Comparison')
    
    sns.boxplot(x=comparison_column, y='reading score', data=df, ax=ax[1])
    ax[1].set_title('Reading Score Comparison')
    
    sns.boxplot(x=comparison_column, y='writing score', data=df, ax=ax[2])
    ax[2].set_title('Writing Score Comparison')
    
    plt.tight_layout()
    st.pyplot(fig)

# Display comparison plots based on user selection
if comparison_type == "Gender":
    plot_comparison(df, 'gender')
elif comparison_type == "Race/Ethnicity":
    plot_comparison(df, 'race/ethnicity')
elif comparison_type == "Parental Level of Education":
    plot_comparison(df, 'parental level of education')
elif comparison_type == "Lunch Type":
    plot_comparison(df, 'lunch')
elif comparison_type == "Test Preparation Course":
    plot_comparison(df, 'test preparation course')
