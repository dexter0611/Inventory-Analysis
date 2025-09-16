import streamlit as st
import pandas as pd
import plotly.express as px
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Helper function to clean data
def clean_data(df):
    # Implement your data cleaning logic here
    return df

# Helper function to generate hash
def generate_hash(data):
    data_str = data.to_csv(index=False)
    return hashlib.sha256(data_str.encode()).digest()

# Simulated Blockchain interaction functions
def simulate_blockchain_interaction(data_hash):
    # Simulates storing and retrieving a hash from a blockchain
    print(f"Simulated storing hash: {data_hash}")
    return f"tx_hash_{data_hash[:8]}"

# Streamlit App
st.title("Inventory Analysis with Blockchain Security")

# Initialize session state variables
if 'stored_hash' not in st.session_state:
    st.session_state.stored_hash = None

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)

    # Check if required columns are already present
    required_columns = {'ProductName', 'Cost', 'QuantitySold'}
    present_columns = set(df.columns)

    if required_columns.issubset(present_columns):
        st.write("Correct column names detected, proceeding without asking for column indices.")
        df_cleaned = clean_data(df)
        st.write("Cleaned Data:")
        st.dataframe(df_cleaned)
        st.write("Columns in cleaned data:", df_cleaned.columns)

             # Generate hash of the cleaned data
        current_hash = generate_hash(df_cleaned)

        if st.session_state.stored_hash is None:
             # If no hash is stored, store the current hash
               st.session_state.stored_hash = current_hash
               tx_hash = simulate_blockchain_interaction(st.session_state.stored_hash)
               st.write(f"Data secured with simulated blockchain. Transaction Hash: {tx_hash}")
               st.write("Data hash has been stored.")

        if current_hash == st.session_state.stored_hash:
                 st.write("Hash matches. Displaying results:")
        
                 try:
                    # Calculate Total Cost and add it as a new column
                     df_cleaned['Cost'] = pd.to_numeric(df_cleaned['Cost'], errors='coerce')
                     df_cleaned['QuantitySold'] = pd.to_numeric(df_cleaned['QuantitySold'], errors='coerce')
                    # Calculate Total Cost and add it as a new column
                     df_cleaned['TotalCost'] = df_cleaned['Cost'] * df_cleaned['QuantitySold']
                     df_aggregated = df_cleaned.groupby('ProductName', as_index=False).agg({'TotalCost': 'sum'})


                     # Display Top 5 Products by Total Cost
                     st.write("Top 5 Products by Total Cost:")
                     top_5 = df_aggregated.nlargest(5, 'TotalCost')
                     st.dataframe(top_5)

                      # Visualization for Top 5 Products
                     st.write("Top 5 Products by Aggregated Total Cost (Bar Chart):")
                     fig_top = px.bar(top_5, x='ProductName', y='TotalCost', title='Top 5 Products by Aggregated Total Cost', color='ProductName')
                     st.plotly_chart(fig_top)
                     # Display Bottom 5 Products by Total Cost
                     st.write("Bottom 5 Products by Total Cost:")
                     bottom_5 = df_aggregated.nsmallest(5, 'TotalCost')
                     st.dataframe(bottom_5)

                     # Visualization for Bottom 5 Products
                     st.write("Bottom 5 Products by Total Cost (Bar Chart):")
                     fig_bottom = px.bar(bottom_5, x='ProductName', y='TotalCost', title='Bottom 5 Products by Total Cost', color='ProductName')
                     st.plotly_chart(fig_bottom)

                     # Machine Learning Model
                     X = df_cleaned[['Cost', 'QuantitySold']]
                     y = df_cleaned['TotalCost']
                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
                     model = RandomForestRegressor()
                     model.fit(X_train, y_train)
                     predictions = model.predict(X_test)
            
                     #  Add interactivity: Slider to filter data based on Total Cost
                     value = st.slider("Select a value range for Total Cost", 
                              min_value=int(df_aggregated['TotalCost'].min()), 
                              max_value=int(df_aggregated['TotalCost'].max()))
                     filtered_df = df_aggregated[df_aggregated['TotalCost'] >= value]
                     st.write("Filtered Data:")
                     st.dataframe(filtered_df)
                 except KeyError as e:
                     st.error(f"KeyError: {e}. Check if column names are correct after cleaning.")
        else:
                st.error("Data hash does not match. Displaying results is not allowed.")

    else:
        st.write("Please specify the column indices for the following fields:")
        product_col_idx = st.number_input("Product Name Column Index (starting from 0):", min_value=0, max_value=len(df.columns) - 1, step=1)
        cost_col_idx = st.number_input("Cost Column Index (starting from 0):", min_value=0, max_value=len(df.columns) - 1, step=1)
        quantity_col_idx = st.number_input("Quantity Sold Column Index (starting from 0):", min_value=0, max_value=len(df.columns) - 1, step=1)
        
        if st.button("Submit"):
             # Rename columns based on user input
             df.rename(columns={df.columns[product_col_idx]: 'ProductName', 
                               df.columns[cost_col_idx]: 'Cost', 
                               df.columns[quantity_col_idx]: 'QuantitySold'}, inplace=True)
             st.write("Columns renamed successfully.")
             st.dataframe(df)  # Show renamed columns
                # Data cleaning
             df_cleaned = clean_data(df)
             st.write("Cleaned Data:")
             st.dataframe(df_cleaned)
    
             # Check columns after cleaning
             st.write("Columns in cleaned data:", df_cleaned.columns)

             # Generate hash of the cleaned data
             current_hash = generate_hash(df_cleaned)

             if st.session_state.stored_hash is None:
             # If no hash is stored, store the current hash
               st.session_state.stored_hash = current_hash
               tx_hash = simulate_blockchain_interaction(st.session_state.stored_hash)
               st.write(f"Data secured with simulated blockchain. Transaction Hash: {tx_hash}")
               st.write("Data hash has been stored.")

             if current_hash == st.session_state.stored_hash:
                 st.write("Hash matches. Displaying results:")
        
                 try:
                     df_cleaned['Cost'] = pd.to_numeric(df_cleaned['Cost'], errors='coerce')
                     df_cleaned['QuantitySold'] = pd.to_numeric(df_cleaned['QuantitySold'], errors='coerce')
                    # Calculate Total Cost and add it as a new column
                     df_cleaned['TotalCost'] = df_cleaned['Cost'] * df_cleaned['QuantitySold']
                     df_aggregated = df_cleaned.groupby('ProductName', as_index=False).agg({'TotalCost': 'sum'})


                     # Display Top 5 Products by Total Cost
                     st.write("Top 5 Products by Total Cost:")
                     top_5 = df_aggregated.nlargest(5, 'TotalCost')
                     st.dataframe(top_5)

                      # Visualization for Top 5 Products
                     st.write("Top 5 Products by Aggregated Total Cost (Bar Chart):")
                     fig_top = px.bar(top_5, x='ProductName', y='TotalCost', title='Top 5 Products by Aggregated Total Cost', color='ProductName')
                     st.plotly_chart(fig_top)
                     # Display Bottom 5 Products by Total Cost
                     st.write("Bottom 5 Products by Total Cost:")
                     bottom_5 = df_aggregated.nsmallest(5, 'TotalCost')
                     st.dataframe(bottom_5)

                     # Visualization for Bottom 5 Products
                     st.write("Bottom 5 Products by Total Cost (Bar Chart):")
                     fig_bottom = px.bar(bottom_5, x='ProductName', y='TotalCost', title='Bottom 5 Products by Total Cost', color='ProductName')
                     st.plotly_chart(fig_bottom)

                     # Machine Learning Model
                     X = df_cleaned[['Cost', 'QuantitySold']]
                     y = df_cleaned['TotalCost']
                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
                     model = RandomForestRegressor()
                     model.fit(X_train, y_train)
                     predictions = model.predict(X_test)
            
                     #  Add interactivity: Slider to filter data based on Total Cost
                     value = st.slider("Select a value range for Total Cost", 
                              min_value=int(df_aggregated['TotalCost'].min()), 
                              max_value=int(df_aggregated['TotalCost'].max()))
                     filtered_df = df_aggregated[df_aggregated['TotalCost'] >= value]
                     st.write("Filtered Data:")
                     st.dataframe(filtered_df)
                 except KeyError as e:
                     st.error(f"KeyError: {e}. Check if column names are correct after cleaning.")
             else:
                st.error("Data hash does not match. Displaying results is not allowed.")
