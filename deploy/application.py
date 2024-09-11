
#Importing the necessary libraries for the Dash App
import dash
from dash import dcc, html, dash_table, callback_context, callback, no_update
from dash.dependencies import Input, Output, State
import pandas as pd

from datetime import datetime
import os #To use the environment variables
import time #To use the time library
import plotly.graph_objects as go #To uPse the Plotly library to create the graph
import plotly.express as px #To use the Plotly Express library to create the graph

#Importing the necessary libraries for the AI Agent
import langchain_mistralai #To use the AI Agent

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder #To use the ChatPromptTemplate and MessagesPlaceholder which are used to create the AI Agent prompt
from langchain_core.messages import HumanMessage #To use the HumanMessage which is used to create the AI Agent message from the user input
import re #To use the regular expression library which is used to clean the user input before sending it to the AI Agent
from dotenv import find_dotenv, load_dotenv #To load the .env file
from langchain_mistralai import ChatMistralAI #To use the ChatMistralAI which is used to create the AI Agent


#Create the variables to store important keys and passwords
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
# Retrieve the MistralAI API key from environment variables
MISTRALAI_API_KEY = os.getenv("mistral_api_key")

#Read the 'concatenated_df.csv' file from github repository
df=pd.read_csv("https://raw.githubusercontent.com/juanchok12/Consoldiated-Assets-for-Banks-and-AI/main/deploy/concatenated_df.csv")

#Function to select columns from the dataframe to create a new dataframe
def select_columns(table_df, column_names):
    new_frame = table_df.loc[:, column_names]
    return new_frame

column_names = ['Bank Name', 'Quarter', 'Date', 'Consolidated Assets','Percentage of Total Cosolidated Assets','Bank ID']  # Replace with your actual column names
table_df = select_columns(df, column_names)
#Convert the table_df into a dataframe
table_df = pd.DataFrame(table_df)
table_df.head()

# Convert the quarters to dates for sorting
def quarter_to_date(quarter_str):
    quarter, year = quarter_str.split('-')
    year = int(year)
    quarter_start_month = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}
    month = quarter_start_month[quarter]
    return datetime(year, month, 1)

# Fetch the unique quarters from the database and sort them by date
quarters = table_df['Quarter'].unique() #Get the unique quarters from the 'Quarter' column
quarters = sorted(quarters, key=quarter_to_date)
options = [{'label': q, 'value': q} for q in quarters]  # Create options for dropdown
latest_quarter = quarters[-1] if quarters else None


# Initialize the Dash app
app = dash.Dash(__name__)

#=========================Layout of the app=========================
app.layout = html.Div([
    html.Div([
        html.H3(style={'textAlign': 'center'}),
        dcc.Input(
            id='bank-name-input',
            type='text',
            placeholder='Enter bank name',
            style={'width': '400px', 'height': '30px', 'margin-right': '100px', 'margin-bottom': '20px', 'margin-top': '20px', 'margin-left': '50px', 'backgroundColor': '#121212', 'color': '#FFFFFF', 'textAlign': 'center', 'border': '1px solid #FFFFFF'}
        ),
 html.Div([
            dcc.Dropdown(
                id='quarter-dropdown',
                options=options,
                value=latest_quarter,
                placeholder='Select a quarter',
                style={'width': '200px', 
                       'height': '30px',
                        'margin-bottom': '10px', 
                        'margin-top': '8px',
                       'backgroundColor': '#121212', 
                       'color': '#FFFFFF'}
            ),
            html.Button(
                "Download CSV", 
                id="btn_csv", 
                n_clicks=0, 
                style={'width': '200px', 
                       'height': '30px', 
                       'margin-bottom': '20px',
                       'margin-top': '40px',
                       'margin-right': '50px', 
                       'backgroundColor': '#121212', 
                       'color': '#FFFFFF',
                       'border': '1px solid #FFFFFF', 
                       'margin-left': '100px'}
            )
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),
    ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),
        
    dcc.Download(id="download-dataframe-csv"),


    dcc.RangeSlider(
        id='assets-range-slider',
        min=0, max=4000000000000, step=1000000000,
        marks={i: f'${i//1000000000000} Trillion' for i in range(0, 4000000000001, 1000000000000)},
        value=[0, 4000000000000],
        tooltip={'placement': 'top', 'always_visible': True},
        included=True,
    ),
    
    dash_table.DataTable(
        id='datatable',
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#333333', 
                      'color': '#E1E1E1', 
                      'fontWeight': 'bold',
                      'border': '1px solid #E1E1E1'},
        style_cell={'backgroundColor': '#121212', 'color': '#E1E1E1', 'border': '1px solid #E1E1E1', 'textAlign': 'center'},
        style_data={'backgroundColor': '#121212', 'color': '#E1E1E1', 'border': '1px solid #E1E1E1'},
        style_data_conditional=[
            {'if': {'state': 'selected'}, 'backgroundColor': '#4f6475', 'border': '1px solid #FFFFFF'},
            {'if': {'state': 'active'}, 'backgroundColor': '#4f6475', 'border': '1px solid #E1E1E1'}
        ],
        cell_selectable=True,
        selected_cells=[], 
        page_size=10, #Set the number of row per page to display
    ),

    #Adding the AI Agent to the app

    html.H1("Plotly AI for Creating Graphs"),
  
    dcc.Textarea(id='user-request', #Its the property that listens to the user input/text area. THis is the value of the user input.
                style={'width': '50%', 
                        'height': 50, 
                        'margin-top': 20,
                        'backgroundColor': '#121212', 
                        'color': '#FFFFFF',
                        },
                placeholder='Enter your request here...'),
    html.Br(),
    html.Button('Submit', id='my-button'),
    dcc.Loading(
        [
            html.Div(id='my-figure', children=''), #Display the graph
            dcc.Markdown(id='content', 
                         children='', #Display the content of the AI Agent which will be the code to create the graph
                        #dangerously_allow_html=True, 
                        style={'padding': '1px', 
                             'margin-top':'1px', # Add margin to the top
                             'backgroundColor':'#121212',
                             'color':'white', # Change the font color to white
                         },
                        className='dash-markdown'
                         ) #Display the content of the AI Agent which will be the code to create the graph
        ],
        type='cube'
    )
    
])

# Callback to update the table based on filters
@app.callback(
    Output('datatable', 'data'),
    [
        Input('bank-name-input', 'value'),
        Input('quarter-dropdown', 'value'),
        Input('assets-range-slider', 'value')
    ]
)
#Function to update the table based on the filters
def update_table(bank_name, select_quarter, assets_range):
    

    #Read the 'concatenated_df.csv' file from the github
    table_df=pd.read_csv("https://raw.githubusercontent.com/juanchok12/Consoldiated-Assets-for-Banks-and-AI/main/deploy/concatenated_df.csv")
    #Selected columns from the dataframe:
    column_names = ['Bank Name', 'Quarter', 'Date', 'Consolidated Assets','Percentage of Total Cosolidated Assets','Bank ID']  # Replace with your actual column names
    table_df = select_columns(table_df, column_names)

    # Apply the quarter filter first to narrow down the data
    if select_quarter:
        select_quarter = str(select_quarter).strip()
        table_df['Quarter'] = table_df['Quarter'].astype(str).str.strip()
        print(f"Filtering by quarter: {select_quarter}")
        
        if select_quarter in table_df['Quarter'].values:
            table_df = table_df[table_df['Quarter'] == select_quarter]
            table_df = table_df.reset_index(drop=True)  # Reset the index after filtering
        else:
            print(f"Warning: Selected quarter '{select_quarter}' is not available in the data.")
            return []


    # Apply filters based on bank name
    if bank_name:
        table_df = table_df[table_df['Bank Name'].str.contains(bank_name, case=False, na=False)]
        table_df = table_df.reset_index(drop=True)  # Reset the index after filtering

    
    if assets_range:
        table_df = table_df[(table_df['Consolidated Assets'] >= assets_range[0]) & (df['Consolidated Assets'] <= assets_range[1])]

    # Sort by Consolidated Assets in descending order
    table_df = table_df.sort_values(by='Consolidated Assets', ascending=False)

    # Convert each row to a dictionary
    data = table_df.to_dict(orient='records')

    #Convert data to a DataFrame and save to CSV
    table_df=pd.DataFrame(data)
    # Set the file directory to save the CSV file
    dir = os.getenv("dir")
    # Save to CSV
    table_df.to_csv(dir+"\\banking_assets_data.csv", index=False)

    return data

print(f"Saving file to: {dir}")

# Callback to download the table into a CSV file
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    State("datatable", 'data'),
    prevent_initial_call=True
)

#Function to download the table into a CSV file
def download_csv(n_clicks, data):
    if n_clicks > 0:
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, "banking_assets_data_download.csv")

#=========================AI Agent=========================


#Describing the AI Agent model and API key
# Choose the model to use
model=ChatMistralAI(
    api_key=MISTRALAI_API_KEY,
    model='codestral-latest'
)

#Create the prompt for the AI Agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
        

            "system", 
            "You're a data visualization expert and use your favourite graphing library Plotly only. "
            "Pick a dark theme to the Plotly data visualization and the code with a background color of #121212 and text color of #FFFFFF. "
           
               "The data is provided as a string representation of the first 10 rows of a CSV file. "
               "Here's the data: {data} "
               "Follow these steps: "
               "1. Convert the string data into a dictionary with appropriate keys and values. "
               "2. Create a pandas DataFrame from this dictionary. "
               "3. Use this DataFrame to create Plotly visualizations as requested. "
               "4. Assign the final Plotly figure to a variable named 'fig'."
               "5. Do not opt for a direct string parsing approach to create the DataFrame and instead use a dictionary-based approach. "
               "Always include all necessary imports in your code."
               "Follow the user's indications when creating the graph."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model #Create the chain to send the prompt to the AI Agent

#The 'get_response' function is used to send the prompt to the AI Agent and get the response from the AI Agent

def get_fig_from_code(code):
    local_variables = {} #Create an empty dictionary to store the local variables
    try:
        exec(code, globals(), local_variables) #Execute the code and store the local variables
        print("Local variables after exec:", local_variables)
        if 'fig' not in local_variables:
            # If 'fig' is not found, look for any Figure object
            for var_name, var_value in local_variables.items(): #Iterate through the local variables
                if isinstance(var_value, go.Figure): #Check if the variable is a Plotly Figure object
                    return var_value #Return the Plotly Figure object
            raise KeyError("No Plotly Figure object found in the generated code")
        return local_variables['fig'] #Return the 'fig' variable which is the Plotly Figure object
    except Exception as e:
        print(f"Error executing code: {str(e)}")
        print("Generated code:")
        print(code)
        raise


#Callback for the AI Agent
@callback(
    Output('my-figure', 'children'),
    Output('content', 'children'),
    Input('my-button', 'n_clicks'),
    State('user-request', 'value'),
    prevent_initial_call=True
)


#Function to create the graph based on the user input
def create_graph(_, user_input): #'user_input' is the value of the user input from the text area, '_'' is the value of the button click
    dir = os.getenv("dir")
    
    df_ai = pd.read_csv(dir+"\\banking_assets_data.csv") #Load the dataset
    # Add the timestamp as a query parameter
    df_ai_10 = df_ai.head(10)
    csv_string = df_ai_10.to_string(index=False)
    
    #Data processing example
    data_processing_example = """
    # Example of how to process the data:
data = {
    'Bank Name': ['JPMORGAN CHASE BK NA/JPMORGAN CHASE & CO', 'BANK OF AMER NA/BANK OF AMER CORP', 'WELLS FARGO BK NA/WELLS FARGO & CO', 'CITIBANK NA/CITIGROUP', 'U S BK NA/U S BC', 'PNC BK NA/PNC FNCL SVC GROUP', 'GOLDMAN SACHS BK USA/GOLDMAN SACHS GROUP THE', 'TRUIST BK/TRUIST FC', 'CAPITAL ONE NA/CAPITAL ONE FC', 'T D BK NA/TD GRP US HOLDS LLC'],
    'Quarter': ['Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024', 'Q1-2024'],
    'Date': ['2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31'],
    'Consolidated Assets': [3.503360e+12, 2.550363e+12, 1.743283e+12, 1.698856e+12, 6.694260e+11, 5.619500e+11, 5.491880e+11, 5.267140e+11, 4.788770e+11, 3.698600e+11],
    'Percentage of Total Cosolidated Assets': [0.1581, 0.1151, 0.0787, 0.0767, 0.0302, 0.0254, 0.0248, 0.0238, 0.0216, 0.0167],
    'Bank ID': [852218, 480228, 451965, 476810, 504713, 817824, 2182786, 852320, 112837, 497404],
    'Bank Location': ['COLUMBUS, OH', 'CHARLOTTE, NC', 'SIOUX FALLS, SD', 'SIOUX FALLS, SD', 'CINCINNATI, OH', 'WILMINGTON, DE', 'NEW YORK, NY', 'CHARLOTTE, NC', 'MC LEAN, VA', 'WILMINGTON, DE'],
    'Domestic Branches': [4912.0, 3744.0, 4297.0, 653.0, 2291.0, 2373.0, 2.0, 1930.0, 273.0, 1179.0],
    'Foreign Branches': [32.0, 23.0, 10.0, 115.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0]
}
    df = pd.DataFrame(data)
    """


    # Send the user input to the AI Agent
    response = chain.invoke(
        {
            "messages": [HumanMessage(content=f"{user_input}\n\nUse this method to process the data:\n{data_processing_example}, and do not use the a direct string parsing approach to create the DataFrame. Assign the final Plotly figure to a variable named 'fig'")], #This is the user input value that is sent to the AI Agent
            "data": csv_string #Send the first 5 rows of the data set to the AI Agent in the form of a string
        }
    )
    result_output = post_process_ai_response(response.content) #Get the response from the AI Agent
    print(result_output) #Print the response from the AI Agent

    # Check if the response contains a code block using regular expression and extract the code block
    code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', result_output, re.DOTALL) 
    if code_block_match: #If the response contains a code block
        code_block = code_block_match.group(1).strip() #Get the code block from the response
        
        print("Original code block:", code_block)  # Print the original code block

        cleaned_code = re.sub(r'(?m)^\s*fig\.show\(\)\s*$', '', code_block) #Remove the 'fig.show()' from the code block
        
        print("Cleaned code block:", cleaned_code)  # Print the cleaned code block

        fig = get_fig_from_code(cleaned_code) #Get the Plotly Figure object from the code block
        return dcc.Graph(figure=fig), result_output #Return the graph and the response from the AI Agent
    else:
        return "", result_output

def post_process_ai_response(content):
    # Replace StringIO usage with dictionary-based approach
    content = content.replace("pd.read_csv(pd.compat.StringIO(data)", "pd.DataFrame(data)")
    content = content.replace("from io import StringIO", "# Data processing using dictionary")
    return content

if __name__ == '__main__':
    app.run_server(debug=True)
