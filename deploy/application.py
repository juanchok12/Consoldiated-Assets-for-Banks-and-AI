
#Importing the necessary libraries for the Dash App
import dash
from dash import Dash, dcc, html, dash_table, callback_context, callback, no_update
from dash.dependencies import Input, Output, State
import pandas as pd

from datetime import datetime
import os #To use the environment variables
import time #To use the time library
import plotly.graph_objects as go #To uPse the Plotly library to create the graph
import plotly.express as px #To use the Plotly Express library to create the graph
import gevent

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder #To use the ChatPromptTemplate and MessagesPlaceholder which are used to create the AI Agent prompt
from langchain_core.messages import HumanMessage #To use the HumanMessage which is used to create the AI Agent message from the user input
import re #To use the regular expression library which is used to clean the user input before sending it to the AI Agent
from dotenv import find_dotenv, load_dotenv #To load the .env file
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import TavilySearchResults
import openai
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from pydantic import BaseModel #To use the BaseModel class which is used to create the data model for the AI Agent
from typing import List #To use the List class which is used to create a list of strings
import json #To use the json library which is used to read the JSON file
from pydantic import ValidationError
import requests #To use the requests library which is used to send the request to the AI Agent
from datetime import datetime, timedelta



#Create the variables to store important keys and passwords
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Retrieve the MistralAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #https://platform.openai.com/api-keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  #https://app.tavily.com/sign-in


search_tool = TavilySearchResults()

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
server = app.server
#=========================Layout of the app=========================
app.layout = html.Div([
    dcc.Store(id='stored-dictionary'), #Store the dictionary in the app
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
                       'margin-left': '100px',
                       'cursor': 'pointer',  # Pointer on hover
                       'borderRadius': '5px',  # Rounded corners
                       }
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

html.Div([
    # Title for the graph section
    html.Div([
        html.H1("Plotly AI for Creating Graphs", style={'textAlign': 'center'})
    ], style={'width': '50%', 'display': 'inline-block', 'textAlign': 'center'}),

    # Title for the insights & analysis section
    html.Div([
        html.H1("Insights & Analysis", style={'textAlign': 'center'})
    ], style={'width': '50%', 'display': 'inline-block', 'textAlign': 'center'})
], style={'display': 'flex', 'width': '100%'}),

    # Div containing both the text area and dropdown menu side by side
    html.Div([
        # Left column (Plotly AI)
        html.Div([
            html.H3('Enter Plot Request:', style={'textAlign': 'center', 'margin-bottom': '0px'}),
            # Text area for user input
            dcc.Textarea(
                id='user-request',
                style={
                    'width': '95%',
                    'height': 50,
                    'margin-top': 20,
                    'backgroundColor': '#121212',
                    'color': '#FFFFFF',
                },
                placeholder='Enter your request here...'
            ),
            # Submit button
            html.Button(
                'Submit',
                id='my-button',
                style={
                    'width': '95%',
                    'margin-top': '20px',
                    'backgroundColor': '#121212',
                    'color': '#FFFFFF',
                    'border': '1px solid #FFFFFF',
                    'borderRadius': '5px',
                    'boxShadow': '3px 3px 10px rgba(0, 0, 0, 0)',
                    'cursor': 'pointer',
                    'padding': '10px 20px',
                    'transition': 'background-color 0.3s, box-shadow 0.1s ease'
                }
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),

        # Right column (Insights & Analysis)
        html.Div([
            html.H3('Select a Bank:', style={'textAlign': 'center', 'margin-bottom': '27px'}),
            # Dropdown menu for bank selection
            dcc.Dropdown(
                id='bank-dropdown',
                options=[
                    {'label': 'JPMorgan Chase & Co', 'value': 'JPMorgan Chase & Co'},
                    {'label': 'Bank of America', 'value': 'Bank of America'},
                    {'label': 'Wells Fargo', 'value': 'Wells Fargo'},
                    {'label': 'Citibank', 'value': 'Citibank'}
                    # Add more banks here
                ],
                value='JPMorgan Chase & Co',  # Default bank selection
                #Set style with the margine being invisible
                style={
                    'backgroundColor': '#121212',
                    'color': '#FFFFFF',
                    'border': '1px solid #FFFFFF',
                    'width': '100%',
                    'textAlign': 'center',
                    'margin': 'auto',
                }
            ),
            # Analyze button
            html.Button(
                'Analyze',
                id='analyze-button',
                style={
                    'width': '95%',
                    'margin-top': '20px',
                    'backgroundColor': '#121212',
                    'color': '#FFFFFF',
                    'border': '1px solid #FFFFFF',
                    'borderRadius': '5px',
                    'boxShadow': '3px 3px 10px rgba(0, 0, 0, 0)',
                    'cursor': 'pointer',
                    'padding': '10px 20px',
                    'transition': 'background-color 0.3s, box-shadow 0.1s ease',
                    'display': 'block',
                    'margin-left': 'auto',
                    'margin-right': 'auto',
                    'margin-top': '35px'
                }
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ], style={'display': 'flex', 'width': '100%'}),
    


# Div containing both the output areas side by side
    html.Div([
        # Left column (Plotly figure and code)
        html.Div([
            dcc.Loading(
                id="loading-left",
                type="graph",
                children=[
                    html.Div(id='my-figure', children=''),  # Div for Plotly figure
                    dcc.Markdown(
                        id='content',
                        children='',
                        style={
                            'padding': '1px',
                            'margin-top': '1px',
                            'backgroundColor': '#121212',
                            'color': 'white',
                        },
                        className='dash-markdown'
                    ),
                ],
                style={'width': '100%'}
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),

        # Right column (additional content)
        html.Div([
            dcc.Loading(
                id="loading-right",
                type="cube",
                children=[
                    dcc.Markdown(
                        id='additional-content',
                        children='''Click 'Analyze' to generate analysis...''',
                        style={
                            'padding': '1px',
                            'margin-top': '1px',
                            'backgroundColor': '#121212',
                            'color': 'white',
                        },
                        className='dash-markdown',
                        link_target='_blank',
                    ),
                ],
                style={'width': '100%'}
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ], style={'display': 'flex', 'width': '100%'})


    ])

#=========================Callbacks=========================


# Callback to update the table based on filters
@app.callback(
    Output('datatable', 'data'),
    Output('stored-dictionary', 'data'),
    [
        Input('bank-name-input', 'value'),
        Input('quarter-dropdown', 'value'),
        Input('assets-range-slider', 'value')
    ]
)
#================Function to update the table based on the filters================================
def update_table(bank_name, select_quarter, assets_range):
    
    global table_df_fil, dictionary, bank_names_string
    #Read te 'concatenated_df.csv' file from the github
    table_df_fil=df
    #Selected columns from the dataframe:
    column_names = ['Bank Name', 'Quarter', 'Date', 'Consolidated Assets','Percentage of Total Cosolidated Assets','Bank ID']  # Replace with your actual column names
    table_df_fil = select_columns(table_df_fil, column_names)


    # Apply the quarter filter first to narrow down the data
    if select_quarter:
        select_quarter = str(select_quarter).strip() # Remove any leading/trailing whitespaces
        #table_df_fil['Quarter'] = table_df_fil['Quarter'].astype(str).str.strip() # astyp
        
        
        if select_quarter in table_df_fil['Quarter'].values:
            table_df_fil = table_df_fil[table_df_fil['Quarter'] == select_quarter]
            table_df_fil = table_df_fil.reset_index(drop=True)  
        else:
            print(f"Warning: Selected quarter '{select_quarter}' is not available in the data.")
            return []


    # Apply filters based on bank name
    if bank_name:
        table_df_fil = table_df_fil[table_df_fil['Bank Name'].str.contains(bank_name, case=False, na=False)]
        table_df_fil = table_df_fil.reset_index(drop=True)  # Reset the index after filtering

    
    if assets_range:
        table_df_fil = table_df_fil[(table_df_fil['Consolidated Assets'] >= assets_range[0]) & (table_df_fil['Consolidated Assets'] <= assets_range[1])]

    # Sort by Consolidated Assets in descending order
    table_df_fil = table_df_fil.sort_values(by='Consolidated Assets', ascending=False)
    #Use the 'Bank Name' column from 'table_df_fil' to get the unique bank names of the banks and convert it to a list string to then feed it to the analyst ai agents
    bank_names = table_df_fil['Bank Name'].unique().tolist()  # Extract unique bank names
    #Reduce the number of bank names to 10
    bank_names = bank_names[:10]
    bank_names_string = ', '.join(bank_names)  # Convert list to string


    # Convert each row to a list of dictionaries
    data = table_df_fil.to_dict(orient='records')

    dictionary=table_df_fil.to_dict(orient='list')



    #Convert data to a DataFrame and save to CSV
    table_df_fil=pd.DataFrame(data)

    return data, dictionary



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
# Describing the AI Agent model and API key
model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model='gpt-4o',  # Or 'gpt-3.5-turbo' based on your preference
)

#Create the prompt for the AI Agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
        

            "system", 
            "You're a data visualization expert and use your favourite graphing library Plotly only. "
            "Pick a dark theme to the Plotly data visualization and the code with a background color of #121212 and text color of #FFFFFF. "
           
               "The data is provided as a string representation of the first 10 rows of a CSV file. "
               "Here's the data: {data}"
               "Follow these steps:"
               "1. Convert the string data into a dictionary with appropriate keys and values. "
               "2. Create a pandas DataFrame from this dictionary. "
               "3. Use this DataFrame to create Plotly visualizations as requested. "
               "4. Assign the final Plotly figure to a variable named 'fig'."
               "5. Do not opt for a direct string parsing approach to create the DataFrame and instead use a dictionary-based approach. "
               "6. In your code, to convert the 'Date' column to a datetime object, use the following code:"
                "df['Date'] = pd.to_datetime(df['Date'])"
                "7. Order of the dates in the dataframe in chronological order:"
                    "df = df.sort_values(by='Date')"
                 
               "Always include all necessary imports in your code."
               "Follow the user's indications when creating the graph."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model #Create the chain to send the prompt to the AI Agent

#The 'get_response' function is used to send the prompt to the AI Agent and get the response from the AI Agent

def get_fig_from_code(code):
    local_variables = {'data': csv_string} #Create a dictionary with the data variable
    #local_variables = {} #Create an empty dictionary to store the local variables
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
    State('stored-dictionary', 'data'),  # Retrieve the stored dictionary here

    #State('bank-dropdown', 'value'),  # Get the selected bank
    prevent_initial_call=True
)


#Function to create the graph based on the user input
def create_graph(_, user_input, stored_dictionary): #'user_input' is the value of the user input from the text area, '_'' is the value of the button click
    if stored_dictionary is None:
        return "", "No data available for visualization."


    global csv_string
    data_ai_10 = {key: value[:10] for key, value in stored_dictionary.items()}

     # Convert the dictionary to a string representation for the AI agent
    csv_string = str(data_ai_10)  # You may adjust the format as needed
    


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


#=========================AI Agentic System=========================





#Define a Pydantic model for the analyst's output
class AnalystOutput(BaseModel):
    analysis_points: List[str] # List of analysis points
    sources: List[str] # List of sources


# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal="""Uncover the latest news and trends about the banks mentioned in the task section.""",
  backstory="""You work at a leading banking think tank. 
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=ChatOpenAI(model="gpt-4o"),
)

writer = Agent(
  role='Banking Content Strategist',
  goal='Craft compelling content from news on the selected banks.',
  backstory="""You are a famous Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives. You avoid complex words so it doesn't sound like you're an AI.""",
  verbose=True,
  allow_delegation=True,
  llm=ChatOpenAI(model="gpt-4o")
)


# Callback for AI analysis output
@callback(
    Output('additional-content', 'children'),
    Input('analyze-button', 'n_clicks'),  # Triggered by the 'Analyze' button
    State('bank-dropdown', 'value'),  # Get the selected bank
    prevent_initial_call=True

)


def activate_agent(n_clicks, bank_chosen):
    if n_clicks is None:
        return no_update
    elif not bank_chosen:
        return "Please select a bank."
    else:
        bank_name = bank_chosen.strip()

        # Task for analyzing the selected bank with adjusted prompt
        task1 = Task(
            description=f"""Conduct a comprehensive analysis of the latest news about {bank_name}.
            Focus solely on {bank_name} and do not include information about other banks.
            Identify key trends, investments, loans, acquisitions, or holdings for {bank_name}.
            Research how macroeconomic factors, such as interest rates, might impact {bank_name}'s performance.
            Provide your findings in bullet points and include the links to the news articles you used.

            Output your findings as a JSON object with the following structure:

            {{
                "analysis_points": ["point1", "point2", ...],
                "sources": ["url1", "url2", ...]
            }}

            Ensure that your output is valid JSON without any code fences or additional text. Do not wrap your JSON output in triple backticks or any markdown formatting.""",
            expected_output="A JSON object with 'analysis_points' and 'sources' fields",
            agent=researcher
        )

        # Task for writing the blog post for the selected bank remains unchanged
        task2 = Task(
            description=f"""Using the insights provided, develop an engaging blog post that highlights the latest concerns and projections for {bank_name}.
            Support your arguments with key financial metrics.
            Focus solely on {bank_name} and do not include information about other banks.""",
            expected_output="Full blog post in the form of four paragraphs for this bank",
            agent=writer
        )

        # Create a crew for sequential processing
        crew = Crew(
            agents=[researcher, writer],
            tasks=[task1, task2],
            verbose=True,
            process=Process.sequential
        )

        # Run the crew and generate the result
        result = crew.kickoff()

        # Access the agent's output from task1
        task_output = task1.output  # This is a TaskOutput object

        # Extract the agent's output content
        if hasattr(task_output, 'content'):
            agent_output_content = task_output.content
        elif hasattr(task_output, 'raw'):
            agent_output_content = task_output.raw
        else:
            return "Error: Could not retrieve agent output content."

        # Print the agent's output for debugging
        print(f"Agent Output Content:\n{agent_output_content}\n")

        # Clean up the agent's output to extract the JSON content
        agent_output_content = agent_output_content.strip()

        # Use regular expressions to remove code fences and extract JSON
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', agent_output_content, re.DOTALL)
        if not json_match:
            json_match = re.search(r'```(.*?)```', agent_output_content, re.DOTALL)
        if json_match:
            agent_output_content = json_match.group(1).strip()
        else:
            # If no code fences are found, attempt to parse the content directly
            agent_output_content = agent_output_content.strip()

        # Parse the agent's output into the AnalystOutput model
        try:
            analyst_output_model = AnalystOutput.model_validate_json(agent_output_content)
        except ValidationError as e:
            return f"Error validating agent output: {e}\n\nAgent output was:\n{agent_output_content}"
        except Exception as e:
            return f"Unexpected error parsing agent output: {e}\n\nAgent output was:\n{agent_output_content}"

        # Extract the analysis points and sources
        analysis_points = analyst_output_model.analysis_points
        sources = analyst_output_model.sources

        # Format the analysis points and sources for display
        analysis_markdown = "### Key Takeaways:\n" + "\n".join(f"- {point}" for point in analysis_points)
        sources_markdown = "### Sources:\n" + "\n".join(f"- [{url}]({url})" for url in sources)
        title_markdown = f"# {bank_name} Analysis and Insights\n\n"

        # Collect the blog post result from task2
        writer_output = task2.output

        # Extract the writer's output content
        if hasattr(writer_output, 'content'):
            full_blog_post = writer_output.content
        elif hasattr(writer_output, 'raw'):
            full_blog_post = writer_output.raw
        else:
            full_blog_post = writer_output  # Assume it's a string

        # Combine all content
        combined_content = f"{title_markdown}\n\n{analysis_markdown}\n\n{full_blog_post}\n\n{sources_markdown}"

        return combined_content


if __name__ == '__main__':
    app.run_server(debug=True)
