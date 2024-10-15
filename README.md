# AI Analysis on Bank Consolidated Assets 🤖💰

https://github.com/user-attachments/assets/38f2ad7f-a338-4139-a697-5da095207531

This app leverages AI to deliver a seamless data experience through two awesome AI analysis tools:

🤖 AI-Powered Data Visualization: Automatically plots insightful, custom graphs of banking data with Plotly. No coding required—just sit back and enjoy the visual breakdown!

📰 AI-Driven Market Insights: Need the latest news on your favorite banks? This agent scours the web for the latest updates and delivers an analytical report—perfect for staying ahead of market trends! 🧠📈

See how to install it in your local enviroment: https://www.youtube.com/watch?v=hU1osEz5FzA

## Data Source
The Board of Governors of the Federal Reserve System publishes quarterly tables with the following important points for our research: 
 * Name of bank
 * Bank ID (know as the RSSD ID, which is a unique identifier assigned to institutions by the Federal Reserve)
 * Consolidated assets

Link: https://www.federalreserve.gov/releases/lbr/

<img src="https://github.com/juanchok12/Concentration-of-Banking/assets/116334702/6f25c9ed-cfc7-4301-9d70-e2940b5dd2a7" width="50%" alt="data_source">

I created a automated data pipeline to:
*	Web scraping each quarterly link 
*	Transforming the data from HTML to CSV documents so that I can manipulate the data.
*	Data cleaning: standardization of column headers and data, adding columns that calculate percentages. 
*	Data wrangling: Setting master data frames, pivoting, concatenating dataframes that develop plot ready data frames.
*	Uploads data frames into Github repository.

The data pipeline was originally constructed to feed the "Consolidated Assets" tab of the Concentration of Banking data app (check the app at: https://www.concentration-of-banking.site/). Now, on a quarterly basis, the pipeline also updates the AI Banking Assets data app. 

![pipeline](https://github.com/user-attachments/assets/c30d0abc-8ef6-4458-ad1f-26767c194186)


