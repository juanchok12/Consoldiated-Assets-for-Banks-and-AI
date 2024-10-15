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

## Table & Filtering
The **AI Banking Assets** application provides a highly interactive tabular view of the data that can be filtered based on various parameters. Users can filter the table by:

- **Quarter:** A dropdown menu allows users to select the quarter they wish to view, dynamically updating the table to display only data for the chosen period.
- **Bank Name:** A text input box where users can enter part or all of a bank’s name to filter results, making it easy to locate specific banks.
- **Consolidated Assets:** A slide bar allows users to filter the table based on the range of consolidated assets. This feature is helpful for focusing on banks within specific asset ranges.

In addition, the app includes a **Download CSV** button that enables users to export the filtered table data to a CSV file, making it easy to download and analyze the data locally.

## AI Agent for Data Visualization
One of the key features of this application is the integration of an **AI Agent** that automates the process of data visualization. The AI agent:

- Uses **OpenAI’s GPT-4** model, one of the most reliable language models, for prompt engineering.
- Ensures standardized outputs despite LLM variability, by using prompt engineering techniques that minimize output variation in the data visualizations.
- Automatically generates interactive **Plotly** visualizations based on the selected bank and financial metrics (such as consolidated assets).

Users simply enter a prompt, and the AI agent creates a chart and provides the corresponding Python code that was used to generate it. This allows users to modify or reuse the code as needed.

## AI Agentic System for News Analysis
The **AI Agentic System** within the app is designed to deliver insightful financial news and analysis for the selected banks. This system includes two main roles:

1. **Senior Research Analyst:**
   - **Goal:** Uncover the latest trends and news about the selected banks.
   - **Backstory:** With years of experience at a leading banking think tank, the analyst specializes in analyzing complex data and delivering actionable insights.

2. **Banking Content Strategist:**
   - **Goal:** Create compelling content from the analyst's findings, summarizing complex banking data into easy-to-understand reports.
   - **Backstory:** A highly renowned content strategist, this role focuses on converting detailed financial analyses into engaging and reader-friendly reports.

The AI Agentic System generates comprehensive reports like the following:

### JPMorgan Chase & Co. Analysis and Insights  
**Key Takeaways:**
- Reported a decrease in net income for Q3 2024 to $12.9 billion, down from $13.2 billion in Q3 2023.
- Earnings per share (EPS) increased from $4.33 to $4.37.
- Increased credit costs are impacting the bank’s performance.
- JPMorgan’s stock is expected to reach new highs in 2024, reflecting optimism in its strategic direction.

**Full Analysis:**
In the third quarter of 2024, JPMorgan Chase & Co. showcased a mixed yet promising financial performance. Although the bank reported a slight dip in net income, it demonstrated resilience with an increase in EPS, suggesting efficiency improvements. As JPMorgan navigates the rising credit costs, it continues to strategically invest in long-term growth, positioning itself for a potential rise in stock value throughout 2024. Analysts remain optimistic about the bank's future performance based on its better-than-expected results.

---

Sources:
- [Market Screener](https://www.marketscreener.com/quote/stock/JPMORGAN-CHASE-CO-37468997/news/JPMorgan-Chase-JPMC-Co-Form-8-K-3Q2024-earning-release-11-October-2024-48058728/)
- [Yahoo Finance](https://finance.yahoo.com/news/jpmorgan-chase-co-jpm-q3-070028462.html)
- [Nasdaq](https://www.nasdaq.com/articles/jpmorgan-can-hit-new-highs-year-heres-why)
