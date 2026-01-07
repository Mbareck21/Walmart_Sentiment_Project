# Walmart Sentiment Project

## Overview
This is a Streamlit application that analyzes customer reviews using NLTK (VADER) to determine sentiment and categorize issues (e.g., Sizing, Quality, Supply Chain).

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    *   Windows: `.\venv\Scripts\activate`
    *   Mac/Linux: `source venv/bin/activate`

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**
    ```bash
    python -m streamlit run app.py
    ```

2.  **Explore Data:**
    *   Click **"🚀 Use Sample Data"** to instantly see the dashboard using the built-in dataset.
    *   Alternatively, upload your own CSV file. The app will automatically look for columns containing "Review Text" and "Department Name".

## License
© 2026 LEMINE MBARECK. This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
