# Voice of Customer (VoC) Insight Engine

**[🚀 View Live Demo](https://walmartvoc.streamlit.app/)**

## Overview
The **Voice of Customer (VoC) Insight Engine** is a specialized analytical tool designed to streamline the analysis of customer feedback for merchandising and product teams. 

Merchandising teams are often overwhelmed by the volume of customer reviews, making it difficult to spot critical quality control or supply chain issues buried in the noise. This application solves that problem by automating the analyst workflow—cleaning data, scoring sentiment, and tagging logic—to surface "needles in the haystack" instantly.

The demo uses the **[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)** dataset from Kaggle.

### Key Features
*   **Sentiment Analysis (NLTK VADER):** Goes beyond simple keyword counting by using the Valence Aware Dictionary and sEntiment Reasoner to understand the intensity and context of customer feedback (e.g., distinguishing "not good" from "good").
*   **Automated Issue Tagging:** A custom rule engine scans for high-risk keywords to automatically categorize root causes into actionable buckets such as **Sizing**, **Quality**, **Supply Chain**, and **Pricing**.
*   **Interactive Visualizations:** Provides instant visibility into risk distribution by department and specific drivers of negative sentiment.
*   **Urgent Action Lists:** Automatically flags and isolates negative reviews related to quality or sizing for immediate attention.

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
