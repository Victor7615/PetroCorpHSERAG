
-----

# HSE Compliance RAG Assistant ⚖️

This project is a sophisticated AI-powered chat assistant designed to help employees of "PetroSafe Global Holdings" understand and comply with the company's Health, Safety, and Environment (HSE) policies.

It uses a **Retrieval-Augmented Generation (RAG)** architecture to provide reliable, accurate, and context-aware answers grounded in a specific policy document. This approach minimizes AI "hallucinations" and ensures that all responses are based on the official source of truth.

The application is built with Python and Streamlit, powered by the **OpenAI API** and a local **FAISS** vector store for the backend, and deployed live on **Render**.

## 🏛️ Architecture Overview

The application follows a modern RAG pattern:

1.  **Frontend**: A user-friendly chat interface built with **Streamlit**.
2.  **Backend Logic**: The Streamlit app orchestrates the RAG process.
3.  **AI Services & Search**:
      * **OpenAI API**: Provides both the embedding model (`text-embedding-3-small`) to convert text to vectors and the chat model (`gpt-4o`) for generating answers.
      * **FAISS (in-memory)**: Stores the indexed policy document and performs rapid vector searches locally within the application.
4.  **Deployment**: The final application is hosted as a Web Service on **Render**.

-----

##  Features

  * **Grounded Answers**: Responses are based solely on the provided HSE policy document, ensuring accuracy.
  * **Intuitive Chat Interface**: A simple, interactive UI for asking compliance questions.
  * **Secure Configuration**: All API keys are managed securely using environment variables.
  * **Scalable Deployment**: Hosted on Render for reliable, continuous access.
  * **Built-in Safety**: The assistant includes guardrails to prevent it from giving unauthorized advice and directs users to supervisors for complex issues.

-----

## 🛠️ Tech Stack

  * **Language**: Python 3.9+
  * **Frontend**: Streamlit
  * **AI Services**: OpenAI API
  * **Vector Search**: FAISS (Facebook AI Similarity Search)
  * **Data Processing**: Langchain, PyPDF2
  * **Deployment**: Render, Git & GitHub

-----

##  Part 1: Setting Up the OpenAI Backend

This is the foundation of the project. The setup is simpler than the full Azure version, as you only need to generate a single API key.

**Prerequisites:** An active OpenAI account with billing set up.

### Step 1: Get Your OpenAI API Key

1.  Navigate to the [OpenAI Platform](https://platform.openai.com/api-keys).
2.  Log in and go to the **API Keys** section in the left menu.
3.  Click **"Create new secret key"**.
4.  Give your key a descriptive name (e.g., `HSE-RAG-Project-Key`) and click **"Create secret key"**.
5.  **Important**: Copy the key immediately and save it somewhere secure. You will not be able to see the full key again after closing the window. This key will be your `OPENAI_API_KEY` value.

-----

##  Part 2: Running the Streamlit App Locally

Now that you have your API key, you can run the Streamlit application on your local machine.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### Step 2: Set Up a Virtual Environment (Recommended)

```bash
# Using Conda
conda create -n hse-app python=3.9
conda activate hse-app
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Your Environment Variables

1.  Create a new file in the root of your project named `.env`.
2.  Add your OpenAI credentials. **Do not use quotes for the key.**

<!-- end list -->

```env
# .env file
# You created this in Part 1
OPENAI_API_KEY="sk-YourSecretOpenAIKeyGoesHere"

# You choose these from OpenAI's available models
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
OPENAI_CHAT_MODEL="gpt-4o"
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

Your application should now be running in a new browser tab\!

-----

##  Part 3: Deploying to Render

The final step is to deploy your application to make it publicly accessible.

### Step 1: Prepare Your Repository

Ensure your project has the following files at the root level:

  * `app.py` (your application code)
  * `requirements.txt` (your dependencies)
  * `.gitignore` (to exclude `.env`, `__pycache__/`, etc.)
 



### Step 2: Create a New Web Service on Render

1.  Log in to your Render account and create a **New Web Service**.
2.  Connect your GitHub repository.
3.  Configure the settings:
      * **Name**: A unique name for your app (e.g., `hse-assist`).
      * **Build Command**: `pip install -r requirements.txt`
      

### Step 3: Add Environment Variables

This is the most critical step. You must add the key-value pairs from your `.env` file to Render's secure environment variable manager.

1.  In your service dashboard, go to the **Environment** tab.
2.  Click **Add Environment Variable** for each key-value pair.
3.  Enter the keys and values exactly as they are in your `.env` file, **without quotes**.

### Step 4: Deploy

Click **Create Web Service**. Render will build and deploy your application. Once complete, you'll have a public URL to access your HSE Compliance Assistant.

-----

## 📂 Project Structure

```
.
├── .streamlit/
│   └── config.toml
├── .gitignore
├── Rag_app.py
├── HSE_Policy.pdf
├── requirements.txt
└── README.md
```
