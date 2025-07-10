# ReceiptAgent

A Python-based agent for automated receipt PDF processing using OCR, data extraction, validation, and human-in-the-loop review. The project leverages LangChain, Google Vertex AI, and Gradio for a robust, semi-automated workflow.

## Features
- Monitors a folder for new receipt PDFs
- Extracts and crops receipt images from PDFs
- Uses Gemini/Vertex AI for data extraction
- Validates extracted data
- Human-in-the-loop review via Gradio for failed/uncertain cases
- Organizes processed PDFs and logs extracted data

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/parvezamm3/receipt_agent.git
cd ReceiptAgent
```

### 2. Set Up the Python Virtual Environment (venv)
Create and activate the environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with your Google Gemini API key:
```
GEMINI_API_KEY=your_google_gemini_api_key
```

### 5. Prepare Folders
The following folders are used by the agent and will be created automatically if missing:
- `pdfs/` (input PDFs)
- `images/` (cropped images)
- `output_pdfs/` (processed PDFs)
- `success_pdfs/` (successfully processed PDFs)
- `error_pdfs/` (PDFs needing human review)

### 6. Run the Agent
```bash
python agent_controller.py
```
The agent will monitor the `pdfs/` folder for new PDFs and process them automatically.

### 7. Running Tests
Tests use dummy data and mock the actual tool functionalities, so no real files or API calls are made.

To run all tests:
```bash
python -m unittest test_agent_controller.py -v
python -m unittest test_tools.py -v
```

## Notes
- Make sure your `service_account_key.json` and any other required credentials are in place if you use Google APIs.
- For production, update the repository URL in the clone step above.
- The project is designed for Windows but should be portable with minor changes.

## License
MIT License 