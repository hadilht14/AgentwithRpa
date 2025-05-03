# AI-Powered HR Assistant

## Overview

This project provides an AI-powered assistant designed to automate and augment Human Resources (HR) tasks. It leverages LangChain for orchestration and Groq's fast LLM backend for reasoning and generation. The system can process various documents (PDFs, images, audio), generate insightful HR reports, analyze CVs, and includes a framework for human-in-the-loop validation for critical decisions.

The solution includes:
*   A web-based user interface (React) for employee requests and HR validation (details in specific frontend docs).
*   A backend (Django/Python) handling business logic, data, and AI agent interactions.
*   A powerful command-line interface (CLI) for direct access to agent functionalities.

*(Note: The conceptual goal includes future integration with legacy systems via AI-driven RPA, which is not fully implemented in this version.)*

---

## Features

The AI HR Assistant offers a range of capabilities accessible primarily via the CLI (`main.py`):

### 1. Document Processing

*   **PDF Processing:**
    *   Extract text content.
    *   Extract tabular data.
    *   Utilize OCR for scanned or image-based PDFs.
*   **Image Processing:**
    *   Extract text (printed and handwritten).
    *   Preprocess images to improve OCR accuracy.
    *   Detect document types (e.g., CV, invoice, form, ID).
    *   Perform LLM-based image analysis (requires `GROQ_API_KEY`).
*   **Audio Processing:**
    *   Transcribe speech to text from various formats.
    *   Extract basic audio features.
    *   Analyze interview audio for metrics (word count, speaking rate, estimated confidence/stress).
    *   Extract HR-related keywords from transcripts.

### 2. HR Reporting

*   **Pending Decisions:**
    *   View automated decisions awaiting HR validation.
    *   Includes confidence scores and justifications.
*   **HR Summary Reports:**
    *   Generate aggregated reports for specified periods (e.g., "last 7 days", "last 30 days", "current month").
    *   Filter reports by department.
    *   Metrics include: Headcount, hires, terminations, open positions, tenure, absence reasons, training completion, etc.
*   **Attrition Risk Analysis:**
    *   Assess attrition risk for individual employees or company-wide.
    *   Identifies contributing factors.
    *   Provides risk-based recommendations.
*   **Data Visualization (Conceptual):**
    *   Capability to generate data for charts (bar, line, pie) representing HR metrics.
*   **Report Export:**
    *   Output reports to JSON format (current default).
    *   (Future) Export to CSV format.

### 3. CV Analysis

*   Extract key information (skills, experience years, education).
*   Analyze skills match against job descriptions (conceptual).
*   Provide suitability recommendations.

---

## System Architecture

The system uses a multi-layered architecture orchestrated by a LangChain agent.

![System Architecture Diagram](architecture_diagram.jpg)
*(Ensure the image file 'architecture_diagram.png' is present in the repository, ideally in the root or an `assets`/`docs` folder. You might need to adjust the path if stored elsewhere, e.g., `![System Architecture Diagram](./docs/architecture_diagram.png)`)*

**Flow:** User Input -> Ingestion Layer (OCR/STT/Parse) -> LangChain Agent -> (Tools / Groq LLM / HR Validation) -> Final Response -> Integration Layer.

---

## Technology Stack

*   **Frontend:** `React.js`, `TypeScript`, `CSS`, `Material Dashboard React` template
*   **Backend:** `Python 3.x`, `Django`, `Django REST Framework`
*   **Database:** `PostgreSQL` (or other Django-supported DB)
*   **AI / Agent Core:** `LangChain`, `Groq API` (`ChatGroq`), `pytesseract` (OCR), `Whisper` (STT)
*   **Environment:** Python Virtual Environments (`venv`), `Git`

---

## Installation and Setup

### Prerequisites

*   Python (3.9+ recommended)
*   Pip
*   Node.js and npm (for frontend)
*   Git
*   Database System (e.g., PostgreSQL, or SQLite for development)
*   Groq API Key (optional, for LLM-features)

### Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>/backend/
    ```
    *(Replace `<repository_url>` and `<repository_name>` with your actual URL and folder name)*

2.  **Create and activate virtual environment:**
    ```bash
    # Create
    python -m venv venv
    # Activate (Linux/macOS)
    source venv/bin/activate
    # Activate (Windows CMD)
    # venv\Scripts\activate.bat
    # Activate (Windows PowerShell)
    # venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:** Create a `.env` file in the `backend/` directory:
    ```ini
    # backend/.env
    GROQ_API_KEY='your_groq_api_key_here' # Optional, needed for LLM features
    DATABASE_URL='your_database_connection_string' # e.g., sqlite:///db.sqlite3 or postgresql://user:pass@host:port/dbname
    SECRET_KEY='generate_a_strong_random_secret_key_here' # Replace with a real secret key
    DEBUG=True # Set to False in production
    ```
    *(Generate a secure `SECRET_KEY`. You can use Django's utilities or online generators.)*

5.  **Configure Database:** Update `django_project/settings.py` if needed (especially if not using `DATABASE_URL` directly via dj-database-url package). Ensure the database specified in `DATABASE_URL` exists.

6.  **Apply Migrations:**
    ```bash
    python manage.py migrate
    ```

7.  **(Optional) Create Superuser:** (For accessing Django Admin)
    ```bash
    python manage.py createsuperuser
    ```

8.  **Run Server:**
    ```bash
    python manage.py runserver
    ```
    *(The backend should now be running, typically at `http://127.0.0.1:8000/`)*

### Frontend Setup

1.  **Navigate to frontend directory:**
    ```bash
    # From the repository root
    cd material-dashboard-react/
    ```
    *(Adjust path if your frontend code is in a different directory)*

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Configure API Endpoint:** Create a `.env` file in the `material-dashboard-react/` directory (or your frontend root):
    ```ini
    # material-dashboard-react/.env
    REACT_APP_API_BASE_URL='http://localhost:8000/api' # Adjust if backend runs elsewhere or API path is different
    ```

4.  **Start Server:**
    ```bash
    npm start
    ```
    *(The frontend should now be running, typically at `http://localhost:3000/`)*

---

## Usage

Interaction with the system is primarily via the Web Interface or the Command-Line Interface (CLI).

### Web Interface

Access the running React application in your browser (typically `http://localhost:3000`). It provides dashboards for employees and HR personnel to submit requests, view data, and perform validations. (Refer to specific UI documentation or explore the interface for details).

### Command-Line Interface (CLI)

The `main.py` script in the `backend/` directory allows direct interaction with the agent's core functions. *(Ensure your backend virtual environment is activated before running these commands)*.

**General Structure:** `python main.py <command> [arguments/options]`

**Commands:**

1.  **`process <file_path>`:** Processes a given file.
    *   `--type {pdf|image|audio}`: (Required) Specifies the file type.
    *   `--query "<query>"`: (Optional) Ask a specific question about the file content (use quotes for multi-word queries).
    *   *Example (PDF Processing):*
        ```bash
        python main.py process "C:\path\to\your\document.pdf" --type pdf
        ```
        *(Output: Shows extracted text content, tables if any, and metadata. The example output for MedConnect.pdf you provided shows the extracted text and metadata.)*
        ```json
        {
          "success": true,
          "text_content": "Project version: Med Connect (2024_2025)\nReporting\nPrepared for:\nProf Nadia GAHLAM-ENSIA\n...",
          "tables": null,
          "metadata": {
            "Title": "MedConnect",
            "Producer": "Skia/PDF m137 Google Docs Renderer"
          }
        }
        ```

2.  **`report <report_type>`:** Generates various HR reports. Outputs JSON to console. *(Currently uses demo data unless connected to a populated DB)*.
    *   `<report_type>`: (Required) One of `pending_decisions`, `hr_summary`, `attrition_risk`.
    *   `--employee_id <id>`: Filter by employee ID (mainly for `attrition_risk`).
    *   `--period <period>`: Filter `hr_summary` by time (e.g., `"last 7 days"`, `"last 30 days"`, `"current month"`, `"2023"`). Use quotes for periods with spaces.
    *   `--department <department>`: Filter `hr_summary` by department name.
    *   *Example (Pending Decisions):*
        ```bash
        python main.py report pending_decisions
        ```
        *(Output: Shows decisions needing validation. The example output you provided is shown below)*
        ```json
        {
          "pending_decisions": [
            {
              "id": "dec_001",
              "action_name": "evaluate_leave_request",
              "parameters": {
                "employee_email": "john.doe@example.com",
                "details": "Annual leave from June 5-12",
                "task_id": "task_001"
              },
              "confidence_score": 0.85,
              "justification": "Employee has accumulated sufficient leave days. Team coverage available.",
              "status": "Pending Validation",
              "timestamp": 1746265501.235294
            },
            {
              "id": "dec_002",
              "action_name": "evaluate_promotion_request",
              "parameters": {
                "employee_email": "jane.smith@example.com",
                "details": "Promotion to Senior Developer",
                "task_id": "task_002"
              },
              "confidence_score": 0.65,
              "justification": "Employee has met performance metrics. Position available in budget.",
              "status": "Pending Validation",
              "timestamp": 1746265501.235294
            }
          ]
        }
        ```
    *   *Example (HR Summary):*
        ```bash
        python main.py report hr_summary --period "last 30 days" --department "Engineering"
        ```
    *   *Example (Attrition Risk):*
        ```bash
        python main.py report attrition_risk --employee_id "emp123"
        ```

3.  **`analyze_cv <cv_path>`:** Parses and analyzes a CV file.
    *   *Example:*
        ```bash
        python main.py analyze_cv /path/to/candidate_resume.pdf
        ```
        *(Output: JSON containing extracted skills, experience, education, etc.)*

---

## Future Work

*   Implement AI-driven RPA module for interacting with legacy UIs.
*   Integrate with a real database system and replace demo data logic.
*   Expand the range of LangChain tools for more HR tasks (calendaring, performance review analysis, etc.).
*   Add CSV export option for reports.
*   Refine UI/UX based on user feedback.
*   Develop comprehensive automated tests (unit, integration, end-to-end).

---

*(License information could be added here if applicable, e.g., MIT License, Apache 2.0)*
