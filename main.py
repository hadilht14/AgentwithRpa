# main.py
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple # Added Tuple
from uuid import uuid4
import time
import json
import io # Added io for potential byte handling
import base64 # Added base64

# Load environment variables from .env file
from dotenv import load_dotenv # type: ignore
load_dotenv()

# LangChain imports - UPDATED FOR LATEST VERSION
from langchain_groq import ChatGroq # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # type: ignore
from langchain_core.tools import tool, StructuredTool # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore # Keep Memory import if needed later
from langchain_core.pydantic_v1 import BaseModel, Field # type: ignore # Use Pydantic v1 from langchain

# Updated imports for agents
from langchain.agents import AgentExecutor, create_react_agent # type: ignore

# --- Import Custom Modules ---
# Ensure these files exist in the same directory or adjust sys.path if needed
try:
    from audio_extractor import AudioExtractor
    from image_extractor import ImageExtractor
    from pdf_extractor import PDFExtractor
    from hr_reports import HRReportGenerator
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure audio_extractor.py, image_extractor.py, pdf_extractor.py, and hr_reports.py exist.")
    exit(1)
# ----------------------------

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Check for API Key ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("FATAL: GROQ_API_KEY environment variable not set.")
    logger.error("Please create a .env file in the project root and add GROQ_API_KEY='your_key'")
    exit("GROQ_API_KEY not found. Exiting.")
else:
    logger.info("GROQ API Key loaded.")

# --- Simulated Data (Replace with a real database in a production app) ---
fake_db = {
    "users": {
        "hr_user@example.com": {"password": "hr_password", "role": "HR"},
        "emp_user@example.com": {"password": "emp_password", "role": "Employee"},
    },
    "decisions": {}, # Store AI decisions {decision_id: decision_data}
    "tasks": { # Store employee tasks/requests
        "emp_user@example.com": [
            {"id": "task_1", "description": "Submit performance review", "status": "Pending"},
            {"id": "leave_1", "description": "Leave Application (Aug 10-12)", "status": "Approved"},
        ]
    },
    "notifications": { # Store notifications
        "emp_user@example.com": [
            {"id": "notif_1", "message": "Salary for July has been disbursed.", "timestamp": "2023-08-01T10:00:00Z"},
        ]
    },
    "agent_log": [], # Log agent actions for the live terminal
    "automated_actions": [], # Log automated tasks
    "extracted_data_cache": {}, # Simple cache for data extracted from files {cache_id: cache_entry}
}

# --- Instantiate Custom Modules ---
# Pass the fake_db reference to the modules so they can interact with it (e.g., for caching, decision creation)
try:
    audio_extractor = AudioExtractor(fake_db)
    image_extractor = ImageExtractor(fake_db)
    pdf_extractor = PDFExtractor(fake_db)
    hr_report_generator = HRReportGenerator(fake_db)
    logger.info("Custom Extractor and Report Generator modules initialized.")
except Exception as e:
    logger.error(f"Error initializing custom modules: {e}", exc_info=True)
    exit("Failed to initialize custom modules. Exiting.")
# ---------------------------------


# --- Define Tool Input Schemas using Pydantic ---

# Schemas for existing tools (keep these)
class SubmitRequestInput(BaseModel):
    user_email: str = Field(description="The email address of the employee submitting the request.")
    request_type: str = Field(description="The type of request being submitted (e.g., 'Leave Application', 'Query').")
    details: str = Field(description="Specific details of the request (e.g., dates for leave, question text).")

class CreateDecisionInput(BaseModel):
    action_name: str = Field(description="A descriptive name for the action being proposed (e.g., 'evaluate_leave_request', 'process_form_image').")
    parameters: Dict[str, Any] = Field(description="Parameters associated with the proposed action.")
    confidence_score: float = Field(description="AI's confidence score (0.0 to 1.0) in the analysis or proposal.")
    justification: str = Field(description="A brief explanation for why this decision is being proposed.")

class ApproveDecisionInput(BaseModel):
    decision_id: str = Field(description="The unique ID of the decision to approve.")
    notes: Optional[str] = Field(None, description="Optional notes from the HR user approving the decision.")

class RejectDecisionInput(BaseModel):
    decision_id: str = Field(description="The unique ID of the decision to reject.")
    reason: str = Field(description="The reason provided by the HR user for rejecting the decision.")

# --- Schemas for NEW Extractor/Reporter Tools ---

class TranscribeAudioFileInput(BaseModel):
    file_path: str = Field(description="Path to the audio file (e.g., /path/to/meeting.mp3, ./dummy_files/sample_audio.wav)")
    language: str = Field(default="en-US", description="Language code for transcription (e.g., 'en-US', 'es-ES')")

class TranscribeAudioBytesInput(BaseModel):
    # Agent typically won't directly use bytes, but schema is useful
    # In a real app, a file upload mechanism would provide bytes
    audio_bytes_placeholder: str = Field(description="Placeholder: Base64 encoded audio bytes would go here in a real app")
    format_hint: str = Field(description="Audio format hint (e.g., 'wav', 'mp3')")
    language: str = Field(default="en-US", description="Language code for transcription")

class ProcessHRAudioInput(BaseModel):
    transcription_id: str = Field(description="Identifier (usually the original filename) for the cached transcription result to process.")
    # Alternatively, could accept text directly:
    # transcription_text: Optional[str] = Field(None, description="Direct transcription text if not using cache")

class ExtractInterviewEvaluationInput(BaseModel):
    transcription_id: str = Field(description="Identifier (usually the original filename) for the cached interview transcription result.")
    # Alternatively:
    # transcription_text: Optional[str] = Field(None, description="Direct transcription text if not using cache")


class ExtractTextFromImageInput(BaseModel):
    image_path: str = Field(description="Path to the image file (e.g., /path/to/document.png, ./dummy_files/sample_form.png)")
    # image_bytes_placeholder: Optional[str] = Field(None, description="Placeholder for Base64 image bytes") # Agent won't use directly
    is_handwritten: bool = Field(default=False, description="Set to true if the image contains primarily handwritten text")

class ProcessFormImageInput(BaseModel):
    image_path: str = Field(description="Path to the form image file (e.g., ./dummy_files/sample_form.png)")
    # image_bytes_placeholder: Optional[str] = Field(None, description="Placeholder for Base64 image bytes")

class ExtractIdDocumentInput(BaseModel):
    image_path: str = Field(description="Path to the ID document image file")
    # image_bytes_placeholder: Optional[str] = Field(None, description="Placeholder for Base64 image bytes")

class AnalyzeHandwrittenNoteInput(BaseModel):
    image_path: str = Field(description="Path to the handwritten note image file (e.g., ./dummy_files/sample_note.png)")
    # image_bytes_placeholder: Optional[str] = Field(None, description="Placeholder for Base64 image bytes")


class ExtractTextFromPdfInput(BaseModel):
    file_path: str = Field(description="Path to the PDF file (e.g., /path/to/document.pdf, ./dummy_files/sample_cv.pdf)")

class ParseCvInput(BaseModel):
    file_path: str = Field(description="Path to the CV/resume PDF file (e.g., ./dummy_files/sample_cv.pdf)")
    # pdf_bytes_placeholder: Optional[str] = Field(None, description="Placeholder for PDF bytes")
    # text: Optional[str] = Field(None, description="Pre-extracted text from the CV") # Less likely agent use case

class AnalyzeCvForRoleInput(BaseModel):
    cv_file_path: str = Field(description="Path to the CV PDF file to analyze (e.g., ./dummy_files/sample_cv.pdf)")
    role_title: str = Field(description="The title of the role to compare against (e.g., 'Software Engineer')")
    required_skills: List[str] = Field(description="List of essential skills for the role")
    preferred_skills: Optional[List[str]] = Field(default=None, description="List of desirable skills for the role")
    min_years_experience: Optional[int] = Field(default=0, description="Minimum years of experience required")
    education_requirements: Optional[Dict[str, str]] = Field(default=None, description="Required education level and field (e.g., {'level': 'Bachelor', 'field': 'Computer Science'})")

class GenerateDetailedHrReportInput(BaseModel):
    period: str = Field(default="last 7 days", description="Time period for the report (e.g., 'last 7 days', 'last month', 'last quarter', 'year to date')")

class GetAttritionRiskReportInput(BaseModel):
    department: Optional[str] = Field(default=None, description="Optional: Filter the report by a specific department name.")

# --- Define Agent Tools ---

# --- Core Tools (Keep these) ---
def get_user_role_impl(user_email: str) -> str:
    """Retrieves the role (HR or Employee) for a given user email."""
    logger.info(f"Tool Used: get_user_role(user_email='{user_email}')")
    user = fake_db["users"].get(user_email)
    if user:
        return user["role"]
    return "User not found"

get_user_role = StructuredTool.from_function(
    func=get_user_role_impl,
    name="get_user_role",
    description="Retrieves the role (HR or Employee) for a given user email."
    # No args_schema needed as arguments are simple types directly inferred
)

# Deprecated Tool Example - kept for reference if needed
# def generate_hr_weekly_report_impl(period: str = "last 7 days") -> Dict[str, Any]:
#     """DEPRECATED: Use generate_detailed_hr_report instead. Generates a basic summary report for HR."""
#     logger.warning("Tool Used: generate_hr_weekly_report (DEPRECATED). Use generate_detailed_hr_report.")
#     report = hr_report_generator.generate_detailed_hr_report(period=period)
#     simple_report = {k: v for k, v in report.items() if k not in ['visualizations', 'recommendations', 'department_stats']}
#     simple_report["report_summary"] = f"Summary for {period}: {report.get('new_hires', 0)} new hires, {report.get('leave_requests_processed', 0)} leaves processed."
#     return simple_report
# generate_hr_weekly_report = StructuredTool.from_function(
#     func=generate_hr_weekly_report_impl,
#     name="generate_hr_weekly_report",
#     description="DEPRECATED: Use generate_detailed_hr_report instead. Generates a basic summary report for HR."
# )

def predict_employee_attrition_impl(employee_id: Optional[str] = None) -> Dict[str, Any]:
    """Predicts potential employee attrition. If no employee_id is given, provides a general overview via the attrition risk report tool."""
    logger.info(f"Tool Used: predict_employee_attrition(employee_id='{employee_id}')")
    fake_db["agent_log"].append({"action": "predict_employee_attrition", "params": {"employee_id": employee_id}, "timestamp": time.time()})

    if employee_id:
        # Placeholder - integrate with a real model or HRReportGenerator if needed
        # Simulate checking if the ID looks like an email (common user identifier here)
        if "@" in employee_id and employee_id in fake_db["users"]:
            risk_score = 0.75 if employee_id == "emp_user@example.com" else 0.2 # Example risk based on ID
            confidence = 0.8
            recommendation = "Schedule 1:1 meeting." if risk_score > 0.5 else "Monitor engagement."
            return {"employee_id": employee_id, "attrition_risk_score": risk_score, "confidence": confidence, "recommendation": recommendation}
        else:
            return {"error": f"Employee ID '{employee_id}' not found or not in a recognized format."}
    else:
        # Redirect to the attrition report tool for an overview
        # return get_attrition_risk_report_tool_impl() # Directly call - might confuse agent
        # Better: Instruct agent to use the report tool
        return {"message": "For a general overview of attrition risk, please use the 'get_attrition_risk_report' tool."}


predict_employee_attrition = StructuredTool.from_function(
    func=predict_employee_attrition_impl,
    name="predict_employee_attrition",
    description="Predicts potential attrition risk for a specific employee ID (use email). If no ID is provided, guides the user to use 'get_attrition_risk_report'."
)

def list_pending_ai_decisions_impl() -> List[Dict[str, Any]]:
    """Lists AI-made decisions that are pending validation by HR."""
    logger.info("Tool Used: list_pending_ai_decisions()")
    # Use the report generator's method to get the raw list
    decisions = hr_report_generator.get_pending_decisions() # Assume this returns the list
    # Format for display / agent consumption (limit details)
    preview_list = []
    for decision in decisions:
        preview_list.append({
            "id": decision.get("id"),
            "action_name": decision.get("action_name"),
            "confidence": decision.get("confidence_score"),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(decision.get("timestamp", 0))),
            "justification_preview": decision.get("justification", "")[:100] + "..."
        })
    fake_db["agent_log"].append({"action": "list_pending_ai_decisions", "params": {}, "timestamp": time.time()})
    return preview_list

list_pending_ai_decisions = StructuredTool.from_function(
    func=list_pending_ai_decisions_impl,
    name="list_pending_ai_decisions",
    description="Lists summaries of AI-made decisions that are pending validation by HR. Use 'get_decision_details' for full information."
)

def create_ai_decision_impl(action_name: str, parameters: Dict, confidence_score: float, justification: str) -> str:
    """Creates a new AI decision record that requires validation. Usually called internally by other tools."""
    # This function should ideally be part of the modules that generate decisions,
    # but having it as a tool allows the agent to potentially create decisions directly if needed,
    # though it's less common for the user to ask for this explicitly.
    # We can make it less prominent in the agent's prompt if needed.
    logger.info(f"Tool Used: create_ai_decision(action_name='{action_name}', parameters={parameters}, confidence_score={confidence_score})")
    decision_id = f"dec_{uuid4()}"
    fake_db["decisions"][decision_id] = {
        "id": decision_id,
        "action_name": action_name,
        "parameters": parameters,
        "confidence_score": confidence_score,
        "justification": justification,
        "status": "Pending Validation",
        "timestamp": time.time()
    }
    fake_db["agent_log"].append({"action": "create_ai_decision", "params": {"action_name": action_name}, "timestamp": time.time()})
    log_automated_action(f"AI proposed decision '{action_name}' ({decision_id})", "Pending Validation")
    return f"Decision {decision_id} created and awaiting validation."

create_ai_decision = StructuredTool.from_function(
    func=create_ai_decision_impl,
    name="create_ai_decision",
    description="INTERNAL USE PRIMARILY: Creates a new AI decision record that requires validation. Usually called automatically by other analysis tools.",
    args_schema=CreateDecisionInput # Add schema here
)

def get_decision_details_impl(decision_id: str) -> Dict[str, Any]:
    """Retrieves detailed information about a specific decision."""
    logger.info(f"Tool Used: get_decision_details(decision_id='{decision_id}')")
    decision = fake_db["decisions"].get(decision_id)
    if not decision:
        return {"error": f"Decision with ID '{decision_id}' not found."}
    fake_db["agent_log"].append({"action": "get_decision_details", "params": {"decision_id": decision_id}, "timestamp": time.time()})
    # Return a copy to avoid modifying the original dict accidentally
    return decision.copy()

get_decision_details = StructuredTool.from_function(
    func=get_decision_details_impl,
    name="get_decision_details",
    description="Retrieves detailed information about a specific decision using its ID.",
    # args_schema inferred
)

def approve_decision_impl(decision_id: str, notes: Optional[str] = None) -> str:
    """Approves a pending AI decision. Only HR users should be able to call this."""
    logger.info(f"Tool Used: approve_decision(decision_id='{decision_id}', notes='{notes}')")

    # Note: Role check should ideally happen before calling the tool (e.g., in the agent logic or UI)
    # but we add a basic check here as a safeguard. In a real app, this would be more robust.
    # user_email = ... # Need context for who is calling this - Agent needs to provide it?
    # The agent prompt asks it to consider the role, so we assume it won't call this for an Employee.

    if decision_id not in fake_db["decisions"]:
        return f"Error: Decision {decision_id} not found."

    decision = fake_db["decisions"][decision_id]
    if decision["status"] != "Pending Validation":
        return f"Error: Decision {decision_id} is already '{decision['status']}', not pending validation."

    # --- Logic for executing the approved action ---
    action_name = decision["action_name"]
    params = decision["parameters"]
    execution_result = f"Action '{action_name}' approved but no specific execution logic implemented yet."
    executed_successfully = False

    try:
        if action_name == "evaluate_leave_request":
            employee_email = params.get("employee_email")
            task_id = params.get("task_id")
            if employee_email and task_id and employee_email in fake_db["tasks"]:
                task_found = False
                for task in fake_db["tasks"][employee_email]:
                    if task["id"] == task_id:
                        task["status"] = "Approved"
                        task_found = True
                        # Add notification
                        if employee_email not in fake_db["notifications"]:
                            fake_db["notifications"][employee_email] = []
                        notif_id = f"notif_{uuid4()}"
                        fake_db["notifications"][employee_email].append({
                            "id": notif_id,
                            "message": f"Your leave request has been approved: {task['description']}",
                            "timestamp": time.time()
                        })
                        execution_result = f"Leave request task {task_id} updated to Approved and notification sent to {employee_email}."
                        executed_successfully = True
                        break
                if not task_found:
                    execution_result = f"Could not find associated leave task {task_id} for {employee_email} to update status."
            else:
                execution_result = "Could not find associated leave task details (employee_email or task_id) in decision parameters."

        elif action_name in ["process_hr_audio_recording", "process_form_image", "process_id_verification_document", "analyze_handwritten_note", "evaluate_cv_for_role"]:
            # Placeholder: In a real app, these might trigger workflows like:
            # - Storing extracted data in employee profiles
            # - Creating follow-up tasks for HR
            # - Sending summary emails
            # - Updating application status in an Applicant Tracking System (ATS)
            execution_result = f"Approved decision for '{action_name}' processed (simulated). Parameters: {params}"
            # Add notification for HR or involved parties if needed
            # e.g., fake_db["notifications"]["hr_user@example.com"].append(...)
            executed_successfully = True # Simulate success

        else:
            execution_result = f"Approved action '{action_name}' executed (simulated - no specific logic defined)."
            executed_successfully = True # Simulate success

        # Update decision status *after* attempting execution
        if executed_successfully:
            decision["status"] = "Approved"
            decision["approval_timestamp"] = time.time()
            decision["approval_notes"] = notes
            decision["execution_result"] = execution_result # Store result
            fake_db["agent_log"].append({"action": "approve_decision", "params": {"decision_id": decision_id, "status": "Success"}, "timestamp": time.time()})
            log_automated_action(f"Decision {decision_id} approved and executed", "Completed", details=execution_result)
            return f"Decision {decision_id} approved. Result: {execution_result}"
        else:
            # Keep pending or move to error state? Let's keep pending and report error.
            # decision["status"] = "Approval Execution Failed" # Option
            fake_db["agent_log"].append({"action": "approve_decision", "params": {"decision_id": decision_id, "status": "Execution Failed"}, "timestamp": time.time()})
            log_automated_action(f"Decision {decision_id} approved but execution failed", "Error", details=execution_result)
            # Don't change status, let HR retry or investigate
            return f"Decision {decision_id} approved, but an error occurred during automated execution: {execution_result}. Decision remains 'Pending Validation'."


    except Exception as e:
        logger.error(f"Error executing approved decision {decision_id} ({action_name}): {str(e)}", exc_info=True)
        # Optionally revert status or mark as failed execution
        # decision["status"] = "Approval Failed" # Or keep Approved but log error
        decision["execution_error"] = str(e) # Log the error within the decision itself
        log_automated_action(f"Decision {decision_id} approval failed during execution", "Error", details=str(e))
        fake_db["agent_log"].append({"action": "approve_decision", "params": {"decision_id": decision_id, "status": "Exception"}, "timestamp": time.time()})
        # Keep pending status to allow investigation
        return f"Decision {decision_id} approved, but a critical error occurred during execution: {str(e)}. Decision remains 'Pending Validation'."


approve_decision = StructuredTool.from_function(
    func=approve_decision_impl,
    name="approve_decision",
    description="Approves a pending AI decision, triggering the associated action (e.g., updating task status, sending notification). Only for HR users.",
    args_schema=ApproveDecisionInput # Add schema
)

def reject_decision_impl(decision_id: str, reason: str) -> str:
    """Rejects a pending AI decision. Only HR users should be able to call this."""
    logger.info(f"Tool Used: reject_decision(decision_id='{decision_id}', reason='{reason}')")
    # Role check assumption as per approve_decision

    if decision_id not in fake_db["decisions"]:
        return f"Error: Decision {decision_id} not found."

    decision = fake_db["decisions"][decision_id]
    if decision["status"] != "Pending Validation":
        return f"Error: Decision {decision_id} is already '{decision['status']}', not pending validation."

    # Update decision status first
    decision["status"] = "Rejected"
    decision["rejection_timestamp"] = time.time()
    decision["rejection_reason"] = reason

    # --- Update related tasks/notify if necessary ---
    action_name = decision["action_name"]
    params = decision["parameters"]
    rejection_action_result = f"Action '{action_name}' rejected."
    action_taken = False

    try:
        if action_name == "evaluate_leave_request":
            employee_email = params.get("employee_email")
            task_id = params.get("task_id")
            if employee_email and task_id and employee_email in fake_db["tasks"]:
                task_found = False
                for task in fake_db["tasks"][employee_email]:
                    if task["id"] == task_id:
                        task["status"] = "Rejected" # Update task status
                        task_found = True
                        # Add notification
                        if employee_email not in fake_db["notifications"]:
                            fake_db["notifications"][employee_email] = []
                        notif_id = f"notif_{uuid4()}"
                        fake_db["notifications"][employee_email].append({
                            "id": notif_id,
                            "message": f"Your leave request ({task['description']}) has been rejected. Reason: {reason}",
                            "timestamp": time.time()
                        })
                        rejection_action_result = f"Leave request task {task_id} updated to Rejected and notification sent to {employee_email}."
                        action_taken = True
                        break
                if not task_found:
                    rejection_action_result = f"Could not find associated leave task {task_id} for {employee_email} to update status upon rejection."
            else:
                rejection_action_result = "Could not find associated leave task details in decision parameters for rejection processing."

        # Add other rejection logic if needed (e.g., notifying applicant for CV rejection)
        elif action_name == "evaluate_cv_for_role":
            applicant_email = params.get("applicant_email") # Assuming email is stored
        if applicant_email:
                # Add notification (or simulate sending rejection email)
                if applicant_email not in fake_db["notifications"]: fake_db["notifications"][applicant_email] = []
                notif_id = f"notif_{uuid4()}"
                fake_db["notifications"][applicant_email].append({
                    "id": notif_id,
                    "message": f"Thank you for your application for {params.get('role_title')}. After careful consideration, we will not be moving forward at this time. Rejection reason: {reason}",
                    "timestamp": time.time()
                })
                rejection_action_result = f"CV evaluation rejected. Notification sent to applicant {applicant_email}."
                action_taken = True
        else:
            rejection_action_result = f"CV evaluation rejected. Applicant email not found in parameters, notification not sent."

        decision["rejection_action_result"] = rejection_action_result # Store result
        fake_db["agent_log"].append({"action": "reject_decision", "params": {"decision_id": decision_id}, "timestamp": time.time()})
        log_automated_action(f"Decision {decision_id} rejected", "Completed", details=f"Reason: {reason}. {rejection_action_result}")
        return f"Decision {decision_id} has been rejected. {rejection_action_result}"

    except Exception as e:
        logger.error(f"Error processing rejection actions for decision {decision_id} ({action_name}): {str(e)}", exc_info=True)
        decision["rejection_error"] = str(e)
        log_automated_action(f"Decision {decision_id} rejection processing failed", "Error", details=str(e))
        # Status is already 'Rejected', but log the error
        return f"Decision {decision_id} rejected, but a critical error occurred during related actions: {str(e)}"


reject_decision = StructuredTool.from_function(
    func=reject_decision_impl,
    name="reject_decision",
    description="Rejects a pending AI decision, updating related tasks/notifications. Only for HR users.",
    args_schema=RejectDecisionInput # Add schema
)

def get_employee_tasks_impl(user_email: str) -> List[Dict[str, Any]]:
    """Retrieves the task history and status for a specific employee."""
    logger.info(f"Tool Used: get_employee_tasks(user_email='{user_email}')")
    # Role check: Allow HR to see anyone's tasks? Or only employee seeing their own?
    # Current setup allows anyone to query if they know the email. Assume agent uses context email.
    fake_db["agent_log"].append({"action": "get_employee_tasks", "params": {"user_email": user_email}, "timestamp": time.time()})
    tasks = fake_db["tasks"].get(user_email, [])
    return [task.copy() for task in tasks] # Return copies

get_employee_tasks = StructuredTool.from_function(
    func=get_employee_tasks_impl,
    name="get_employee_tasks",
    description="Retrieves the task history and status for a specific employee using their email address."
    # args_schema inferred
)

def get_employee_notifications_impl(user_email: str) -> List[Dict[str, Any]]:
    """Retrieves notifications for a specific employee."""
    logger.info(f"Tool Used: get_employee_notifications(user_email='{user_email}')")
    # Role check as above
    fake_db["agent_log"].append({"action": "get_employee_notifications", "params": {"user_email": user_email}, "timestamp": time.time()})
    notifications = fake_db["notifications"].get(user_email, [])
    # Sort by timestamp descending?
    try:
        sorted_notifications = sorted(
            notifications,
            key=lambda x: x.get('timestamp', 0) if isinstance(x.get('timestamp'), (int, float)) else 0,
            reverse=True
        )
    except Exception: # Handle cases where timestamp might be missing or wrong type
        sorted_notifications = notifications # Fallback to original order
    return [notif.copy() for notif in sorted_notifications] # Return copies

get_employee_notifications = StructuredTool.from_function(
    func=get_employee_notifications_impl,
    name="get_employee_notifications",
    description="Retrieves recent notifications for a specific employee using their email address."
    # args_schema inferred
)

def submit_employee_request_impl(user_email: str, request_type: str, details: str) -> str:
    """Allows an employee to submit a request (e.g., leave application). If it's a type requiring evaluation (like leave), it automatically triggers an AI decision proposal for HR validation."""
    logger.info(f"Tool Used: submit_employee_request(user_email='{user_email}', request_type='{request_type}', details='{details}')")

    if user_email not in fake_db["users"] or fake_db["users"][user_email]["role"] != "Employee":
        # Basic authorization check
        # return f"Error: User {user_email} not found or is not an Employee."
        # Agent should ideally not call this for HR user, but let it proceed if it does.
        logger.warning(f"Request submitted by non-employee or unknown user: {user_email}")
        # Or should HR be able to submit on behalf of someone? For now, let it pass.

    if user_email not in fake_db["tasks"]:
        fake_db["tasks"][user_email] = []
    task_id = f"task_{uuid4()}"
    task_description = f"{request_type}: {details}"


    new_task = {
        "id": task_id,
        "description": task_description,
        "status": "Submitted - Pending Review", # Initial status
        "timestamp": time.time()
    }
    fake_db["tasks"][user_email].append(new_task)
    fake_db["agent_log"].append({"action": "submit_employee_request", "params": {"user_email": user_email, "request_type": request_type}, "timestamp": time.time()})

    # *** AI Decision Proposal Trigger ***
    # Convert request type to lower for robust checking
    normalized_request_type = request_type.lower()
    if "leave" in normalized_request_type or "time off" in normalized_request_type:
        # Use the create_ai_decision tool/function
        try:
            # Pass relevant info to the decision creator
            decision_result_msg = create_ai_decision_impl(
                action_name="evaluate_leave_request",
                parameters={
                    "employee_email": user_email,
                    "details": details,
                    "task_id": task_id,
                    "request_type": request_type # Include original request type
                },
                confidence_score=0.65, # Simulate medium confidence needing review
                justification="Standard leave request submitted via employee portal/agent. Requires validation of available leave balance, policy compliance, and team coverage."
            )
            # Update task status to reflect AI review initiation
            new_task["status"] = "Pending HR Validation"
            log_automated_action(f"Leave request {task_id} submitted by {user_email}", "Pending Validation", details=decision_result_msg)
            return f"Request {task_id} submitted and is now pending HR validation. {decision_result_msg}"
        except Exception as e:
            logger.error(f"Failed to create AI decision for leave request {task_id}: {str(e)}", exc_info=True)
            log_automated_action(f"Leave request {task_id} submitted by {user_email}", "Error during AI processing", details=str(e))
            # Keep status as 'Pending Review' if AI decision fails
            new_task["status"] = "Submitted - Review Needed (AI Error)"
            return f"Request {task_id} submitted, but an error occurred during initial processing. It will be reviewed manually by HR."
    # --- END AI Decision Trigger ---

    # Default case for requests not triggering AI decision
    log_automated_action(f"Request {task_id} ({request_type}) submitted by {user_email}", "Completed")
    return f"Request {task_id} submitted successfully and is pending review by HR."

submit_employee_request = StructuredTool.from_function(
    func=submit_employee_request_impl,
    name="submit_employee_request",
    description="Allows an employee to submit a request (e.g., 'Leave Application', 'IT Support', 'Policy Query'). Automatically triggers AI analysis and HR decision proposal for certain types like leave/time off requests.",
    args_schema=SubmitRequestInput # Add schema
)

# --- NEW Tools from Extractors and Reporters ---

# Audio Extractor Tools
def transcribe_audio_file_tool_impl(file_path: str, language: str = "en-US") -> Dict[str, Any]:
    """Transcribes speech from an audio file (e.g., MP3, WAV, OGG). Stores result."""
    logger.info(f"Tool Used: transcribe_audio_file(file_path='{file_path}', language='{language}')")
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found at path: {file_path}"}

    try:
        result = audio_extractor.transcribe_audio_file(file_path, language)
        if "text" in result and not result.get("error"):
            # Cache the result for potential follow-up processing
            cache_id = os.path.basename(file_path) # Use filename as cache key
            fake_db["extracted_data_cache"][cache_id] = {
                "type": "audio_transcription",
                "data": result, # Store the full transcription result
                "timestamp": time.time(),
                "original_path": file_path # Store original path for reference
                }
            log_automated_action(f"Audio transcribed: {file_path}", "Completed", details=f"Cache ID: {cache_id}")
            return {"status": "success", "transcription_id": cache_id, "preview": result["text"][:150] + "...", "duration_seconds": result.get("duration")}
        else:
            log_automated_action(f"Audio transcription failed: {file_path}", "Error", details=result.get("error"))
            return {"status": "error", "message": result.get("error", "Unknown transcription error")}
    except Exception as e:
        logger.error(f"Error in transcribe_audio_file tool for {file_path}: {e}", exc_info=True)
        log_automated_action(f"Audio transcription failed: {file_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during transcription: {e}"}

transcribe_audio_file_tool = StructuredTool.from_function(
    func=transcribe_audio_file_tool_impl,
    name="transcribe_audio_file",
    description="Transcribes speech from an audio file path (supports common formats like MP3, WAV, M4A). Verifies file existence. Returns a status and a transcription ID (usually the filename) for further processing.",
    args_schema=TranscribeAudioFileInput
)

def process_hr_audio_tool_impl(transcription_id: str) -> Dict[str, Any]:
    """Processes a previously transcribed HR audio recording for topics, actions, dates."""
    logger.info(f"Tool Used: process_hr_audio(transcription_id='{transcription_id}')")
    cached_data_entry = fake_db["extracted_data_cache"].get(transcription_id)

    if not cached_data_entry:
        return {"status": "error", "message": f"Transcription ID '{transcription_id}' not found in cache. Please transcribe the file first using 'transcribe_audio_file'."}
    if cached_data_entry["type"] != "audio_transcription":
        return {"status": "error", "message": f"Cache ID '{transcription_id}' does not correspond to an audio transcription."}

    transcription_data = cached_data_entry["data"]
    original_path = cached_data_entry.get("original_path", transcription_id) # Get original path if available

    try:
        # Pass the full transcription data (which might include text, duration etc.)
        processing_result = audio_extractor.process_hr_audio(transcription_data)

        if "error" not in processing_result:
            # Create a decision proposal based on the processing
            # The audio_extractor method should handle calling create_ai_decision_impl
            decision_msg = audio_extractor.create_audio_decision(transcription_data, processing_result, transcription_id)
            log_automated_action(f"HR audio processed: {original_path}", "Completed", details=f"Transcription ID: {transcription_id}. Decision proposed: {decision_msg}")
            # Return summary and confirmation of decision proposal
            return {
                "status": "success",
                "summary": processing_result, # Contains topics, actions etc.
                "decision_proposal_result": decision_msg # Message confirming proposal
                }
        else:
            log_automated_action(f"HR audio processing failed: {original_path}", "Error", details=processing_result.get("error"))
            return {"status": "error", "message": processing_result.get("error")}
    except Exception as e:
        logger.error(f"Error in process_hr_audio tool for {transcription_id}: {e}", exc_info=True)
        log_automated_action(f"HR audio processing failed: {original_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during HR audio processing: {e}"}


process_hr_audio_tool = StructuredTool.from_function(
    func=process_hr_audio_tool_impl,
    name="process_hr_audio",
    description="Analyzes a previously transcribed audio (using its transcription ID) for HR-related content (topics, actions, dates, sentiment) and automatically proposes an AI decision for HR review.",
    args_schema=ProcessHRAudioInput
)

def extract_interview_evaluation_tool_impl(transcription_id: str) -> Dict[str, Any]:
    """Extracts evaluation metrics from a previously transcribed job interview recording."""
    logger.info(f"Tool Used: extract_interview_evaluation(transcription_id='{transcription_id}')")
    cached_data_entry = fake_db["extracted_data_cache"].get(transcription_id)

    if not cached_data_entry:
        return {"status": "error", "message": f"Transcription ID '{transcription_id}' not found in cache. Please transcribe the file first."}
    if cached_data_entry["type"] != "audio_transcription":
        return {"status": "error", "message": f"Cache ID '{transcription_id}' does not correspond to an audio transcription."}

    transcription_data = cached_data_entry["data"]
    original_path = cached_data_entry.get("original_path", transcription_id)
    transcription_text = transcription_data.get("text", "")

    if not transcription_text:
        return {"status": "error", "message": f"No transcription text found for ID '{transcription_id}'."}

    try:
        evaluation_result = audio_extractor.extract_interview_evaluation(transcription_text)
        # Optionally create a decision here too? For now, just return results.
        # Could cache the evaluation result:
        # fake_db["extracted_data_cache"][transcription_id + "_eval"] = {"type": "interview_eval", "data": evaluation_result, ...}
        log_automated_action(f"Interview evaluation extracted: {original_path}", "Completed", details=f"Transcription ID: {transcription_id}")
        return {"status": "success", "evaluation": evaluation_result}
    except Exception as e:
        logger.error(f"Error in extract_interview_evaluation tool for {transcription_id}: {e}", exc_info=True)
        log_automated_action(f"Interview evaluation failed: {original_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during interview evaluation: {e}"}


extract_interview_evaluation_tool = StructuredTool.from_function(
    func=extract_interview_evaluation_tool_impl,
    name="extract_interview_evaluation",
    description="Analyzes a previously transcribed job interview audio (using its transcription ID) to extract evaluation metrics like skills mentioned, sentiment, key topics, and potential red flags.",
    args_schema=ExtractInterviewEvaluationInput
)


# Image Extractor Tools
def extract_text_from_image_tool_impl(image_path: str, is_handwritten: bool = False) -> Dict[str, Any]:
    """Extracts text from an image file using OCR. Handles printed or handwritten text."""
    logger.info(f"Tool Used: extract_text_from_image(image_path='{image_path}', is_handwritten={is_handwritten})")
    if not image_extractor.ocr_available:
        # Check OCR availability at the start
        logger.warning("OCR capability is not available. Image text extraction will fail.")
        return {"status": "error", "message": "OCR capability is not available on the server. Cannot process images."}
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"File not found at path: {image_path}"}

    try:
        result = image_extractor.extract_text_from_image(image_path=image_path, is_handwritten=is_handwritten)
        if "text" in result and not result.get("error"):
            cache_id = os.path.basename(image_path) # Use filename as cache key
            fake_db["extracted_data_cache"][cache_id] = {
                "type": "image_ocr",
                "data": result, # Store OCR result (text, confidence)
                "timestamp": time.time(),
                "original_path": image_path
                }
            log_automated_action(f"Image OCR completed: {image_path}", "Completed", details=f"Cache ID: {cache_id}. Handwritten: {is_handwritten}")
            return {"status": "success", "ocr_id": cache_id, "preview": result["text"][:150] + "...", "confidence": result.get("confidence")}
        else:
            log_automated_action(f"Image OCR failed: {image_path}", "Error", details=result.get("error"))
            return {"status": "error", "message": result.get("error", "Unknown OCR error")}
    except Exception as e:
        logger.error(f"Error in extract_text_from_image tool for {image_path}: {e}", exc_info=True)
        log_automated_action(f"Image OCR failed: {image_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during image text extraction: {e}"}


extract_text_from_image_tool = StructuredTool.from_function(
    func=extract_text_from_image_tool_impl,
    name="extract_text_from_image",
    description="Extracts text from an image file path using OCR. Verifies file existence. Specify if handwritten. Returns status and OCR ID (usually filename).",
    args_schema=ExtractTextFromImageInput
)

def process_form_image_tool_impl(image_path: str) -> Dict[str, Any]:
    """Processes an image of a form, extracts text via OCR, and attempts to structure the data."""
    logger.info(f"Tool Used: process_form_image(image_path='{image_path}')")
    if not image_extractor.ocr_available:
        logger.warning("OCR capability is not available. Form processing will fail.")
        return {"status": "error", "message": "OCR capability is not available on the server. Cannot process form images."}
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"File not found at path: {image_path}"}

    try:
        # This method should perform OCR and structuring
        result = image_extractor.process_form_image(image_path=image_path)

        if "form_data" in result and not result.get("error"):
            cache_id = os.path.basename(image_path) + "_form"
            fake_db["extracted_data_cache"][cache_id] = {
                "type": "form_data",
                "data": result, # Store structured data, raw text, confidence etc.
                "timestamp": time.time(),
                "original_path": image_path
                }
            # Propose a decision - handled within image_extractor.process_form_image now assumed
            decision_msg = image_extractor.create_document_decision(result, "form_processing", image_path)
            log_automated_action(f"Form image processed: {image_path}", "Completed", details=f"Cache ID: {cache_id}. Decision proposed: {decision_msg}")
            return {"status": "success", "form_data": result["form_data"], "ocr_confidence": result.get("confidence"), "decision_proposal_result": decision_msg}
        else:
            log_automated_action(f"Form image processing failed: {image_path}", "Error", details=result.get("error"))
            return {"status": "error", "message": result.get("error", "Error processing form")}
    except Exception as e:
        logger.error(f"Error in process_form_image tool for {image_path}: {e}", exc_info=True)
        log_automated_action(f"Form image processing failed: {image_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during form processing: {e}"}


process_form_image_tool = StructuredTool.from_function(
    func=process_form_image_tool_impl,
    name="process_form_image",
    description="Processes an image file path suspected to be a form. Verifies file existence. Extracts fields using OCR, structures the data, and automatically proposes an AI decision for HR review.",
    args_schema=ProcessFormImageInput
)

def extract_id_document_tool_impl(image_path: str) -> Dict[str, Any]:
    """Extracts information from an image of an ID card or document using OCR."""
    logger.info(f"Tool Used: extract_id_document(image_path='{image_path}')")
    if not image_extractor.ocr_available:
        logger.warning("OCR capability is not available. ID extraction will fail.")
        return {"status": "error", "message": "OCR capability is not available on the server. Cannot process ID documents."}
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"File not found at path: {image_path}"}

    try:
        result = image_extractor.extract_id_document(image_path=image_path)
        if "id_data" in result and not result.get("error"):
            cache_id = os.path.basename(image_path) + "_id"
            fake_db["extracted_data_cache"][cache_id] = {
                "type": "id_data",
                "data": result,
                "timestamp": time.time(),
                "original_path": image_path
                }
            # Propose a decision - handled within image_extractor assumed
            decision_msg = image_extractor.create_document_decision(result, "id_verification", image_path)
            log_automated_action(f"ID document processed: {image_path}", "Completed", details=f"Cache ID: {cache_id}. Decision proposed: {decision_msg}")
            return {"status": "success", "id_data": result["id_data"], "ocr_confidence": result.get("confidence"), "decision_proposal_result": decision_msg}
        else:
            log_automated_action(f"ID document extraction failed: {image_path}", "Error", details=result.get("error"))
            return {"status": "error", "message": result.get("error", "Error extracting ID data")}
    except Exception as e:
        logger.error(f"Error in extract_id_document tool for {image_path}: {e}", exc_info=True)
        log_automated_action(f"ID document extraction failed: {image_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during ID document extraction: {e}"}


extract_id_document_tool = StructuredTool.from_function(
    func=extract_id_document_tool_impl,
    name="extract_id_document",
    description="Processes an image file path suspected to be an ID document. Verifies file existence. Extracts fields using OCR, structures the data, and automatically proposes an AI decision for HR review.",
    args_schema=ExtractIdDocumentInput
)

def analyze_handwritten_note_tool_impl(image_path: str) -> Dict[str, Any]:
    """Analyzes an image of a handwritten note using OCR and basic sentiment analysis."""
    logger.info(f"Tool Used: analyze_handwritten_note(image_path='{image_path}')")
    if not image_extractor.ocr_available:
        logger.warning("OCR capability is not available. Handwritten note analysis will fail.")
        return {"status": "error", "message": "OCR capability is not available on the server. Cannot analyze handwritten notes."}
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"File not found at path: {image_path}"}

    try:
        result = image_extractor.analyze_handwritten_note(image_path=image_path)
        if "text" in result and not result.get("error"):
            cache_id = os.path.basename(image_path) + "_note"
            fake_db["extracted_data_cache"][cache_id] = {
                "type": "note_analysis",
                "data": result,
                "timestamp": time.time(),
                "original_path": image_path
                }
            # Propose a decision - handled within image_extractor assumed
            decision_msg = image_extractor.create_document_decision(result, "handwritten_note_analysis", image_path)
            log_automated_action(f"Handwritten note analyzed: {image_path}", "Completed", details=f"Cache ID: {cache_id}. Decision proposed: {decision_msg}")
            return {
                "status": "success",
                "analysis": {
                    "text": result["text"],
                    "sentiment": result.get("sentiment"),
                    "key_info": result.get("key_information") # Assuming extractor provides this
                    },
                "ocr_confidence": result.get("confidence"),
                "decision_proposal_result": decision_msg
                }
        else:
            log_automated_action(f"Handwritten note analysis failed: {image_path}", "Error", details=result.get("error"))
            return {"status": "error", "message": result.get("error", "Error analyzing note")}
    except Exception as e:
        logger.error(f"Error in analyze_handwritten_note tool for {image_path}: {e}", exc_info=True)
        log_automated_action(f"Handwritten note analysis failed: {image_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during handwritten note analysis: {e}"}


analyze_handwritten_note_tool = StructuredTool.from_function(
    func=analyze_handwritten_note_tool_impl,
    name="analyze_handwritten_note",
    description="Analyzes an image file path of a handwritten note. Verifies file existence. Extracts text using OCR, performs basic sentiment/keyword analysis, and automatically proposes an AI decision for HR review.",
    args_schema=AnalyzeHandwrittenNoteInput
)


# PDF Extractor Tools
def extract_text_from_pdf_tool_impl(file_path: str) -> Dict[str, Any]:
    """Extracts raw text content from a PDF file."""
    logger.info(f"Tool Used: extract_text_from_pdf(file_path='{file_path}')")
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found at path: {file_path}"}

    try:
        text = pdf_extractor.extract_text_from_pdf(file_path)
        if not text.startswith("Error:"):
            cache_id = os.path.basename(file_path) # Use filename as cache key
            fake_db["extracted_data_cache"][cache_id] = {
                "type": "pdf_text",
                "data": {"text": text}, # Store text in a dict for consistency
                "timestamp": time.time(),
                "original_path": file_path
                }
            log_automated_action(f"PDF text extracted: {file_path}", "Completed", details=f"Cache ID: {cache_id}. Length: {len(text)}")
            return {"status": "success", "pdf_text_id": cache_id, "preview": text[:200] + "...", "char_count": len(text)}
        else:
            # Error message returned by the extractor function
            log_automated_action(f"PDF text extraction failed: {file_path}", "Error", details=text)
            return {"status": "error", "message": text}
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf tool for {file_path}: {e}", exc_info=True)
        log_automated_action(f"PDF text extraction failed: {file_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during PDF text extraction: {e}"}


extract_text_from_pdf_tool = StructuredTool.from_function(
    func=extract_text_from_pdf_tool_impl,
    name="extract_text_from_pdf",
    description="Extracts raw text from a PDF file path. Verifies file existence. Returns status and a PDF text ID (usually filename) for potential further processing.",
    args_schema=ExtractTextFromPdfInput
)

def parse_cv_tool_impl(file_path: str) -> Dict[str, Any]:
    """Parses a CV/Resume PDF file, extracting structured information like personal info, education, experience, skills."""
    logger.info(f"Tool Used: parse_cv(file_path='{file_path}')")
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found at path: {file_path}"}

    try:
        # This method performs text extraction and parsing
        cv_data = pdf_extractor.parse_cv(file_path=file_path)

        if "error" not in cv_data:
            cache_id = os.path.basename(file_path) + "_cv"
            fake_db["extracted_data_cache"][cache_id] = {
                "type": "cv_data",
                "data": cv_data, # Contains structured info and potentially raw text
                "timestamp": time.time(),
                "original_path": file_path
                }
            log_automated_action(f"CV parsed: {file_path}", "Completed", details=f"Cache ID: {cache_id}. Name: {cv_data.get('personal_info', {}).get('name', 'N/A')}")
            # Exclude raw text from direct return to agent unless needed
            cv_summary = {k: v for k, v in cv_data.items() if k != 'raw_text'}
            return {"status": "success", "cv_id": cache_id, "cv_summary": cv_summary}
        else:
            log_automated_action(f"CV parsing failed: {file_path}", "Error", details=cv_data.get("error"))
            return {"status": "error", "message": cv_data.get("error", "Error parsing CV")}
    except Exception as e:
        logger.error(f"Error in parse_cv tool for {file_path}: {e}", exc_info=True)
        log_automated_action(f"CV parsing failed: {file_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during CV parsing: {e}"}


parse_cv_tool = StructuredTool.from_function(
    func=parse_cv_tool_impl,
    name="parse_cv",
    description="Parses a CV/Resume PDF file path to extract structured data (personal info, education, experience, skills). Verifies file existence. Returns status and CV ID (filename + '_cv').",
    args_schema=ParseCvInput
)

def analyze_cv_for_role_tool_impl(cv_file_path: str, role_title: str, required_skills: List[str], preferred_skills: Optional[List[str]] = None, min_years_experience: Optional[int] = 0, education_requirements: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Analyzes a parsed CV against specific job role requirements and proposes an evaluation decision."""
    logger.info(f"Tool Used: analyze_cv_for_role(cv_file_path='{cv_file_path}', role_title='{role_title}')")

    # 1. Check if CV exists
    if not os.path.exists(cv_file_path):
        return {"status": "error", "message": f"CV file not found at path: {cv_file_path}"}

    # 2. Parse the CV first (or retrieve from cache)
    cv_cache_id = os.path.basename(cv_file_path) + "_cv"
    cached_data_entry = fake_db["extracted_data_cache"].get(cv_cache_id)
    cv_data = None

    if cached_data_entry and cached_data_entry["type"] == "cv_data":
        # Check timestamp? Maybe re-parse if too old? For now, use cache if available.
        cv_data = cached_data_entry["data"]
        logger.info(f"Using cached CV data for {cv_file_path} (ID: {cv_cache_id})")
    else:
        logger.info(f"Parsing CV {cv_file_path} first...")
        parse_result = parse_cv_tool_impl(file_path=cv_file_path)
        if parse_result["status"] == "success":
            # Retrieve the just-cached data
            cv_cache_id_from_parse = parse_result["cv_id"]
            if fake_db["extracted_data_cache"].get(cv_cache_id_from_parse):
                cv_data = fake_db["extracted_data_cache"][cv_cache_id_from_parse]["data"]
            else:
                # Should not happen if parse_cv_tool works correctly
                logger.error(f"Cache inconsistency after parsing CV {cv_file_path}")
                return {"status": "error", "message": "Internal error: Failed to retrieve parsed CV data after successful parsing."}
        else:
            return {"status": "error", "message": f"Could not parse CV '{cv_file_path}' needed for analysis: {parse_result['message']}"}

    if not cv_data:
        # Should be caught above, but double-check
        return {"status": "error", "message": f"Failed to obtain parsed CV data for '{cv_file_path}'."}

    # 3. Define Role Requirements Dictionary
    role_reqs = {
        "title": role_title,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills or [],
        "min_years_experience": min_years_experience or 0,
        "education_requirements": education_requirements or {}
    }

    # 4. Perform Analysis (within pdf_extractor)
    try:
        analysis_result = pdf_extractor.analyze_cv_for_role(cv_data, role_reqs)

        # 5. Create Automated Decision (within pdf_extractor)
        decision_msg = pdf_extractor.create_automated_cv_decision(cv_data, role_reqs, analysis_result, cv_file_path)
        log_automated_action(f"CV analyzed for role '{role_title}': {cv_file_path}", "Completed", details=f"Decision proposed: {decision_msg}. Match score: {analysis_result.get('overall_match_score')}")

        return {
            "status": "success",
            "analysis_summary": analysis_result, # Contains match scores, comments etc.
            "decision_proposal_result": decision_msg
        }
    except Exception as e:
        logger.error(f"Error in analyze_cv_for_role tool for {cv_file_path}: {e}", exc_info=True)
        log_automated_action(f"CV analysis for role '{role_title}' failed: {cv_file_path}", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred during CV analysis: {e}"}


analyze_cv_for_role_tool = StructuredTool.from_function(
    func=analyze_cv_for_role_tool_impl,
    name="analyze_cv_for_role",
    description="Analyzes a CV PDF (specified by file path) against job requirements (title, skills, experience, education). It will parse the CV if not already cached. Automatically proposes an evaluation decision for HR review.",
    args_schema=AnalyzeCvForRoleInput
)


# HR Report Generator Tools
def generate_detailed_hr_report_tool_impl(period: str = "last 7 days") -> Dict[str, Any]:
    """Generates a comprehensive HR report with stats, trends, and potentially visualizations."""
    logger.info(f"Tool Used: generate_detailed_hr_report(period='{period}')")
    try:
        report = hr_report_generator.generate_detailed_hr_report(period=period)
        # Remove potentially large/complex data before returning to LLM
        report.pop("visualizations", None) # Remove viz data
        # Maybe shorten long lists or text fields if needed
        # e.g., report["recommendations"] = report.get("recommendations", [])[:3]
        log_automated_action(f"Detailed HR report generated", "Completed", details=f"Period: {period}")
        return {"status": "success", "report_summary": report}
    except Exception as e:
        logger.error(f"Error generating detailed HR report for period {period}: {e}", exc_info=True)
        log_automated_action(f"Detailed HR report generation failed", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred while generating the detailed HR report: {e}"}


generate_detailed_hr_report_tool = StructuredTool.from_function(
    func=generate_detailed_hr_report_tool_impl,
    name="generate_detailed_hr_report",
    description="Generates a detailed HR report for a specified period (e.g., 'last 7 days', 'last month', 'last quarter', 'year to date'), including key metrics, stats, trends, departmental breakdown, and recommendations.",
    args_schema=GenerateDetailedHrReportInput
)

def get_pending_decisions_report_tool_impl() -> Dict[str, Any]:
    """Generates a report detailing all AI decisions pending HR validation."""
    logger.info("Tool Used: get_pending_decisions_report()")
    try:
        # This report generator method should format the data appropriately
        report = hr_report_generator.get_pending_decisions_report()
        log_automated_action(f"Pending decisions report generated", "Completed", details=f"Count: {report.get('summary', {}).get('total_pending', 'N/A')}")
        # Return the structured report directly
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Error generating pending decisions report: {e}", exc_info=True)
        log_automated_action(f"Pending decisions report generation failed", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred while generating the pending decisions report: {e}"}

get_pending_decisions_report_tool = StructuredTool.from_function(
    func=get_pending_decisions_report_tool_impl,
    name="get_pending_decisions_report",
    description="Generates a report summarizing AI decisions currently pending HR validation, including counts by type, age, priority, and a list of decision IDs."
)

def get_attrition_risk_report_tool_impl(department: Optional[str] = None) -> Dict[str, Any]:
    """Generates a report on employee attrition risk, optionally filtered by department."""
    logger.info(f"Tool Used: get_attrition_risk_report(department='{department}')")
    try:
        report = hr_report_generator.get_attrition_risk_report(department=department)
        log_automated_action(f"Attrition risk report generated", "Completed", details=f"Department: {department or 'All'}")
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Error generating attrition risk report for department {department}: {e}", exc_info=True)
        log_automated_action(f"Attrition risk report generation failed", "Error", details=str(e))
        return {"status": "error", "message": f"An unexpected error occurred while generating the attrition risk report: {e}"}


get_attrition_risk_report_tool = StructuredTool.from_function(
    func=get_attrition_risk_report_tool_impl,
    name="get_attrition_risk_report",
    description="Generates a report detailing employee attrition risk analysis, including overall risk levels, departmental breakdown, key contributing factors, trends, and recommendations. Can be filtered by a specific department name.",
    args_schema=GetAttritionRiskReportInput
)


# --- Helper function for logging automated actions ---
def log_automated_action(description: str, status: str, details: Optional[str] = None):
    """Logs an action performed automatically by the system or tools."""
    log_message = f"Automated Action: {description} - Status: {status}" + (f" - Details: {details}" if details else "")
    logger.info(log_message)
    log_entry = {
        "description": description,
        "status": status,
        "timestamp": time.time(),
        "timestamp_readable": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    }
    if details:
        # Truncate long details to avoid bloating the log
        log_entry["details"] = details[:500] + ('...' if len(details) > 500 else '')
    fake_db["automated_actions"].insert(0, log_entry) # Insert at beginning for recent first
    # Keep only the last N automated actions?
    fake_db["automated_actions"] = fake_db["automated_actions"][:100]

# --- Assemble ALL Tools ---
# Filter out the deprecated tool if desired
tools = [
    # Core tools
    get_user_role,
    predict_employee_attrition,
    list_pending_ai_decisions,
    get_decision_details,
    # create_ai_decision, # Mostly internal, maybe hide from agent description? Or keep for flexibility.
    approve_decision,
    reject_decision,
    get_employee_tasks,
    get_employee_notifications,
    submit_employee_request,
    # Audio tools
    transcribe_audio_file_tool,
    process_hr_audio_tool,
    extract_interview_evaluation_tool,
    # Image tools
    extract_text_from_image_tool,
    process_form_image_tool,
    extract_id_document_tool,
    analyze_handwritten_note_tool,
    # PDF tools
    extract_text_from_pdf_tool,
    parse_cv_tool,
    analyze_cv_for_role_tool,
    # Report tools
    generate_detailed_hr_report_tool,
    get_pending_decisions_report_tool,
    get_attrition_risk_report_tool,
]

# Check if OCR is available and potentially remove image tools if not
if not image_extractor.ocr_available:
    logger.warning("OCR capability not available. Removing image processing tools from the agent.")
    tools = [t for t in tools if t.name not in [
        "extract_text_from_image",
        "process_form_image",
        "extract_id_document",
        "analyze_handwritten_note",
    ]]


logger.info(f"Total tools available to agent: {len(tools)}")
# Log available tool names
# logger.debug(f"Available tools: {[t.name for t in tools]}")


# --- Initialize LLM and LangChain Agent ---
logger.info("Initializing LLM and LangChain Agent...")
try:
    llm = ChatGroq(
        temperature=0.1, # Low temp for more deterministic behavior
        model_name="llama3-8b-8192", # Or llama3-70b-8192 for potentially better reasoning
        api_key=GROQ_API_KEY
    )
except Exception as e:
    logger.error(f"Failed to initialize ChatGroq LLM: {e}", exc_info=True)
    exit("LLM initialization failed. Exiting.")

# Define Agent Prompt Template (Updated Description)
# Note: We pass user_role and user_email directly to invoke,
# but including them in the prompt context helps the LLM remember.
prompt_template = """
You are 'HR Assist AI', a specialized AI assistant integrated into our company's internal HR application.
Your current user is logged in with Role: {user_role} and Email: {user_email}.
Your primary goal is to understand the user's request, determine the appropriate action based on their role and available tools, execute the action(s), and provide a clear, concise, and helpful response.

**ROLE-BASED ACCESS CONTROL (IMPORTANT):**
*   **HR Users:** Have full access to all tools, including generating reports, managing AI decisions (list, get details, approve, reject), processing all file types, analyzing CVs, and viewing employee data (tasks, notifications, attrition risk).
*   **Employee Users:** Have limited access. They can submit requests (`submit_employee_request`), view their own tasks (`get_employee_tasks`) and notifications (`get_employee_notifications`), and ask general questions (which might not require a tool). They CANNOT approve/reject decisions, generate company-wide reports, or access other employees' data.
*   **Your Actions:** Before using a tool restricted to HR (like `approve_decision`, `reject_decision`, `generate_detailed_hr_report`, `get_attrition_risk_report`, `analyze_cv_for_role`), double-check that the current user's role is 'HR'. If an Employee asks for something they aren't allowed, politely inform them it's restricted or guide them to an allowed action (like submitting a query request).

**TOOL USAGE GUIDELINES:**

1.  **File Processing Workflow:**
    *   **Requirement:** User requests involving files (audio, image, PDF) usually require a **file path**. If the user mentions a file but doesn't provide the full path, ASK FOR IT. Assume paths provided might be relative (e.g., `dummy_files/sample_cv.pdf`) or absolute.
    *   **Verification:** Most file tools verify if the path exists before processing.
    *   **Step 1 (Extraction/Transcription):** Use the primary tool for the file type (e.g., `transcribe_audio_file`, `extract_text_from_pdf`, `extract_text_from_image`, `parse_cv`). These tools often return an ID (like `transcription_id`, `pdf_text_id`, `ocr_id`, `cv_id`) which is usually based on the filename.
    *   **Step 2 (Analysis/Processing):** If further analysis is needed (e.g., analyze HR content in audio, evaluate an interview, process a form/ID/note image, analyze a CV for a role), use the corresponding processing tool (e.g., `process_hr_audio`, `extract_interview_evaluation`, `process_form_image`, `analyze_cv_for_role`) and provide the ID obtained in Step 1.
    *   **Example:** To analyze an HR meeting audio at `path/to/meeting.mp3`:
        1. Use `transcribe_audio_file` with `file_path='path/to/meeting.mp3'`. Get back `transcription_id='meeting.mp3'`.
        2. Use `process_hr_audio` with `transcription_id='meeting.mp3'`.

2.  **AI Decision Proposals:**
    *   Several tools automatically analyze content and propose an AI-generated decision for HR review (e.g., `process_hr_audio`, `process_form_image`, `extract_id_document`, `analyze_handwritten_note`, `analyze_cv_for_role`). These tools will return a `decision_proposal_result` message confirming this.
    *   When such a tool succeeds, inform the user that the analysis is complete AND that a decision proposal has been created and is awaiting HR validation.
    *   HR users can then use `list_pending_ai_decisions` to see pending items, `get_decision_details` to view a specific one, and `approve_decision` or `reject_decision` (providing the `decision_id`) to act on it.

3.  **Clarity and Conciseness:** Use the tools sequentially if needed. Synthesize the results into a single, clear response for the user. Don't just dump raw tool output unless necessary (like extracted text preview). Confirm actions taken (e.g., "Request submitted", "Decision approved", "Report generated"). If an error occurs, state it clearly but politely.

You have access to the following tools:
{tools}

Use the ReAct format for your reasoning and actions:

Question: The user's input query or request.
Thought:
1.  Identify the user's role ({user_role}) and email ({user_email}).
2.  Analyze the user's intent. What do they want to achieve?
3.  Is this action allowed for their role? If not, plan a polite refusal or alternative.
4.  Does the request involve a file? If yes, do I have the file path? If not, I must ask for it.
5.  Which tool(s) are needed? Is there a sequence (e.g., transcribe then process)?
6.  What parameters does each tool need? (e.g., file_path, transcription_id, decision_id, reason, user_email).
7.  Formulate the plan: Call tool A, get result (maybe an ID), then call tool B using the ID.
8.  If an AI decision will be proposed automatically by a tool, note this to inform the user.
9.  Plan the final response structure based on expected tool outputs.
Action:
```json
{{
    "action": "tool_name",
    "action_input": {{ "parameter1": "value1", "parameter2": "value2" }}
}} Observation: The result returned by the tool (JSON dictionary).
...(Repeat Thought/Action/Observation as needed for multi-step tasks)...
Thought: I have now gathered all the necessary information or completed the required actions. The user's role is {user_role}. I will formulate a final response summarizing the outcome, confirming actions, presenting key results concisely, and mentioning any proposed AI decisions or errors encountered.
Final Answer: The final, user-facing response. Be professional, direct, and helpful.
Begin!
User Context: Role={user_role}, Email={user_email}
Question: {input}
{agent_scratchpad}
"""
try:
    prompt = PromptTemplate.from_template(prompt_template)
    # Create the Agent (React Definition)
    agent = create_react_agent(llm, tools, prompt)

    # Create the Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Set to False for cleaner production output
        handle_parsing_errors="I encountered an issue understanding the tool's response or formatting my own. Could you please rephrase, or perhaps the task is too complex right now?",
        max_iterations=10, # Prevent runaway loops
        # memory=... # Add memory here if needed
    )
    logger.info("LangChain Agent Executor created successfully.")
except Exception as e:
    logger.error(f"Failed to create LangChain agent/executor: {e}", exc_info=True)
    exit("Agent creation failed. Exiting.")
# --- Main function to run the agent ---
def run_agent(user_email, user_role, user_input):
    """
    Run the agent with a specific user's context and input.
    """
    logger.info(f"Running agent for user '{user_email}' (Role: {user_role}) with input: '{user_input}'")
    start_time = time.time()
    try:
        # Include user context directly in the input dictionary for the agent
        # The agent prompt also gets these, but passing them here makes them directly accessible
        # in the agent's execution steps if needed by intermediate logic.
        response = agent_executor.invoke({
        "input": user_input,
        "user_email": user_email, # Pass context
        "user_role": user_role, # Pass context
        "agent_scratchpad": "" # Initialize scratchpad (though often handled internally)
        # "chat_history": [], # Add memory state here if using memory
        })
        output = response.get("output", "Agent did not return a standard output.")
    except Exception as e:
        logger.error(f"Error during agent execution for input '{user_input}': {str(e)}", exc_info=True) # Log traceback
        output = f"Sorry, I encountered an unexpected error while processing your request. The technical details logged are: {str(e)}"
    finally:
        end_time = time.time()
        logger.info(f"Agent execution finished in {end_time - start_time:.2f} seconds for user '{user_email}'.")

    return output

# Example usage / Interactive Mode 
if __name == "main": # type: ignore
    print("\n--- HR Assist AI ---")
    print("Initializing...")
# Helper function to add sample data and create dummy files
def setup_test_environment():
    print("\nSetting up test environment...")
    # Add a sample decision if none exist
    pending_decisions = [d for d in fake_db["decisions"].values() if d["status"] == "Pending Validation"]
    if not pending_decisions:
        decision_id_1 = f"dec_{uuid4()}"
        fake_db["decisions"][decision_id_1] = {
            "id": decision_id_1, "action_name": "evaluate_leave_request",
            "parameters": {"employee_email": "emp_user@example.com", "details": "Vacation Aug 20-25", "task_id": "task_sample_leave"},
            "confidence_score": 0.70, "justification": "Standard request submitted via portal, needs availability check.",
            "status": "Pending Validation", "timestamp": time.time() - 3600 # 1 hour ago
        }
        # Add corresponding task
        if "emp_user@example.com" not in fake_db["tasks"]: fake_db["tasks"]["emp_user@example.com"] = []
        # Avoid adding duplicate tasks if run multiple times
        if not any(t["id"] == "task_sample_leave" for t in fake_db["tasks"]["emp_user@example.com"]):
            fake_db["tasks"]["emp_user@example.com"].append({
                "id": "task_sample_leave", "description": "Leave Application: Vacation Aug 20-25", "status": "Pending HR Validation", "timestamp": time.time() - 3600
            })
            print(f"-> Added sample pending decision: {decision_id_1} and task: task_sample_leave")
    else:
        print(f"-> Found existing pending decisions ({len(pending_decisions)}). Skipping sample decision creation.")


    # Create dummy files for testing (if they don't exist)
    dummy_files_dir = "dummy_files"
    os.makedirs(dummy_files_dir, exist_ok=True)
    print(f"-> Ensured dummy directory exists: '{os.path.abspath(dummy_files_dir)}'")

    dummy_paths = {
        "pdf": os.path.abspath(os.path.join(dummy_files_dir, "sample_cv.pdf")),
        "image_form": os.path.abspath(os.path.join(dummy_files_dir, "sample_form.png")),
        "image_note": os.path.abspath(os.path.join(dummy_files_dir, "sample_note.png")),
        "image_id": os.path.abspath(os.path.join(dummy_files_dir, "sample_id.png")), # Add sample ID image
        "audio": os.path.abspath(os.path.join(dummy_files_dir, "sample_audio.wav")),
    }

    created_files = []

    # Create dummy PDF (requires reportlab)
    try:
        from reportlab.pdfgen import canvas # type: ignore
        from reportlab.lib.pagesizes import letter # type: ignore
        if not os.path.exists(dummy_paths["pdf"]):
            c = canvas.Canvas(dummy_paths["pdf"], pagesize=letter)
            c.drawString(72, 800, "Sample Curriculum Vitae")
            c.drawString(72, 780, "Name: Alex Chen")
            c.drawString(72, 765, "Email: alex.chen@email.dev")
            c.drawString(72, 750, "Phone: 555-123-4567")
            c.drawString(72, 720, "Summary:")
            c.drawString(90, 705, "Experienced software developer with 5 years in Python and cloud tech.")
            c.drawString(72, 675, "Experience:")
            c.drawString(90, 660, "Software Engineer | Tech Corp | 2019 - Present")
            c.drawString(90, 645, "- Developed backend services using Python (Flask, Django).")
            c.drawString(90, 630, "- Worked with AWS (EC2, S3, Lambda).")
            c.drawString(72, 600, "Education:")
            c.drawString(90, 585, "B.Sc. Computer Science | University of Tech | 2019")
            c.drawString(72, 555, "Skills:")
            c.drawString(90, 540, "Python, Java, SQL, AWS, Docker, Git, Agile")
            c.save()
            created_files.append(dummy_paths["pdf"])
    except ImportError:
        logger.warning(f"reportlab not installed. Skipping dummy PDF creation. Provide your own PDF for testing at {dummy_paths['pdf']}")
    except Exception as e:
        logger.error(f"Error creating dummy PDF: {e}", exc_info=True)

    # Create dummy Images (requires Pillow)
    try:
        from PIL import Image, ImageDraw, ImageFont
        # Try to find a common font
        common_fonts = ["arial.ttf", "DejaVuSans.ttf", "Verdana.ttf"] # Add more if needed
        font_path = None
        for f in common_fonts:
            try:
                # This might require OS-specific paths or fontconfig
                font_test = ImageFont.truetype(f, 15)
                font_path = f
                break
            except IOError:
                continue

        font = ImageFont.load_default() # Fallback
        if font_path:
            try:
                font = ImageFont.truetype(font_path, 15)
                font_large = ImageFont.truetype(font_path, 20)
            except IOError:
                logger.warning(f"Could not load font {font_path}, using default.")
                font = ImageFont.load_default()
                font_large = font
        else:
            logger.warning("Could not find common fonts (arial, DejaVuSans, Verdana). Using default PIL font.")
            font_large = font


        # Form Image
        if not os.path.exists(dummy_paths["image_form"]):
            img = Image.new('RGB', (600, 400), color = (240, 240, 240))
            d = ImageDraw.Draw(img)
            d.text((30,30), "Employee Information Form", fill=(0,0,0), font=font_large)
            d.rectangle([(30, 80), (570, 110)], outline=(0,0,0))
            d.text((40, 85), "Full Name:", fill=(50,50,50), font=font)
            d.text((150, 85), "Alice Wonderland", fill=(0,0,0), font=font) # Sample data
            d.rectangle([(30, 120), (570, 150)], outline=(0,0,0))
            d.text((40, 125), "Department:", fill=(50,50,50), font=font)
            d.text((150, 125), "Sales", fill=(0,0,0), font=font)
            d.rectangle([(30, 160), (570, 190)], outline=(0,0,0))
            d.text((40, 165), "Start Date:", fill=(50,50,50), font=font)
            d.text((150, 165), "2023-01-15", fill=(0,0,0), font=font)
            d.rectangle([(30, 200), (570, 230)], outline=(0,0,0))
            d.text((40, 205), "Emergency Contact:", fill=(50,50,50), font=font)
            d.text((200, 205), "Bob The Builder (555-987-6543)", fill=(0,0,0), font=font)
            img.save(dummy_paths["image_form"])
            created_files.append(dummy_paths["image_form"])

        # Note Image (simulating handwriting)
        if not os.path.exists(dummy_paths["image_note"]):
            img = Image.new('RGB', (400, 200), color = (255, 255, 220)) # Yellowish paper
            d = ImageDraw.Draw(img)
            # Use default font which might look less handwritten
            d.text((20,20), "Hi HR Team,", fill=(0,0,100), font=font)
            d.text((20,50), "Please review my leave request task_123.", fill=(0,0,100), font=font)
            d.text((20,80), "Need approval before end of week.", fill=(0,0,100), font=font)
            d.text((20,110), "Thanks,", fill=(0,0,100), font=font)
            d.text((20,140), "Bob Employee", fill=(0,0,100), font=font)
            img.save(dummy_paths["image_note"])
            created_files.append(dummy_paths["image_note"])

        # ID Card Image (simple simulation)
        if not os.path.exists(dummy_paths["image_id"]):
            img = Image.new('RGB', (500, 300), color = (200, 220, 255)) # Light blue background
            d = ImageDraw.Draw(img)
            d.rectangle([(10, 10), (490, 290)], outline=(0,0,0), width=2)
            d.text((150, 30), "COMPANY ID CARD", fill=(0,0,50), font=font_large)
            d.rectangle([(30, 80), (150, 200)], outline=(50,50,50)) # Placeholder for photo
            d.text((80, 130), "PHOTO", fill=(100,100,100), font=font)
            d.text((180, 90), "Name: Charlie Chaplin", fill=(0,0,0), font=font)
            d.text((180, 120), "ID: EMP789", fill=(0,0,0), font=font)
            d.text((180, 150), "Department: IT Support", fill=(0,0,0), font=font)
            d.text((180, 180), "Issue Date: 2022-05-01", fill=(0,0,0), font=font)
            d.text((180, 210), "Expiry Date: 2025-05-01", fill=(0,0,0), font=font)
            img.save(dummy_paths["image_id"])
            created_files.append(dummy_paths["image_id"])

    except ImportError:
        logger.warning(f"Pillow not installed. Skipping dummy Image creation. Provide your own images for testing at {dummy_paths['image_form']}, {dummy_paths['image_note']}, {dummy_paths['image_id']}")
    except Exception as e:
        logger.error(f"Error creating dummy Images: {e}", exc_info=True)

    # Create dummy Audio (requires pydub) - silent wav
    try:
        from pydub import AudioSegment
        if not os.path.exists(dummy_paths["audio"]):
            # Create a short silent wav file. Transcription will yield nothing.
            # For real testing, use an actual audio file with speech.
            silence = AudioSegment.silent(duration=2000) # 2 seconds silent wav
            silence.export(dummy_paths["audio"], format="wav")
            created_files.append(dummy_paths["audio"])
    except ImportError:
        logger.warning(f"pydub not installed. Skipping dummy Audio creation. Provide your own audio for testing at {dummy_paths['audio']}")
    except Exception as e:
        logger.error(f"Error creating dummy Audio: {e}", exc_info=True)


    if created_files:
        print(f"-> Created dummy files: {', '.join([os.path.basename(f) for f in created_files])}")

    print("\n--- File Paths for Testing ---")
    for key, path in dummy_paths.items():
        exists = os.path.exists(path)
        print(f"{key.replace('_', ' ').title()} Path: {path} {'(Exists)' if exists else '(MISSING!)'}")
    print("-" * 30)

    # Check OCR again here after setup
    if 'ImageExtractor' in globals() and not image_extractor.ocr_available:
        print("\nWARNING: OCR features (image text extraction) require Tesseract.")
        print("Please install Tesseract-OCR and ensure it's in your system's PATH.")
        print("Image processing tools will be unavailable or may fail.")
        print("-" * 30)

setup_test_environment() # Call helper to set up

# Set up default user credentials
hr_email = "hr_user@example.com"
hr_role = "HR"
emp_email = "emp_user@example.com"
emp_role = "Employee"

# User selection
print("\nChoose a user profile to simulate:")
print(f"1. {hr_role} ({hr_email})")
print(f"2. {emp_role} ({emp_email})")

while True:
    user_choice = input("Enter your choice (1/2): ").strip()
    if user_choice == "1":
        email = hr_email
        role = hr_role
        break
    elif user_choice == "2":
        email = emp_email
        role = emp_role
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")

print(f"\nLogged in as {role} ({email})")
print("Type 'quit' or 'exit' to end the session.")
print("Tip: Use the absolute paths printed above when referring to dummy files.")

# Display available commands based on role
print("\nExample commands you can try:")
print(f"- What is my role?")
print(f"- What tools can I use?") # General LLM question
# File paths - use abspath to ensure they work regardless of run location
pdf_path = os.path.abspath('dummy_files/sample_cv.pdf')
form_path = os.path.abspath('dummy_files/sample_form.png')
note_path = os.path.abspath('dummy_files/sample_note.png')
id_path = os.path.abspath('dummy_files/sample_id.png')
audio_path = os.path.abspath('dummy_files/sample_audio.wav') # Silent audio

if role.upper() == "HR":
    print("\n--- HR Examples ---")
    print(f"- Show pending decisions")
    pending_id = next((d["id"] for d in fake_db["decisions"].values() if d["status"] == "Pending Validation"), None)
    if pending_id:
        print(f"- Get details for decision {pending_id}")
        print(f"- Approve decision {pending_id} notes='Looks good, proceed'")
        print(f"- Reject decision {pending_id} reason='Insufficient details provided in request'")
    else:
        print("- (No pending decisions found for approve/reject examples)")
    print(f"- Generate detailed HR report for last month")
    print(f"- Show attrition risk report")
    print(f"- Show attrition risk report for Sales") # Example department
    print(f"- Check attrition risk for employee {emp_email}")
    print(f"- Extract text from PDF at '{pdf_path}'")
    print(f"- Parse CV at '{pdf_path}'")
    print(f"- Analyze CV '{pdf_path}' for the role 'Software Engineer' requiring skills ['Python', 'AWS'] with 3 years experience")
    # Example AnalyzeCvForRoleInput structure - agent needs to construct this:
    # analyze_cv_for_role(cv_file_path='...', role_title='Software Engineer', required_skills=['Python', 'AWS'], min_years_experience=3)
    if image_extractor.ocr_available:
        print(f"- Extract text from image '{form_path}'")
        print(f"- Process form image at '{form_path}'")
        print(f"- Extract ID document info from '{id_path}'")
        print(f"- Analyze handwritten note '{note_path}'")
    else:
        print("- (Image processing examples skipped - OCR not available)")
    print(f"- Transcribe audio file at '{audio_path}'") # Will transcribe silence
    # print(f"- Process HR audio recording with ID 'sample_audio.wav'") # Requires transcription first

else: # Employee
    print("\n--- Employee Examples ---")
    print(f"- Show my tasks")
    print(f"- Show my notifications")
    print(f"- Apply for leave from Sep 1 to Sep 5 using vacation days")
    print(f"- Submit IT support request: 'My monitor is flickering'")
    print(f"- Ask about the company policy on remote work") # Example of non-tool query

# Main input loop
while True:
    try:
        query = input(f"\n{role} ({email})> ").strip()
        if query.lower() in ["quit", "exit"]:
            break
        if not query:
            continue

        # --- Run the agent ---
        result = run_agent(email, role, query)
        # --- ------------- ---

        print(f"\n--- HR Assist AI ---\n{result}\n")

        # Optional: Display recent automated actions log for debugging/visibility
        # print("--- Recent Automated Actions (Internal Log) ---")
        # for log in fake_db["automated_actions"][:3]: # Show last 3 actions
        #    print(f"- {log['timestamp_readable']} [{log['status']}] {log['description']} {log.get('details', '')}")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main interactive loop: {e}")
        logger.error("Unexpected error in main loop", exc_info=True)

print("\nSession ended.")