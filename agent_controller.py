from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, Tool
from langchain_core.messages import BaseMessage, HumanMessage # Needed for GraphState history
from langchain_core.runnables import RunnableLambda # For conditional routing in LangGraph
from langchain_core.exceptions import OutputParserException # Specific exception for handling LLM output
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Callable, Any
try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # For older Python versions
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
import re
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tools import (
    extract_and_crop_receipt_images,
    extract_data_from_images,
    validate_extracted_data,
    manage_processed_receipt_files,
    request_human_review,
    human_validation_with_gradio
)
from tools import PDFServer
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# --- Project Folder Setup ---
DRIVE_PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))  # Dynamically set to current project folder
input_pdf_folder = os.path.join(DRIVE_PROJECT_FOLDER, "pdfs")
success_pdf_folder = os.path.join(DRIVE_PROJECT_FOLDER, "success_pdfs")
output_pdf_folder = os.path.join(DRIVE_PROJECT_FOLDER, "output_pdfs")
cropped_images_folder = os.path.join(DRIVE_PROJECT_FOLDER, "images")
json_output_file = os.path.join(DRIVE_PROJECT_FOLDER, "extracted_receipt_data.json")
error_pdf_folder = os.path.join(DRIVE_PROJECT_FOLDER, "error_pdfs") # For human review


# Create folders if they don't exist
os.makedirs(input_pdf_folder, exist_ok=True)
os.makedirs(success_pdf_folder, exist_ok=True)
os.makedirs(output_pdf_folder, exist_ok=True)
os.makedirs(cropped_images_folder, exist_ok=True)
os.makedirs(error_pdf_folder, exist_ok=True)
if not os.path.exists(json_output_file):
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
        print(f"Created new empty JSON file at: {json_output_file}")

# 1. Define Graph State
class GraphState(TypedDict):
    """
    Represents the state of our receipt processing workflow.
    Values are "patches" (updates to the state).
    """
    pdf_path: str
    image_paths: Annotated[list[str], "append"] # Use append to collect multiple paths
    extracted_data: NotRequired[str]
    validated_data: NotRequired[str]
    human_validation_result: NotRequired[str] # "APPROVED:[json]" or "REJECTED:[feedback]"
    error_message: NotRequired[str]
    processed_status: NotRequired[str] # "SUCCESS" or "FAILED"


    # 2. Define Nodes (as functions)

def call_extract_images(state: GraphState) -> GraphState:
    """Node to extract and crop images from the PDF."""
    print("\n--- Node: call_extract_images ---")
    pdf_path = state["pdf_path"]
    try:
        result = extract_and_crop_receipt_images.invoke({
            "pdf_path": pdf_path,
            "cropped_images_folder": cropped_images_folder
        })
        if result.startswith("ERROR:"):
            return {
                "pdf_path": pdf_path,
                "image_paths": [],
                "error_message": result,
                "processed_status": "FAILED"
            }
        # Result is JSON string like {'filename': ['path1', 'path2']}
        parsed_result = json.loads(result)
        image_paths = list(parsed_result.values())[0] # Get the list of paths
        return {
                "pdf_path": pdf_path,
                "image_paths": image_paths,
                "processed_status": "SUCCESS"
            }
    except Exception as e:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": [],
            "error_message": f"Extract images failed: {e}",
            "processed_status": "FAILED"
        }
    
def call_extract_data(state: GraphState) -> GraphState:
    """Node to extract structured data from images using Gemini."""
    print("\n--- Node: call_extract_data ---")
    image_paths = state["image_paths"]
    if not image_paths:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": [],
            "error_message": "No images available for data extraction.",
            "processed_status": "FAILED"
        }
    
    try:
        result = extract_data_from_images.invoke({
            "image_paths_json_str": json.dumps(image_paths)
        })
        if result.startswith("ERROR:"):
            return {
                "pdf_path": state.get("pdf_path", ""),
                "image_paths": state.get("image_paths", []),
                "error_message": result,
                "processed_status": "FAILED"
            }
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "extracted_data": result
        } # result is already JSON string
    except Exception as e:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "error_message": f"Extract data failed: {e}",
            "processed_status": "FAILED"
        }
    
def call_validate_data(state: GraphState) -> GraphState:
    """Node to validate the extracted data."""
    print("\n--- Node: call_validate_data ---")
    extracted_data = state.get("extracted_data")
    if not extracted_data:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "error_message": "No data available for validation.",
            "processed_status": "FAILED"
        }
    
    try:
        result = validate_extracted_data.invoke({"json_data_str": extracted_data})
        if result.startswith("ERROR:"):
            return {
                "pdf_path": state.get("pdf_path", ""),
                "image_paths": state.get("image_paths", []),
                "validated_data": result
            } # Store error message in validated_data for routing
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "validated_data": result
        } # "OK:[json_str]"
    except Exception as e:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "error_message": f"Validate data failed: {e}",
            "processed_status": "FAILED"
        }
    
def call_human_validation(state: GraphState) -> GraphState:
    """Node for human validation via Gradio."""
    print("\n--- Node: call_human_validation ---")
    pdf_path = state["pdf_path"]
    extracted_data = state.get("extracted_data")
    image_paths = state.get("image_paths")
    validated_data = state.get("validated_data")

    if not extracted_data or not image_paths:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "error_message": "Missing data or images for human validation.",
            "processed_status": "FAILED"
        }

    try:
        # print(f"[DEBUG] pdf_path: {pdf_path}")
        # print(f"[DEBUG] os.path.exists(pdf_path): {os.path.exists(pdf_path)}")
        # print(f"[DEBUG] os.listdir(os.path.dirname(pdf_path)): {os.listdir(os.path.dirname(pdf_path))}")
        pdf_server = PDFServer(directory=os.path.dirname(pdf_path))
        result = human_validation_with_gradio.invoke({
            "original_pdf_path": pdf_path,
            "extracted_data_str": extracted_data,
            "image_paths_json_str": json.dumps(image_paths),
            "validated_data_str": validated_data,
            "pdf_server":pdf_server
        })
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "validated_data": result
        } # "APPROVED:[json]" or "REJECTED:[feedback]"
    except Exception as e:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "error_message": f"Human validation failed: {e}",
            "processed_status": "FAILED"
        }
    
def call_manage_files(state: GraphState) -> GraphState:
    """Node to rename and move the processed PDF and update log."""
    print("\n--- Node: call_manage_files ---")
    pdf_path = state["pdf_path"]
    validated_data = state.get("validated_data") # Should be "OK:[json]"
    # print(validated_data)
    if not validated_data or not validated_data.startswith("APPROVED:"):
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "error_message": "Invalid validated data for file management.",
            "processed_status": "FAILED"
        }

    # Extract clean JSON from "OK:[json]" string
    extracted_json_str = validated_data[9:].strip() 

    try:
        json.loads(extracted_json_str)
        result = manage_processed_receipt_files.invoke({
            "original_pdf_path": pdf_path,
            "extracted_json_str": extracted_json_str,
            "output_pdf_folder": output_pdf_folder,
            "master_json_file_path": json_output_file,
            "success_pdf_folder":success_pdf_folder
        })
        if result.startswith("ERROR:"):
            return {
                "pdf_path": state.get("pdf_path", ""),
                "image_paths": state.get("image_paths", []),
                "error_message": result,
                "processed_status": "FAILED"
            }
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "processed_status": "SUCCESS"
        }
    except json.JSONDecodeError as e:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "error_message": f"Validated data is not valid JSON: {e}",
            "processed_status": "FAILED"
        }
    except Exception as e:
        return {
            "pdf_path": state.get("pdf_path", ""),
            "image_paths": state.get("image_paths", []),
            "error_message": f"Manage files failed: {e}",
            "processed_status": "FAILED"
        }
    
def call_request_review(state: GraphState) -> GraphState:
    """Node to move PDF to error folder for human review."""
    print("\n--- Node: call_request_review ---")
    pdf_path = state["pdf_path"]
    problem_description = state.get("error_message", "Unknown error or rejection during workflow.")
    
    try:
        result = request_human_review.invoke({
            "pdf_path": pdf_path,
            "problem_description": problem_description,
            "error_pdf_folder": error_pdf_folder
        })
        if result.startswith("ERROR:"):
            return {
                "pdf_path": pdf_path,
                "image_paths": [],
                "error_message": result,
                "processed_status": "FAILED"
            }
        return {
            "pdf_path": pdf_path,
            "image_paths": [],
            "processed_status": "FAILED",
            "error_message": result
        } # Indicate failure
    except Exception as e:
        return {
            "pdf_path": pdf_path,
            "image_paths": [],
            "error_message": f"Request review failed: {e}",
            "processed_status": "FAILED"
        }


# 3. Define Conditional Edges (Routers)

# def route_validation_result(state: GraphState) -> str:
#     """Routes based on validation result (OK or ERROR)."""
#     print(f"\n--- Router: route_validation_result. Current validated_data: {state.get('validated_data')} ---")
#     if state.get("validated_data") and state.get("validated_data","").startswith("OK:"):
#         return "manage_files"
#     elif state.get("validated_data") and state.get("validated_data", "[]").startswith("ERROR:"):
#         return "human_validation" # If validation fails, go to human validation
#     else:
#         return "request_review" # Fallback if validation result is unexpected

def route_human_validation_result(state: GraphState) -> str:
    """Routes based on human validation result (APPROVED or REJECTED)."""
    print(f"\n--- Router: route_human_validation_result. Current human_validation_result: {state.get('validated_data')} ---")
    if state.get("validated_data") and state["validated_data"].startswith("APPROVED:"):
        return "manage_files"
    else:
        # If rejected or unexpected, go to request review
        state["error_message"] = state.get("validated_data", "Human validation rejected or inconclusive.")
        return "request_review"
    
# 4. Build the Graph
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("extract_images", call_extract_images)
workflow.add_node("extract_data", call_extract_data)
workflow.add_node("validate_data", call_validate_data)
workflow.add_node("human_validation", call_human_validation)
workflow.add_node("manage_files", call_manage_files)
workflow.add_node("request_review", call_request_review)

# Set Entry Point
workflow.set_entry_point("extract_images")

# Add Edges
workflow.add_edge("extract_images", "extract_data")
workflow.add_edge("extract_data", "validate_data")
workflow.add_edge("validate_data", "human_validation")

# Conditional edge after validation
# workflow.add_conditional_edges(
#     "validate_data",
#     route_validation_result,
#     {
#         "manage_files": "manage_files",
#         "human_validation": "human_validation",
#         "request_review": "request_review",
#     },
# )

# Conditional edge after human validation
workflow.add_conditional_edges(
    "human_validation",
    route_human_validation_result,
    {
        "manage_files": "manage_files",
        "request_review": "request_review",
    },
)

# Edges to END or final error handling
workflow.add_edge("manage_files", END)
workflow.add_edge("request_review", END) # Review request is the end of the process for that PDF

# Compile the graph
app = workflow.compile()


# # --- Main Processing Loop using pooling(Updated for LangGraph) ---
# def monitor_and_process_pdfs(input_dir, processed_dir, error_dir, crop_img_dir, json_log_file):
#     """Monitors input_dir for new PDFs and processes them using the LangGraph agent."""
    
#     while True:
#         print(f"\nMonitoring '{input_dir}' for new PDFs...")
#         current_pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

#         for pdf_filename in current_pdfs:
#             pdf_path = os.path.join(input_dir, pdf_filename)

#             if not os.path.exists(pdf_path):
#                 print(f"Skipping '{pdf_filename}' (already moved from input folder).")
#                 continue

#             print(f"\n--- New PDF detected: {pdf_filename}. Starting LangGraph workflow. ---")
            
#             initial_state = GraphState(
#                 pdf_path=pdf_path,
#                 image_paths=[], # Will be appended
#             )

#             try:
#                 # Iterate through the graph steps
#                 final_state = None
#                 for s in app.stream(initial_state):
#                     print(f"Current state: {s}")
#                     final_state = s

#                 # Check the final state to determine outcome
#                 final_output = list(final_state.values())[0] if final_state else {}
#                 status = final_output.get('processed_status')
#                 if status == "SUCCESS":
#                     print(f"\n--- LangGraph workflow completed SUCCESSFULLY for {pdf_filename} ---")
#                 elif status == "FAILED":
#                     print(f"\n--- LangGraph workflow FAILED for {pdf_filename}: {final_state.get('error_message', 'No specific error message.')} ---")
#                     # If it failed and wasn't moved by request_review tool within graph,
#                     # try to move it now.
#                     if os.path.exists(pdf_path):
#                         print(f"Attempting to move {pdf_filename} to error folder due to workflow failure...")
#                         try:
#                             request_human_review.invoke({
#                                 "pdf_path": pdf_path,
#                                 "problem_description": f"LangGraph workflow failed: {final_state.get('error_message', 'No specific error message.')}",
#                                 "error_pdf_folder": error_dir
#                             })
#                             print(f"Moved '{pdf_filename}' to error folder.")
#                         except Exception as move_err:
#                             print(f"CRITICAL: Failed to move '{pdf_filename}' to error folder after graph failure: {move_err}")
#                 else:
#                      print(f"\n--- LangGraph workflow finished for {pdf_filename} with unknown status. Final state: {final_state} ---")
#                      if os.path.exists(pdf_path):
#                         print(f"Attempting to move {pdf_filename} to error folder due to unknown status...")
#                         try:
#                             request_human_review.invoke({
#                                 "pdf_path": pdf_path,
#                                 "problem_description": "LangGraph workflow finished with unknown status.",
#                                 "error_pdf_folder": error_dir
#                             })
#                             print(f"Moved '{pdf_filename}' to error folder.")
#                         except Exception as move_err:
#                             print(f"CRITICAL: Failed to move '{pdf_filename}' to error folder after graph completion: {move_err}")


#             except Exception as e:
#                 print(f"\n--- An unexpected critical error occurred outside LangGraph execution for {pdf_filename}: {e} ---")
#                 # If an error occurs even outside the graph, try to move the PDF
#                 try:
#                     if os.path.exists(pdf_path):
#                         request_human_review.invoke({
#                             "pdf_path": pdf_path,
#                             "problem_description": f"Critical system error: {e}",
#                             "error_pdf_folder": error_dir
#                         })
#                         print(f"Moved '{pdf_filename}' to error folder due to critical error.")
#                 except Exception as move_err:
#                     print(f"CRITICAL: Also failed to move original PDF to error folder after critical crash: {move_err}")

#         try:
#             time.sleep(5)
#         except KeyboardInterrupt:
#             import sys
#             print("👋  Monitoring stopped by user.")
#             sys.exit(0)
class PDFHandler(FileSystemEventHandler):
    def __init__(self, input_dir, processed_dir, error_dir, crop_img_dir, json_log_file):
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.error_dir = error_dir
        self.crop_img_dir = crop_img_dir
        self.json_log_file = json_log_file

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            pdf_path = event.src_path
            pdf_filename = os.path.basename(pdf_path)
            print(f"\n--- New PDF detected: {pdf_filename}. Starting LangGraph workflow. ---")

            initial_state = GraphState(
                pdf_path=pdf_path,
                image_paths=[],
            )

            try:
                final_state = None
                for s in app.stream(initial_state):
                    print(f"Current state: {s}")
                    final_state = s

                final_output = list(final_state.values())[0] if final_state else {}
                status = final_output.get('processed_status')

                if status == "SUCCESS":
                    print(f"\n--- LangGraph workflow completed SUCCESSFULLY for {pdf_filename} ---")

                else:
                    error_message = final_state.get('error_message', 'No specific error message.')
                    print(f"\n--- LangGraph workflow FAILED for {pdf_filename}: {error_message} ---")
                    self.move_to_error(pdf_path, pdf_filename, error_message)

            except Exception as e:
                print(f"\n--- Critical error occurred for {pdf_filename}: {e} ---")
                self.move_to_error(pdf_path, pdf_filename, f"Critical system error: {e}")

    def move_to_error(self, pdf_path, pdf_filename, reason):
        try:
            if os.path.exists(pdf_path):
                request_human_review.invoke({
                    "pdf_path": pdf_path,
                    "problem_description": reason,
                    "error_pdf_folder": self.error_dir
                })
                print(f"Moved '{pdf_filename}' to error folder.")
        except Exception as move_err:
            print(f"CRITICAL: Failed to move '{pdf_filename}' to error folder: {move_err}")

def monitor_and_process_pdfs(input_dir, processed_dir, error_dir, crop_img_dir, json_log_file):
    event_handler = PDFHandler(input_dir, processed_dir, error_dir, crop_img_dir, json_log_file)
    observer = Observer()
    observer.schedule(event_handler, path=input_dir, recursive=False)
    observer.start()
    print(f"📁 Monitoring folder: {input_dir}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        import sys
        print("🛑 Stopping monitor...")
        observer.stop()
        observer.join()
        print("✅ Folder monitor exited cleanly.")
        sys.exit(0)
    
# --- Main Execution Block ---
if __name__ == "__main__":
    # Start monitoring and processing
    print(f"Starting receipt processing LangGraph agent. Input folder: {input_pdf_folder}")
    monitor_and_process_pdfs(
            input_pdf_folder,
            output_pdf_folder,
            error_pdf_folder,
            cropped_images_folder,
            json_output_file
        )
    