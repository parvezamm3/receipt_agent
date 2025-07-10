import os
import time
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool 
from PIL import Image 


# Load environment variables (API key, project ID)
load_dotenv()


# --- Project Folder Setup ---
DRIVE_PROJECT_FOLDER = r"C:\Users\parvez427\Dev\ReciptAgent" # Use raw string or forward slashes
input_pdf_folder = os.path.join(DRIVE_PROJECT_FOLDER, "pdfs")
output_pdf_folder = os.path.join(DRIVE_PROJECT_FOLDER, "output_pdfs")
cropped_images_folder = os.path.join(DRIVE_PROJECT_FOLDER, "images")
json_output_file = os.path.join(DRIVE_PROJECT_FOLDER, "extracted_receipt_data.json")
error_pdf_folder = os.path.join(DRIVE_PROJECT_FOLDER, "error_pdfs") # For human review


# Create folders if they don't exist
os.makedirs(input_pdf_folder, exist_ok=True)
os.makedirs(output_pdf_folder, exist_ok=True)
os.makedirs(cropped_images_folder, exist_ok=True)
os.makedirs(error_pdf_folder, exist_ok=True)

# --- Import Tools ---
from tools import (
    extract_and_crop_receipt_images,
    extract_data_from_images,
    validate_extracted_data,
    manage_processed_receipt_files,
    request_human_review,
    human_validation_with_gradio
)

# --- 1. Initialize Gemini LLM ---
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",
    temperature=0.1,
    project=os.getenv("GOOGLE_CLOUD_PROJECT") # 
)

# --- TEMPORARY LLM TEST ---
print("\n--- Testing LLM directly ---")
try:
    test_message = "Hello, please tell me a very short, cheerful story about a robot."
    response = llm.invoke(test_message)
    print(f"LLM Response (Type: {type(response)}):")
    if response and hasattr(response, 'content'):
        print(response.content)
    else:
        print("WARNING: LLM response object did not have a 'content' attribute or was empty.")
except Exception as e:
    print(f"ERROR: Direct LLM call failed: {e}")
    print("Please check your GOOGLE_APPLICATION_CREDENTIALS or GEMINI_API_KEY, project ID, and network connection.")
print("--- End LLM Test ---\n")
# --- END TEMPORARY LLM TEST ---

# --- 2. Define the Agent's Tools ---
all_tools = [
    extract_and_crop_receipt_images,
    extract_data_from_images,
    validate_extracted_data,
    manage_processed_receipt_files,
    request_human_review,
    human_validation_with_gradio
]

# --- Create a dictionary for easy access to tools by name ---
tools_by_name = {tool.name: tool for tool in all_tools}


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """あなたは領収書処理ワークフローを自動化するAIアシスタントです。
        与えられたPDFから領収書データを抽出し、検証し、整理することがあなたの任務です。
        提供されたツールのみを使用してください。

        ワークフローのステップ:
        1. PDFから領収書画像を抽出し、トリミングします (extract_and_crop_receipt_images)。
        2. 抽出された画像から構造化された領収書データを抽出します (extract_data_from_images)。
        3. 抽出されたデータを検証します (validate_extracted_data)。
            - 検証が失敗した場合、人間による検証のためにデータを提示します (human_validation_with_gradio)。
            - 人間がデータを承認しない場合、そのPDFをレビューのためにエラーフォルダに移動します (request_human_review)。
        4. 処理された領収書ファイル（元のPDF）を名前変更して出力フォルダに移動し、マスターログを更新します (manage_processed_receipt_files)。
        5. 何らかのエラーが発生し、自身で解決できない場合は、PDFをエラーフォルダに移動して人間によるレビューを要求します (request_human_review)。

        常にこれらの手順に従い、各ステップの出力に基づいて行動してください。
        最終的なJSONデータが承認され、ファイルが正常に管理された場合にのみ、処理が完了したと見なしてください。
        ステップをスキップしないでください。

        ツールを使用する際は、ツール名とそのパラメータを正確に指定してください。
        以下の**厳密なReActパターン**に従って応答してください。これ以外の形式で応答しないでください:
        Thought: あなたの次の思考。何をするべきか、どのツールを使うべきか、なぜそれをするのかを日本語で考えます。
        Action: 実行するツールの名前 (利用可能なツール名の中から選択してください)。
        Action Input: ツールの入力として渡すJSON形式の引数。
        """),
        MessagesPlaceholder(variable_name="tools"),      # <--- CRITICAL CHANGE: Added this
        MessagesPlaceholder(variable_name="tool_names"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", """あなたは領収書処理ワークフローを自動化するAIアシスタントです。
#         与えられたPDFから領収書データを抽出し、検証し、整理することがあなたの任務です。
#         提供されたツールのみを使用してください。

#         ワークフローのステップ:
#         1. PDFから領収書画像を抽出し、トリミングします (extract_and_crop_receipt_images)。
#         2. 抽出された画像から構造化された領収書データを抽出します (extract_data_from_images)。
#         3. 抽出されたデータを検証します (validate_extracted_data)。
#             - 検証が失敗した場合、人間による検証のためにデータを提示します (human_validation_with_gradio)。
#             - 人間がデータを承認しない場合、そのPDFをレビューのためにエラーフォルダに移動します (request_human_review)。
#         4. 処理された領収書ファイル（元のPDF）を名前変更して出力フォルダに移動し、マスターログを更新します (manage_processed_receipt_files)。
#         5. 何らかのエラーが発生し、自身で解決できない場合は、PDFをエラーフォルダに移動して人間によるレビューを要求します (request_human_review)。

#         常にこれらの手順に従い、各ステップの出力に基づいて行動してください。
#         最終的なJSONデータが承認され、ファイルが正常に管理された場合にのみ、処理が完了したと見なしてください。
#         ステップをスキップしないでください。

#         ツールを使用する際は、ツール名とそのパラメータを正確に指定してください。
#         以下の**厳密なReActパターン**に従って応答してください。これ以外の形式で応答しないでください:
#         Thought: あなたの次の思考。何をするべきか、どのツールを使うべきか、なぜそれをするのかを日本語で考えます。
#         Action: 実行するツールの名前 (利用可能なツール名の中から選択してください)。
#         Action Input: ツールの入力として渡すJSON形式の引数。
#         """), # End of the main system message
#         MessagesPlaceholder(variable_name="tools"),      # <--- CRITICAL CHANGE: Added this
#         MessagesPlaceholder(variable_name="tool_names"), # <--- CRITICAL CHANGE: Added this
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )


# --- 4. Create the ReAct Agent ---
agent = create_react_agent(llm, all_tools, prompt)
from langchain_core.messages import SystemMessage
formatted_tools = [SystemMessage(content=str(tool)) for tool in all_tools]
# --- 5. Initialize Agent Executor with Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# --- 6. Main Processing Loop ---
def monitor_and_process_pdfs(input_dir, processed_dir, error_dir, crop_img_dir, json_log_file):
    """Monitors input_dir for new PDFs and processes them using the agent."""
    
    while True:
        print(f"\nMonitoring '{input_dir}' for new PDFs...")
        current_pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

        for pdf_filename in current_pdfs:
            pdf_path = os.path.join(input_dir, pdf_filename)

            if not os.path.exists(pdf_path):
                print(f"Skipping '{pdf_filename}' (already moved from input folder).")
                continue

            print(f"\n--- New PDF detected: {pdf_filename}. Starting agent workflow. ---")
            
            try:
                # Clear memory for each new PDF to avoid cross-contamination
                memory.clear() 
                
                # Invoke the agent executor with the PDF path as input
                agent_executor.invoke({"input": f"Process the receipt PDF located at: {pdf_path}",
                                    "tools": [SystemMessage(content=str(tool)) for tool in all_tools], 
                                    "tool_names": [tool.name for tool in all_tools],})
                
                print(f"\n--- Agent workflow completed for {pdf_filename} ---")

            except Exception as e:
                print(f"\n--- An unexpected error occurred during agent execution for {pdf_filename}: {e} ---")
                # If the agent itself crashes, attempt to move the original PDF to error folder manually
                try:
                    if os.path.exists(pdf_path): # Only move if it hasn't been moved by a tool already
                        tools_by_name["request_human_review"].invoke({
                            "pdf_path": pdf_path,
                            "problem_description": f"Agent workflow crashed: {e}",
                            "error_pdf_folder": error_dir # Use error_dir here, not error_pdf_folder directly
                        })
                        print(f"Moved '{pdf_filename}' to error folder due to agent crash.")
                except Exception as move_err:
                    print(f"CRITICAL: Also failed to move original PDF to error folder: {move_err}")

        time.sleep(5) # Check every 5 seconds


# def monitor_and_process_pdfs(input_dir, processed_dir, error_dir, crop_img_dir, json_log_file):
#     """Monitors input_dir for new PDFs and processes them step-by-step."""
    
#     while True:
#         print(f"\nMonitoring '{input_dir}' for new PDFs...")
#         current_pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

#         for pdf_filename in current_pdfs:
#             pdf_path = os.path.join(input_dir, pdf_filename)

#             if not os.path.exists(pdf_path):
#                 print(f"Skipping '{pdf_filename}' (already moved from input folder).")
#                 continue

#             print(f"\n--- New PDF detected: {pdf_filename}. Starting sequential workflow. ---")
            
#             try:
#                 # Step 1: Extract and Crop Images
#                 print(f"\n--- Step 1: Extracting and cropping images from {pdf_filename} ---")
#                 image_paths_json_str = tools_by_name["extract_and_crop_receipt_images"].invoke({
#                     "pdf_path": pdf_path,
#                     "cropped_images_folder": crop_img_dir
#                 })
#                 print(f"Image Extraction Result: {image_paths_json_str}")
#                 if "ERROR:" in image_paths_json_str:
#                     raise Exception(f"Image extraction failed: {image_paths_json_str}")
                
#                 # The image_paths_json_str contains a dictionary, get the list of paths
#                 image_paths_data = json.loads(image_paths_json_str)
#                 # Assuming the structure is {'pdf_filename': ['path1', 'path2']}
#                 # We need the list of paths for the next step.
#                 image_paths_list = []
#                 # Iterate through values of the dictionary, expecting lists of paths
#                 for _, value in image_paths_data.items():
#                     if isinstance(value, list):
#                         image_paths_list.extend(value)
                
#                 if not image_paths_list:
#                     raise Exception("No valid image paths were extracted for processing.")
                
#                 image_paths_for_gemini = json.dumps(image_paths_list) # Re-dump as list string


#                 # Step 2: Extract Data from Images
#                 print(f"\n--- Step 2: Extracting data from images using Gemini for {pdf_filename} ---")
#                 # The extract_data_from_images tool directly uses the imported llm, no need to pass it here.
#                 extracted_data_str = tools_by_name["extract_data_from_images"].invoke({
#                     "image_paths_json_str": image_paths_for_gemini
#                 })
#                 print(f"Data Extraction Result: {extracted_data_str}")
#                 if "ERROR:" in extracted_data_str:
#                     raise Exception(f"Data extraction failed: {extracted_data_str}")

#                 # Step 3: Validate Extracted Data
#                 print(f"\n--- Step 3: Validating extracted data for {pdf_filename} ---")
#                 validation_result = tools_by_name["validate_extracted_data"].invoke({
#                     "json_data_str": extracted_data_str
#                 })
#                 print(f"Validation Result: {validation_result}")
                
#                 final_extracted_data_json = extracted_data_str # Initialize with current data
                
#                 if validation_result.startswith("ERROR:"):
#                     print(f"Validation FAILED: {validation_result}")
#                     # If validation fails, proceed to human validation
#                     print(f"\n--- Step 4: Human Validation (Gradio) for {pdf_filename} (due to validation errors) ---")
#                     human_validation_result = tools_by_name["human_validation_with_gradio"].invoke({
#                         "extracted_data_str": extracted_data_str, # Pass the original, unvalidated data
#                         "image_paths_json_str": image_paths_for_gemini
#                     })
#                     print(f"Human Validation Result: {human_validation_result}")

#                     if human_validation_result.startswith("APPROVED:"):
#                         final_extracted_data_json = human_validation_result.replace("APPROVED:", "", 1).strip()
#                         print("Human APPROVED and potentially corrected the data.")
#                     else:
#                         raise Exception(f"Human REJECTED the data or an error occurred during human validation: {human_validation_result}")
#                 else:
#                     # Validation was OK, extract the cleaned JSON from "OK:[json]"
#                     final_extracted_data_json = validation_result.replace("OK:", "", 1).strip()
#                     print("Validation OK. Proceeding with processed data.")

#                 # Step 5: Manage Processed Files
#                 print(f"\n--- Step 5: Managing processed files for {pdf_filename} ---")
#                 manage_result = tools_by_name["manage_processed_receipt_files"].invoke({
#                     "original_pdf_path": pdf_path,
#                     "extracted_json_str": final_extracted_data_json,
#                     "output_pdf_folder": processed_dir,
#                     "master_json_file_path": json_log_file
#                 })
#                 print(f"File Management Result: {manage_result}")
#                 if "ERROR:" in manage_result:
#                     raise Exception(f"File management failed: {manage_result}")
                
#                 print(f"\n--- Workflow completed successfully for {pdf_filename} ---")

#             except Exception as e:
#                 print(f"\n--- An error occurred during sequential execution for {pdf_filename}: {e} ---")
#                 try:
#                     if os.path.exists(pdf_path):
#                         tools_by_name["request_human_review"].invoke({
#                             "pdf_path": pdf_path,
#                             "problem_description": f"Sequential workflow failed: {e}",
#                             "error_pdf_folder": error_pdf_folder # Ensure this uses the correct variable
#                         })
#                     print(f"Moved '{pdf_filename}' to error folder for human review.")
#                 except Exception as move_err:
#                     print(f"CRITICAL: Also failed to move original PDF to error folder after sequential error: {move_err}")

#         time.sleep(5) # Check every 5 seconds

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create a dummy PDF for testing if input_pdf_folder is empty
    if not os.listdir(input_pdf_folder):
        print(f"'{input_pdf_folder}' is empty.")


    # Start monitoring and processing
    print(f"Starting receipt processing agent. Input folder: {input_pdf_folder}")
    monitor_and_process_pdfs(
        input_pdf_folder,
        output_pdf_folder,
        error_pdf_folder,
        cropped_images_folder,
        json_output_file
    )