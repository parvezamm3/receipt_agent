import os
import fitz
from PIL import Image, ImageChops
import io
import cv2 
import numpy as np
import json
import shutil
from datetime import datetime
import re
from langchain_core.tools import tool
import google.generativeai as genai
from dotenv import load_dotenv
import gradio as gr
import time
import sys
from functools import partial
import http.server
import socketserver
import socket
import threading
import requests

# Start PDF SERVER to serve pdf file in gradio
class ThreadedHTTPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class PDFServer:
    def __init__(self, directory: str, port: int = 0):  # Use 0 for dynamic port
        self.directory = directory
        self.port = port if port else self._get_free_port()
        self.httpd = None
        self.thread = None
        self.lock = threading.Lock()

    def _get_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            return s.getsockname()[1]

    def start(self):
        with self.lock:
            if self.httpd:
                return  # Already running

            handler =  partial(http.server.SimpleHTTPRequestHandler, directory=self.directory)
            self.httpd = ThreadedHTTPServer(('localhost', self.port), handler)
            self.httpd.timeout = 0.5

            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            print(f"[PDFServer] Serving {self.directory} at http://localhost:{self.port}")

    def stop(self):
        with self.lock:
            if self.httpd:
                print("[PDFServer] Shutting down HTTP server...")
                try:
                    try:
                        requests.get(f"http://localhost:{self.port}/__shutdown_ping__", timeout=1)
                    except:
                        pass

                    httpd = self.httpd
                    thread = self.thread
                    print(httpd, thread)
                    self.httpd = None
                    self.thread = None
                    print(f"[PDFServer] Before shutdown - thread alive: {self.thread.is_alive() if self.thread else 'No thread'}")
                    self.httpd.shutdown()
                    self.httpd.server_close()
                    print("[PDFServer] HTTP server shutdown complete.")
                except Exception as e:
                    print(f"[PDFServer] Error during shutdown: {e}")
                finally:
                    self.httpd = None
                    self.thread = None

    def get_url(self, filename: str) -> str:
        return f"http://localhost:{self.port}/{filename}"
    
# --- Load Environment Variables for tools.py ---
load_dotenv()

# Configure genai. This will use GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Helper Function ---
def _resize_image_for_gemini(image, max_size=2000):
    """Resizes a PIL Image to a maximum dimension for Gemini's input limits."""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


# --- Tool 1: Extract and Crop Images from PDF ---
@tool
def extract_and_crop_receipt_images(pdf_path: str, cropped_images_folder: str) -> str:
    """
    Extracts images from a PDF, crops the main content area (receipt),
    and saves cropped images to the output folder.
    Returns a JSON string of {'pdf_filename': ['path/to/img1.png', 'path/to/img2.png', ...]}
    or an error message.
    """
    cropped_image_paths = []
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    if not os.path.exists(pdf_path):
        return f"ERROR: PDF file not found at {pdf_path}"
    if not os.path.isdir(cropped_images_folder):
        os.makedirs(cropped_images_folder, exist_ok=True) # Ensure folder exists

    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=300) # Use a higher DPI for better OCR quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Automatic cropping of whitespace with padding
            bg = Image.new(img.mode, img.size, (255, 255, 255))
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()

            if bbox:
                padding = 20
                left, upper, right, lower = bbox
                img_width, img_height = img.size

                left = max(0, left - padding)
                upper = max(0, upper - padding)
                right = min(img_width, right + padding)
                lower = min(img_height, lower + padding)

                cropped_img = img.crop((left, upper, right, lower))
            else:
                cropped_img = img # No content detected, keep full page

            output_image_path = os.path.join(cropped_images_folder, f"{base_filename}_page_{page_num}.png")
            cropped_img.save(output_image_path)
            cropped_image_paths.append(output_image_path)

        doc.close()
        return json.dumps({os.path.basename(pdf_path): cropped_image_paths})
    except Exception as e:
        return f"ERROR: Failed to extract and crop images from '{pdf_path}': {e}"
    

# --- Tool 2: Extract Receipt Data using Gemini Multi-modal ---
@tool
def extract_data_from_images(image_paths_json_str: str, model_name: str = "gemini-2.0-flash-lite") -> str:
    """
    Extracts structured data from a list of receipt image paths using the Gemini API.
    Input must be a JSON string like '["path/to/img1.png", "path/to/img2.png"]'.
    Returns a JSON string of the extracted receipt data, or an error message.
    """
    try:
        image_paths = json.loads(image_paths_json_str)
        if not isinstance(image_paths, list):
            return "ERROR: Input image_paths_json_str must be a JSON list of strings."

        model = genai.GenerativeModel(model_name)
        prompt_parts = [
            """
            あなたは領収書のデータを抽出し、構造化するエキスパートAIアシスタントです。
            提供された領収書画像から以下の詳細を抽出してください。領収書は複数のページにわたる場合があります。
            すべてのページからの情報を単一のJSONオブジェクトに統合してください。フィールドが見つからない場合は「null」を使用します。
            出力は、JSONオブジェクト以外の余分なテキストやフォーマットを含まない、クリーンなJSONオブジェクトでなければなりません。

            抽出するフィールド:
            - "宛名" (Addressee): サービス/製品の受取人の名前。
            - "日付" (Date): 取引の日付。YYYYMMDD形式で指定してください。
            - "金額" (Amount): 取引の合計金額。数字のみ、カンマや通貨記号なしで返答してください。
            - "相手先" (Vendor): ベンダー情報。辞書形式 { "名前"(Name), "住所" (Address), "電話番号" (Phone Number) }。
            - "登録番号" (Invoice Registration Number): 日本のインボイス登録番号。
            - "摘要" (Description): 簡単な説明または品目の詳細。リスト形式 [[名前, 数量, 単価, 合計]]。

            結果を単一の、クリーンなJSONオブジェクトとして出力してください。
            """
        ]

        image_parts = []
        for path in image_paths:
            if not os.path.exists(path):
                print(f"Warning: Image file not found at {path}. Skipping.")
                continue
            try:
                img = Image.open(path)
                resized_img = _resize_image_for_gemini(img) # Use internal helper
                image_parts.append(resized_img)
            except Exception as e:
                return f"ERROR: Could not load or resize image {path}. Error: {e}"

        if not image_parts:
            return "ERROR: No valid images were successfully loaded for extraction."

        prompt_parts.extend(image_parts)
        response = model.generate_content(prompt_parts)
        raw_text = response.text

        if raw_text.startswith("```json"):
            cleaned_text = raw_text.replace("```json", "", 1).strip()
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()
        else:
            cleaned_text = raw_text.strip()

        json_result = json.loads(cleaned_text)
        return json.dumps(json_result, ensure_ascii=False) # Return JSON string
    except json.JSONDecodeError as e:
        return f"ERROR: Gemini output not valid JSON. Raw: {raw_text[:200]}... Error: {e}"
    except Exception as e:
        return f"ERROR: Failed to extract data from images ({image_paths_json_str}): {e}"


# --- Tool 3: Validate Extracted Receipt Data ---
@tool
def validate_extracted_data(json_data_str: str) -> str:
    """
    Validates extracted receipt data for correct format (dateYYYYMMDD, amount numeric, vendor exists, etc.).
    Input must be a JSON string of the extracted data.
    Returns 'OK:[validated_json_str]' if valid and potentially cleaned/formatted,
    otherwise returns 'ERROR:[description of all validation errors]'.
    """
    try:
        data = json.loads(json_data_str)
    except json.JSONDecodeError:
        return "ERROR: Input is not a valid JSON string for validation."

    errors = []
    
    # Validate 日付 (Date)
    date_str = data.get("日付")
    if not date_str:
        errors.append("日付 (Date) is missing.")
    else:
        try:
            # Attempt to parse various date formats
            parsed_date = None
            for fmt in ["%Y年%m月%d日", "%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"]:
                try:
                    parsed_date = datetime.strptime(date_str.replace(' ', ''), fmt)
                    break
                except ValueError:
                    continue
            if parsed_date:
                data["日付"] = parsed_date.strftime("%Y%m%d") # Update to YYYYMMDD for consistency
            else:
                errors.append(f"日付 '{date_str}' は認識される YYYYMMDD、YYYY/MM/DD、または YYYY-MM-DD 形式ではありません。")
        except Exception:
             errors.append(f"日付 '{date_str}' は予期しない形式であるか無効です。")

    # Validate 金額 (Amount)
    amount_str = str(data.get("金額", "")).replace(",", "").strip()
    if not amount_str:
        errors.append("金額は抽出されていません。ファイルを確認してください。")
    elif not re.fullmatch(r'\d+', amount_str): # Check if purely numeric
        errors.append(f"金額 '{data.get('金額')}' は純粋な数値ではありません。")
    else:
        data["金額"] = amount_str # Store as string without commas

    # Validate 相手先 (Vendor Name)
    vendor_obj = data.get("相手先", {})
    vendor_name = vendor_obj.get("名前")
    if not vendor_name or not isinstance(vendor_name, str) or not vendor_name.strip():
        errors.append("相手先が抽出されていません。ファイルを確認してください。")

    registration_no = data.get("登録番号", "")
    if not registration_no:
        errors.append("登録番号が抽出されていません。ファイルを確認してください。")

    # # Validate 摘要 (Description) - check if it's a list of lists with 4 items
    # description_items = data.get("摘要")
    # if description_items is not None: # Allow null, but if present, must be list
    #     if not isinstance(description_items, list):
    #         errors.append("摘要 (Description) is not a list.")
    #     else:
    #         for i, item in enumerate(description_items):
    #             if not (isinstance(item, list) and len(item) == 4 and all(isinstance(x, (str, int, float)) for x in item)):
    #                 errors.append(f"摘要 item {i} is not in [name, quantity, unit_price, total] format or has invalid types.")

    if errors:
        return "ERROR: " + "\n".join(errors)
    else:
        # Return OK along with the potentially cleaned/formatted JSON data
        return "OK:" + json.dumps(data, ensure_ascii=False)


# --- Tool 4: Manage Processed Files (Rename, Copy, Update Master JSON) ---
@tool
def manage_processed_receipt_files(original_pdf_path: str, extracted_json_str: str,
                                   output_pdf_folder: str, master_json_file_path: str, success_pdf_folder: str) -> str:
    """
    Renames the original PDF file based on extracted data, copies it to the output folder,
    and updates a master JSON log file.
    Input must include original_pdf_path and the extracted data as a JSON string.
    Returns 'SUCCESS:[new_filename]' or 'ERROR:[description]'.
    """
    try:
        extracted_data = json.loads(extracted_json_str)
    except json.JSONDecodeError:
        return f"ERROR: manage_processed_receipt_files received invalid JSON string: {extracted_json_str}"

    if not all(k in extracted_data for k in ["日付", "金額"]):
        return "ERROR: Extracted data missing essential fields (日付 or 金額) for renaming."

    date = extracted_data["日付"]
    amount = extracted_data["金額"]
    vendor_name = extracted_data.get("相手先", {}).get("名前", "不明")
    vendor_name_sanitized = "".join(c for c in vendor_name if c.isalnum() or c in (' ', '_', '-')).strip() or "不明"
    if len(vendor_name_sanitized) > 50: # Truncate for filename length
        vendor_name_sanitized = vendor_name_sanitized[:50]

    # Generate a unique random string to prevent filename collisions
    import uuid
    random_string = uuid.uuid4().hex[:6]

    original_filename = os.path.basename(original_pdf_path)
    new_filename_base = f"{date}_{amount}_{vendor_name_sanitized}"
    new_pdf_name = f"{new_filename_base}_{random_string}.pdf"
    destination_path = os.path.join(output_pdf_folder, new_pdf_name)

    # Handle potential filename conflicts (though UUID should minimize this)
    counter = 1
    while os.path.exists(destination_path):
        destination_path = os.path.join(output_pdf_folder, f"{new_filename_base}_{random_string}_{counter}.pdf")
        counter += 1

    if not os.path.exists(original_pdf_path):
        return f"ERROR: Original PDF '{original_pdf_path}' not found for copying."

    try:
        print(f"--- Attempting to copy '{original_pdf_path}' to '{destination_path}' ---")
        shutil.copy(original_pdf_path, destination_path)
        print(f"--- Successfully copied to '{destination_path}'. ---")

        master_data = {}
        print(master_data)
        if os.path.exists(master_json_file_path):
            with open(master_json_file_path, "r", encoding="utf-8") as f:
                try:
                    master_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Master JSON '{master_json_file_path}' is corrupt or empty. Starting fresh.")
                    master_data = {}

        # Use the new filename as the key in the master log
        extracted_data["元名"] = original_filename
        master_data[new_pdf_name] = extracted_data
        
        with open(master_json_file_path, "w", encoding="utf-8") as f:
            json.dump(master_data, f, ensure_ascii=False, indent=4)
        print(f"Updated master JSON log at {master_json_file_path}")

        success_pdf_path = os.path.join(success_pdf_folder, original_filename)
        shutil.move(original_pdf_path, success_pdf_path)
        print(f"--- Successfully copied to '{success_pdf_path}'. ---")
        # Update the master JSON log

        return f"SUCCESS: Copied '{original_filename}' to '{destination_path}'. Master log updated. New filename: {new_pdf_name}"

    except Exception as e:
        return f"ERROR: Failed to copy or update log for '{original_filename}': {e}"
# --- Tool 5: Request Human Review ---
@tool
def request_human_review(pdf_path: str, problem_description: str, error_pdf_folder: str) -> str:
    """
    Moves a problematic PDF to an error folder for human review and returns a confirmation message.
    """
    if not os.path.exists(pdf_path):
        return f"ERROR: Cannot move '{pdf_path}' for human review; file not found."
    if not os.path.isdir(error_pdf_folder):
        os.makedirs(error_pdf_folder, exist_ok=True)

    try:
        destination_path = os.path.join(error_pdf_folder, os.path.basename(pdf_path))
        shutil.move(pdf_path, destination_path)
        return (f"HUMAN_REVIEW_REQUIRED: Moved '{os.path.basename(pdf_path)}' to '{error_pdf_folder}' "
                f"for manual review. Problem: {problem_description}")
    except Exception as e:
        return f"ERROR: Failed to move '{pdf_path}' to error folder for human review: {e}"
# import sys
from typing import List
# --- Tool 6: Human Validation with Gradio UI ---
@tool
def human_validation_with_gradio(
    original_pdf_path: str, 
    extracted_data_str: str, 
    image_paths_json_str: str, 
    validated_data_str: str,
    pdf_server: PDFServer) -> str:
    """
    Launches a Gradio interface for human validation of extracted data and images.
    Displays the original PDF (with download for zoom), images, and a JSON editor.
    Returns 'APPROVED:[json_data]' or 'REJECTED:[feedback]' based on human input.
    """
    try:
        extracted_data = json.loads(extracted_data_str)
        image_paths: List[str] = json.loads(image_paths_json_str)
        # PDF path validation
        if not os.path.exists(original_pdf_path):
             return "REJECTED: Original PDF file not found for display."

        # Determine initial validation status message
        initial_editor_data = (
            json.loads(validated_data_str[3:])
            if validated_data_str.startswith("OK:")
            else extracted_data
        )

        validation_status = (
            "Initial Validation Status: SUCCESS"
            if validated_data_str.startswith("OK:")
            else f"Initial Validation Status: FAILED - {validated_data_str[6:]}"
        )

        # Start server
        pdf_server.start()
        pdf_filename = os.path.basename(original_pdf_path)
        pdf_url = pdf_server.get_url(pdf_filename)

        # Shared state
        validation_result_holder = []
        server_ready_to_close = threading.Event()

        def submit(edited_json: str, action_type: str, feedback: str = ""):
            try:
                parsed = json.loads(edited_json)
                if action_type == "approve":
                    result = "APPROVED:" + json.dumps(parsed, ensure_ascii=False)
                else:
                    result = f"REJECTED: Human rejected. Feedback: {feedback}"
                validation_result_holder.append(result)
                server_ready_to_close.set()
                return f"Process completed. You selected: {action_type}. Please close the browser."
            except Exception as e:
                error_msg = f"REJECTED: JSON error: {e}"
                validation_result_holder.append(error_msg)
                server_ready_to_close.set()
                return f"Process completed. You selected: {action_type}. Please close the browser."
        
        def embed_pdf(url): return f'<iframe src="{url}" width="100%" height="600px" style="border:none;"></iframe>'

        demo = gr.Blocks()

        with demo:
            gr.Markdown("# Receipt Data Human Validation")
            gr.Markdown("Please review the extracted data and the original receipt images. Edit the JSON if necessary.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"**{validation_status}**") 
                with gr.Column():
                    status = gr.Textbox(label="Status", interactive=False, visible=True)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Original PDF Document")
                    gr.HTML(embed_pdf(pdf_url))

                    if image_paths:
                        gr.Markdown("## Extracted Images")
                        for i, path in enumerate(image_paths):
                            if os.path.exists(path):
                                gr.Image(value=path, label=f"Image {i+1}", type="filepath", height=200)
                with gr.Column():
                    gr.Markdown("## Extracted Data (Editable JSON)")
                    json_editor = gr.Textbox(
                        value=json.dumps(initial_editor_data, ensure_ascii=False, indent=2),
                        label="Edit Receipt Data (JSON Format)",
                        lines=15, # Give it some height for easier editing
                        interactive=True, 
                    )
                    feedback = gr.Textbox(placeholder="Reason for rejection (optional)")
                    with gr.Row():
                        approve_btn = gr.Button("Approve & Submit (OK)", variant="primary")
                        reject_btn = gr.Button("Reject", variant="stop")
            

            approve_btn.click(
                fn=lambda x: submit(x, "approve"),
                inputs=[json_editor],
                outputs=[status]
            )
            reject_btn.click(
                fn=lambda x, y: submit(x, "reject", y),
                inputs=[json_editor, feedback],
                outputs=[status]
            )
            
        print("\n--- Gradio UI Launched for Human Validation ---")
        # Launch in thread to avoid blocking LangGraph
        def launch_ui():
            demo.launch(share=False, inbrowser=True, show_api=False)

        # Start Gradio in a separate thread
        gradio_thread = threading.Thread(target=launch_ui)
        gradio_thread.start()

        # Wait for the signal that the submission is complete
        server_ready_to_close.wait() # This will block until server_ready_to_close.set() is called
        print("[Gradio] Human submitted response. Cleaning up...")
        pdf_server.stop()
        print("Server signaled to close. Waiting briefly for shutdown...")
        
        if gradio_thread.is_alive():
             print("Gradio thread is still alive after signal. Attempting to join...")
             # Timeout the join to prevent indefinite blocking
             gradio_thread.join(timeout=3) 

        print("Gradio thread likely closed or timed out. Proceeding with workflow.")
        
        while not validation_result_holder:
            print("Waiting for human validation via Gradio UI... (Please close the Gradio tab after submitting)")
            time.sleep(0.5) # Check every 0.5 seconds

        return validation_result_holder[0] if validation_result_holder else "REJECTED: No response received."

    except Exception as e:
        print(f"ERROR: An error occurred during Gradio human validation setup: {e}")
        return f"ERROR: An error occurred during Gradio human validation setup: {e}"