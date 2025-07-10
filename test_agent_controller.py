import unittest
import os
import json
import shutil
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path to import the module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_controller import (
    GraphState,
    call_extract_images,
    call_extract_data,
    call_validate_data,
    call_human_validation,
    call_manage_files,
    call_request_review,
    route_human_validation_result,
    PDFHandler
)

class TestAgentController(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_pdf_path = "test.pdf"
        self.cropped_images_folder = "test_cropped_images"
        self.output_pdf_folder = "test_output_pdfs"
        self.success_pdf_folder = "test_success_pdfs"
        self.error_pdf_folder = "test_error_pdfs"
        self.master_json_file_path = "test_master_data.json"

        # Create dummy folders
        os.makedirs(self.cropped_images_folder, exist_ok=True)
        os.makedirs(self.output_pdf_folder, exist_ok=True)
        os.makedirs(self.success_pdf_folder, exist_ok=True)
        os.makedirs(self.error_pdf_folder, exist_ok=True)

        # Create a dummy PDF file
        with open(self.test_pdf_path, "w") as f:
            f.write("dummy pdf content")

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.cropped_images_folder)
        shutil.rmtree(self.output_pdf_folder)
        shutil.rmtree(self.success_pdf_folder)
        shutil.rmtree(self.error_pdf_folder)
        if os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)
        if os.path.exists(self.master_json_file_path):
            os.remove(self.master_json_file_path)

    @patch('agent_controller.extract_and_crop_receipt_images')
    def test_call_extract_images_success(self, mock_tool):
        """Test successful image extraction."""
        mock_tool.invoke.return_value = json.dumps({"test.pdf": ["img1.png"]})
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=[])
        result = call_extract_images(state)
        self.assertEqual(result["processed_status"], "SUCCESS")
        self.assertEqual(result["image_paths"], ["img1.png"])

    @patch('agent_controller.extract_and_crop_receipt_images')
    def test_call_extract_images_error(self, mock_tool):
        """Test image extraction with an error."""
        mock_tool.invoke.return_value = "ERROR: Test error"
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=[])
        result = call_extract_images(state)
        self.assertEqual(result["processed_status"], "FAILED")
        self.assertEqual(result["error_message"], "ERROR: Test error")

    @patch('agent_controller.extract_data_from_images')
    def test_call_extract_data_success(self, mock_tool):
        """Test successful data extraction."""
        mock_tool.invoke.return_value = '{"key": "value"}'
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=["img1.png"], extracted_data="")
        result = call_extract_data(state)
        self.assertEqual(result["extracted_data"], '{"key": "value"}')

    @patch('agent_controller.validate_extracted_data')
    def test_call_validate_data_success(self, mock_tool):
        """Test successful data validation."""
        mock_tool.invoke.return_value = "OK:{\"key\": \"value\"}"
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=["img1.png"], extracted_data='{}', validated_data="")
        result = call_validate_data(state)
        self.assertTrue(result["validated_data"].startswith("OK:"))

    @patch('agent_controller.human_validation_with_gradio')
    def test_call_human_validation_approved(self, mock_tool):
        """Test human validation with approval."""
        mock_tool.invoke.return_value = "APPROVED:{\"key\": \"value\"}"
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=["img1.png"], extracted_data='{}', validated_data="OK:{}")
        result = call_human_validation(state)
        self.assertTrue(result["validated_data"].startswith("APPROVED:"))

    @patch('agent_controller.manage_processed_receipt_files')
    def test_call_manage_files_success(self, mock_tool):
        """Test successful file management."""
        mock_tool.invoke.return_value = "SUCCESS:test.pdf"
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=["img1.png"], validated_data="APPROVED:{\"key\": \"value\"}")
        # The router should have updated validated_data
        state["validated_data"] = state["validated_data"]
        result = call_manage_files(state)
        self.assertEqual(result["processed_status"], "SUCCESS")

    @patch('agent_controller.request_human_review')
    def test_call_request_review_success(self, mock_tool):
        """Test successful request for human review."""
        mock_tool.invoke.return_value = "HUMAN_REVIEW_REQUIRED:Test problem"
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=[], error_message="Test problem")
        result = call_request_review(state)
        self.assertEqual(result["processed_status"], "FAILED")
        self.assertTrue(result["error_message"].startswith("HUMAN_REVIEW_REQUIRED:"))

    def test_route_human_validation_result_approved(self):
        """Test routing for approved human validation."""
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=[], validated_data="APPROVED:{}")
        result = route_human_validation_result(state)
        self.assertEqual(result, "manage_files")

    def test_route_human_validation_result_rejected(self):
        """Test routing for rejected human validation."""
        state = GraphState(pdf_path=self.test_pdf_path, image_paths=[], validated_data="REJECTED:Feedback")
        result = route_human_validation_result(state)
        self.assertEqual(result, "request_review")

    @patch('agent_controller.request_human_review')
    def test_pdf_handler_on_created(self, mock_tool):
        """Test the PDF handler's on_created method."""
        handler = PDFHandler(
            input_dir=".",
            processed_dir=self.output_pdf_folder,
            error_dir=self.error_pdf_folder,
            crop_img_dir=self.cropped_images_folder,
            json_log_file=self.master_json_file_path
        )
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = self.test_pdf_path

        with patch('agent_controller.app.stream') as mock_stream:
            mock_stream.return_value = [{"__end__": {"processed_status": "SUCCESS"}}]
            handler.on_created(mock_event)
            mock_stream.assert_called_once()

if __name__ == '__main__':
    unittest.main()