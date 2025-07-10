
import unittest
import os
import json
import shutil
from unittest.mock import patch, MagicMock
from PIL import Image

# Add the parent directory to the Python path to import the tools module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import (
    extract_and_crop_receipt_images,
    extract_data_from_images,
    validate_extracted_data,
    manage_processed_receipt_files,
    request_human_review,
)

class TestTools(unittest.TestCase):

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
        
        # Create a dummy image file using a context manager to ensure closure
        from PIL import Image
        with Image.new('RGB', (100, 100), color='red') as img:
            img.save(os.path.join(self.cropped_images_folder, "dummy_image.png"))
        self.dummy_image_path = os.path.join(self.cropped_images_folder, "dummy_image.png")

    def tearDown(self):
        """Clean up test environment."""
        import time
        time.sleep(0.1)  # Give the OS a moment to release the file
        shutil.rmtree(self.cropped_images_folder)
        shutil.rmtree(self.output_pdf_folder)
        shutil.rmtree(self.success_pdf_folder)
        shutil.rmtree(self.error_pdf_folder)
        if os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)
        if os.path.exists(self.master_json_file_path):
            os.remove(self.master_json_file_path)

    @patch('tools.extract_and_crop_receipt_images')
    def test_extract_and_crop_receipt_images_success(self, mock_tool):
        mock_tool.func.return_value = json.dumps({os.path.basename(self.test_pdf_path): [self.dummy_image_path]})
        result = mock_tool.func(self.test_pdf_path, self.cropped_images_folder)
        result_json = json.loads(result)
        self.assertIn(os.path.basename(self.test_pdf_path), result_json)
        self.assertEqual(len(result_json[os.path.basename(self.test_pdf_path)]), 1)
        self.assertEqual(result_json[os.path.basename(self.test_pdf_path)][0], self.dummy_image_path)

    @patch('tools.extract_and_crop_receipt_images')
    def test_extract_and_crop_receipt_images_pdf_not_found(self, mock_tool):
        mock_tool.func.return_value = "ERROR: PDF not found"
        result = mock_tool.func("non_existent.pdf", self.cropped_images_folder)
        self.assertTrue(result.startswith("ERROR:"))

    @patch('tools.extract_data_from_images')
    def test_extract_data_from_images_success(self, mock_tool):
        mock_tool.func.return_value = '{"key": "value"}'
        image_paths_json = json.dumps([self.dummy_image_path])
        result = mock_tool.func(image_paths_json)
        self.assertEqual(result, '{"key": "value"}')

    @patch('tools.extract_data_from_images')
    def test_extract_data_from_images_invalid_json(self, mock_tool):
        mock_tool.func.return_value = "ERROR: Invalid JSON"
        result = mock_tool.func("not a json string")
        self.assertTrue(result.startswith("ERROR:"))

    @patch('tools.validate_extracted_data')
    def test_validate_extracted_data_success(self, mock_tool):
        mock_tool.func.return_value = "OK:{\"key\": \"value\"}"
        valid_data = json.dumps({"日付": "20250101", "金額": "1000", "相手先": {"名前": "Test Vendor"}, "登録番号": "T123456789"})
        result = mock_tool.func(valid_data)
        self.assertTrue(result.startswith("OK:"))

    @patch('tools.validate_extracted_data')
    def test_validate_extracted_data_missing_fields(self, mock_tool):
        mock_tool.func.return_value = "ERROR: 金額は抽出されていません。"
        invalid_data = json.dumps({"日付": "20250101"})
        result = mock_tool.func(invalid_data)
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("金額は抽出されていません。", result)

    @patch('tools.validate_extracted_data')
    def test_validate_extracted_data_invalid_date(self, mock_tool):
        mock_tool.func.return_value = "ERROR: 日付が不正です。"
        invalid_data = json.dumps({"日付": "2025/13/01", "金額": "1000", "相手先": {"名前": "Test Vendor"}, "登録番号": "T123456789"})
        result = mock_tool.func(invalid_data)
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("日付", result)

    @patch('tools.validate_extracted_data')
    def test_validate_extracted_data_non_numeric_amount(self, mock_tool):
        mock_tool.func.return_value = "ERROR: 金額が不正です。"
        invalid_data = json.dumps({"日付": "20250101", "金額": "1000a", "相手先": {"名前": "Test Vendor"}, "登録番号": "T123456789"})
        result = mock_tool.func(invalid_data)
        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn("金額", result)

    @patch('tools.manage_processed_receipt_files')
    def test_manage_processed_receipt_files_success(self, mock_tool):
        mock_tool.func.return_value = "SUCCESS:test.pdf"
        extracted_data = json.dumps({"日付": "20250101", "金額": "1000", "相手先": {"名前": "Test Vendor"}})
        result = mock_tool.func(self.test_pdf_path, extracted_data, self.output_pdf_folder, self.master_json_file_path, self.success_pdf_folder)
        self.assertTrue(result.startswith("SUCCESS:"))

    @patch('tools.manage_processed_receipt_files')
    def test_manage_processed_receipt_files_invalid_json(self, mock_tool):
        mock_tool.func.return_value = "ERROR: Invalid JSON"
        result = mock_tool.func(self.test_pdf_path, "not a json string", self.output_pdf_folder, self.master_json_file_path, self.success_pdf_folder)
        self.assertTrue(result.startswith("ERROR:"))

    @patch('tools.manage_processed_receipt_files')
    def test_manage_processed_receipt_files_missing_fields(self, mock_tool):
        mock_tool.func.return_value = "ERROR: Missing fields"
        extracted_data = json.dumps({"日付": "20250101"})
        result = mock_tool.func(self.test_pdf_path, extracted_data, self.output_pdf_folder, self.master_json_file_path, self.success_pdf_folder)
        self.assertTrue(result.startswith("ERROR:"))

    @patch('tools.request_human_review')
    def test_request_human_review_success(self, mock_tool):
        mock_tool.func.return_value = "HUMAN_REVIEW_REQUIRED:Test problem"
        result = mock_tool.func(self.test_pdf_path, "Test problem", self.error_pdf_folder)
        self.assertTrue(result.startswith("HUMAN_REVIEW_REQUIRED:"))

    @patch('tools.request_human_review')
    def test_request_human_review_file_not_found(self, mock_tool):
        mock_tool.func.return_value = "ERROR: File not found"
        result = mock_tool.func("non_existent.pdf", "Test problem", self.error_pdf_folder)
        self.assertTrue(result.startswith("ERROR:"))

if __name__ == '__main__':
    unittest.main()
