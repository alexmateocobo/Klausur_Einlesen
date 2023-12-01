from main_document_scanner import document_scanner
from main_text_detector import text_detector

scanned_document = document_scanner('IMG_4801.png')
detected_text = text_detector(scanned_document)