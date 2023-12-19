from src.document_scanner.main_document_scanner import document_scanner
from src.image_capture.image_capture import capture_image
from src.text_detector.main_text_detector import text_detector

def main():
    captured_image = capture_image()
    scanned_document = document_scanner(captured_image)
    detected_text = text_detector(scanned_document)

if __name__ == '__main__':
    main()