from main_document_scanner import document_scanner
from main_text_detector import text_detector

def main():
    scanned_document = document_scanner('image.png')
    detected_text = text_detector(scanned_document)

if __name__ == '__main__':
    main()