import pdfplumber
from PIL import Image

def save_pdf_page_as_image(pdf_path, page_num, output_image_path):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        image = page.to_image(resolution=300)  # High resolution for better OCR
        image.save(output_image_path)

# Save the first page of the FAQ.pdf as an image
save_pdf_page_as_image("../FAQ.pdf", 0, "page_1.png")
