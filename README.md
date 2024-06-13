Hello!!

I want you to be a better HR employee.

Don't tell anyone you're using me :)

Run this in a python environmen:

To set up OpenAi key:
set OPENAPI_KEY=your_access_token

For dependencies:
pip install openai  langchain langchain_openai pypdf streamlit streamlit_chat
pip install camelot-py[cv]
pip install pdfminer.six
pip install pdfplumber pytesseract pdf2image

Install Tesseract-OCR:
Download the Tesseract installer for Windows from this link https://github.com/UB-Mannheim/tesseract/wiki  and follow the installation instructions. Add path (usually C:\Program Files\Tesseract-OCR).
Download Poppler for Windows:

Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases
Download the latest release zip file (e.g., poppler-0.68.0_x86.zip or poppler-0.68.0_x64.zip depending on your system architecture).
Extract the Poppler binaries:
Extract the contents of the downloaded zip file to a directory, for example, C:\poppler.
Add Poppler to your system path:
In the Environment Variables window, add the path to the bin folder inside the extracted Poppler folder (e.g., C:\poppler\bin).

To run:
streamlit run HRAC.py

Have fun!
