import logging
import os
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from pathlib import Path
from google.cloud import vision
from google.oauth2 import service_account

from backend.models import ExtractedText, MenuScanResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Menu Scanner API", version="1.0.0")

BACKEND_DIR = Path(__file__).resolve().parent


# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOC_AI_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOC_AI_KEY")
GOOGLE_VISION_CREDENTIALS = {
    "type": os.environ.get("TYPE"),
    "project_id": os.environ.get("PROJECT_ID"),
    "private_key_id": os.environ.get("PRIVATE_KEY_ID"),
    # Handle escaped newlines
    "private_key": os.environ.get("PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.environ.get("CLIENT_EMAIL"),
    "client_id": os.environ.get("CLIENT_ID"),
    "auth_uri": os.environ.get("AUTH_URI"),
    "token_uri": os.environ.get("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.environ.get("AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.environ.get("CLIENT_X509_CERT_URL"),
    "universe_domain": os.environ.get("UNIVERSE_DOMAIN")
}

if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or not AZURE_DOCUMENT_INTELLIGENCE_KEY:
    raise ValueError(
        "Azure Document Intelligence credentials not found in environment variables")
if not GOOGLE_VISION_CREDENTIALS:
    raise ValueError(
        "Google vision credentials not found in environment variables")


document_intelligence_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
)

credentials = service_account.Credentials.from_service_account_info(
    GOOGLE_VISION_CREDENTIALS)
google_vision_client = vision.ImageAnnotatorClient(credentials=credentials)


@app.get("/")
def read_index():
    return FileResponse(BACKEND_DIR.joinpath("frontend", "index.html"))


@app.post("/scan-menu", response_model=MenuScanResult)
async def scan_menu_item(file: UploadFile = File(...)):
    """
    Scan menu item from uploaded image using Azure Document Intelligence
    """
    start_time = time.time()

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )

        # Read the uploaded file
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(
            f"Processing image: {file.filename}, Size: {len(file_content)} bytes")

        extracted_texts = []
        raw_text_parts = []

        image = vision.Image(content=file_content)

        response = google_vision_client.text_detection(image=image)
        texts = response.text_annotations

        for text in texts:
            extracted_texts.append(ExtractedText(
                content=text.description,
            ))

        poller = document_intelligence_client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=file_content)
        )

        result: AnalyzeResult = poller.result()

        if result.pages:
            for page in result.pages:
                if page.lines:
                    for line in page.lines:
                        text_content = line.content
                        raw_text_parts.append(text_content)

                        line_confidence = None
                        if page.words:
                            line_words = []
                            for word in page.words:
                                if any(word.span.offset >= span.offset and
                                       (word.span.offset +
                                        word.span.length) <= (span.offset + span.length)
                                       for span in line.spans):
                                    line_words.append(word)

                            if line_words:
                                line_confidence = sum(
                                    word.confidence for word in line_words if word.confidence) / len(line_words)

                        extracted_texts.append(ExtractedText(
                            content=text_content,
                            confidence=line_confidence,
                        ))

        # Combine all text
        raw_text = '\n'.join(raw_text_parts)

        processing_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"Successfully processed image. Extracted {len(extracted_texts)} text elements in {processing_time}ms")

        return MenuScanResult(
            extracted_text=extracted_texts,
            raw_text=raw_text,
            processing_time_ms=processing_time,
            success=True
        )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        error_message = str(e)
        logger.error(f"Error processing image: {error_message}")

        return MenuScanResult(
            extracted_text=[],
            raw_text="",
            processing_time_ms=processing_time,
            success=False,
            error_message=error_message
        )
