import logging
import os
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Menu Scanner API", version="1.0.0")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Document Intelligence configuration
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOC_AI_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOC_AI_KEY")

if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or not AZURE_DOCUMENT_INTELLIGENCE_KEY:
    raise ValueError(
        "Azure Document Intelligence credentials not found in environment variables")

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
)

# Response models


class ExtractedText(BaseModel):
    content: str
    confidence: Optional[float] = None
    bounding_box: Optional[List[float]] = None


class MenuScanResult(BaseModel):
    extracted_text: List[ExtractedText]
    raw_text: str
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


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

        # Use the Read API (prebuilt-read model) for text extraction
        # Following the official Azure sample pattern
        poller = document_intelligence_client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=file_content)
        )

        # Wait for the analysis to complete
        result: AnalyzeResult = poller.result()

        # Extract text with confidence scores and bounding boxes
        extracted_texts = []
        raw_text_parts = []

        if result.pages:
            for page in result.pages:
                if page.lines:
                    for line in page.lines:
                        # Extract text content
                        text_content = line.content
                        raw_text_parts.append(text_content)

                        # Get confidence score (lines don't have confidence, but words do)
                        # We'll calculate average confidence from words in this line
                        line_confidence = None
                        if page.words:
                            line_words = []
                            for word in page.words:
                                # Check if word belongs to this line
                                if any(word.span.offset >= span.offset and
                                       (word.span.offset +
                                        word.span.length) <= (span.offset + span.length)
                                       for span in line.spans):
                                    line_words.append(word)

                            if line_words:
                                line_confidence = sum(
                                    word.confidence for word in line_words if word.confidence) / len(line_words)

                        # Get bounding box coordinates
                        bounding_box = None
                        # if hasattr(line, 'polygon') and line.polygon:
                        #     bounding_box = [
                        #         coord for point in line.polygon for coord in [point.x, point.y]]

                        extracted_texts.append(ExtractedText(
                            content=text_content,
                            confidence=line_confidence,
                            bounding_box=bounding_box
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


#  uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


# curl -X 'POST' 'http://161.97.127.38:100/api/v1/chats/recommend-promotion' -H 'accept: application/json' -H 'Authorization: Bearer <access-token>' -H 'Content-Type: application/json' -d '{"prompt": "string", "llm": "OPENAI", "chat_history": [], "store_ids": [], "stream": false}'
