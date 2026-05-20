import fitz  # PyMuPDF 라이브러리 [cite: 369-370]
import re    # 정규표현식 라이브러리 [cite: 372]
from pydantic import BaseModel
from typing import List

# 1. 데이터 규격 정의 (Schema) [cite: 378-386]
class DocumentChunk(BaseModel):
    page_number: int
    content: str
    source_file: str

class DataPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path # 사용자가 보낸 파일 경로 [cite: 393]

    # 2. PDF에서 텍스트 추출 [cite: 396, 417-427]
    def parse_pdf(self) -> List[DocumentChunk]:
        doc = fitz.open(self.file_path)
        chunks = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            cleaned_text = self._clean_text(text)
            
            # 페이지별로 객체 생성
            chunks.append(DocumentChunk(
                page_number=page_num + 1,
                content=cleaned_text,
                source_file=self.file_path
            ))
        return chunks

    # 3. 텍스트 정제 (정규표현식 활용) [cite: 428-438]
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text) # 줄바꿈 및 불필요한 공백 제거
        text = re.sub(r'[^\w\s\.\?\!\,]', '', text) # 특수 기호 제거
        return text.strip()