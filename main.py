from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
import shutil
import os

import models
from database import engine, SessionLocal
from preprocessor import DataPreprocessor
from graph_builder import KnowledgeGraphBuilder
from analyzer import AnswerAnalyzer
from quiz_generator import QuizGenerator

models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="맞춤형 학습 진단 시스템 API")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------
# [데이터 전송 규격 (그릇)]
# ---------------------------------------------------------
class UserAnswerSubmit(BaseModel):
    lecture_id: int
    question_text: str
    user_answer: str

class AnswerSubmission(BaseModel):
    user_id: int
    material_id: int
    user_answer: str
    question_context: str

class FeedbackResponse(BaseModel):
    record_id: int
    error_type: str
    weak_point: str
    suggested_content: str
    status: str

@app.get("/")
def read_root():
    return {"message": "서버가 정상 작동 중입니다. Swagger UI(/docs)로 접속해주세요."}

# ---------------------------------------------------------
# [1단계] 강의자료 입력 및 지식 그래프 생성
# ---------------------------------------------------------
@app.post("/api/v1/materials", status_code=202)
async def upload_material(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = "temp_" + file.filename
    with open(file_location, "wb+") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    preprocessor = DataPreprocessor(file_location)
    processed_data = preprocessor.parse_pdf()
    
    new_material = models.LectureMaterial(title=file.filename, raw_content=processed_data[0].content)
    db.add(new_material)
    db.commit()
    db.refresh(new_material)
    
    graph_builder = KnowledgeGraphBuilder()
    extracted_concepts = graph_builder.extract_and_save_concepts(processed_data, db, new_material.id)
        
    os.remove(file_location)
    
    return {
        "message": "자료 업로드 및 지식 그래프 생성 완료",
        "lecture_id": new_material.id,
        "extracted_concepts": extracted_concepts
    }

# ---------------------------------------------------------
# [2단계] 퀴즈 생성 (해당 자료를 바탕으로 출제)
# ---------------------------------------------------------
@app.get("/api/v1/materials/{material_id}/quiz")
async def create_quiz(material_id: int, db: Session = Depends(get_db)):
    # 1. DB에서 해당 번호표의 강의 자료 꺼내오기
    material = db.query(models.LectureMaterial).filter(
        models.LectureMaterial.id == material_id
    ).first()

    if not material:
        return {"error": "해당 강의 자료를 찾을 수 없습니다."}

    # 2. 퀴즈 생성기 가동 (내부적으로 3문제를 내도록 설정되었다고 가정)
    quiz_gen = QuizGenerator()
    quiz_content = quiz_gen.generate_quiz(material.raw_content)

    return {
        "material_id": material_id,
        "status": "퀴즈 생성 완료!",
        "quiz_data": quiz_content.split("\n")
    }

# ---------------------------------------------------------
# [3단계] 정답 해설 및 피드백 (유저의 답안을 받고 채점)
# ---------------------------------------------------------
@app.post("/api/v1/quiz/grade")
async def grade_quiz_answer(submit_data: UserAnswerSubmit):
    # 유저가 맞췄는지 틀렸는지, 보완점은 무엇인지 짚어주는 프롬프트
    evaluation_prompt = (
        "너는 친절한 대학 전공 과목 조교야. "
        "사용자가 다음 문제에 대해 답안을 제출했어.\n"
        "문제: {0}\n"
        "사용자 답안: {1}\n\n"
        "이 답안이 정답인지 오답인지 판단하고, 왜 틀렸는지 혹은 어떤 부분을 더 보완해야 하는지 상세히 설명해줘."
    ).format(submit_data.question_text, submit_data.user_answer)
    
    # 향후 LangChain이나 LLM을 연결하여 evaluation_prompt를 던져주는 곳
    ai_response = "AI가 분석한 정답 해설 및 보완점 피드백이 여기에 출력됩니다."
    
    return {
        "lecture_id": submit_data.lecture_id,
        "question": submit_data.question_text,
        "user_answer": submit_data.user_answer,
        "ai_feedback": ai_response
    }

# ---------------------------------------------------------
# [추가 기능] 기존 오답 패턴 분석 및 리포트 (유지)
# ---------------------------------------------------------
@app.post("/api/v1/feedback/analyze")
async def analyze_answer(submission: AnswerSubmission, db: Session = Depends(get_db)):
    analyzer = AnswerAnalyzer()
    error_pattern = analyzer.analyze_error(submission.user_answer, submission.question_context)
    
    weak_node = db.query(models.KnowledgeNode).filter(
        models.KnowledgeNode.material_id == submission.material_id
    ).first()
    
    if not weak_node:
        return {"error": "해당 강의의 지식 그래프가 없습니다."}
    
    new_record = models.FeedbackLoop(
        member_id=submission.user_id,
        weak_node_id=weak_node.id,
        error_type=error_pattern,
        user_answer=submission.user_answer,
        customized_feedback="분석 진행 중..."
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    
    return {
        "status": "분석 완료",
        "record_id": new_record.id,
        "detected_error_type": error_pattern.value,
        "weak_point_concept": weak_node.concept_name
    }

@app.get("/api/v1/reports/{user_id}", response_model=List[FeedbackResponse])
async def get_comprehensive_report(user_id: int, db: Session = Depends(get_db)):
    records = db.query(models.FeedbackLoop).filter(
        models.FeedbackLoop.member_id == user_id
    ).all()
    
    if not records:
        return []
        
    reports = []
    for record in records:
        weak_node = db.query(models.KnowledgeNode).filter(
            models.KnowledgeNode.id == record.weak_node_id
        ).first()
        weak_point_name = weak_node.concept_name if weak_node else "알 수 없는 개념"
        
        mock_suggested_content = "현재 '" + weak_point_name + "' 개념의 이해도가 낮습니다. 강의 PDF의 해당 섹션을 다시 참고하여 복습해 보세요."
        
        reports.append({
            "record_id": record.id,
            "error_type": record.error_type.value,
            "weak_point": weak_point_name,
            "suggested_content": mock_suggested_content,
            "status": "COMPLETED"
        })
        
    return reports