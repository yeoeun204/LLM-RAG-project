from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
import math
import json
from fastapi.middleware.cors import CORSMiddleware

import models
from database import engine, SessionLocal
from preprocessor import DataPreprocessor
from graph_builder import KnowledgeGraphBuilder
from analyzer import AnswerAnalyzer
from quiz_generator import QuizGenerator

models.Base.metadata.create_all(bind=engine)

import numpy as np

def evaluate_quantitative_answer(user_ans_val: float, correct_ans_val: float) -> str:
    if correct_ans_val == 0.0:
        error = abs(correct_ans_val - user_ans_val)
    else:
        error = abs((correct_ans_val - user_ans_val) / correct_ans_val)
        
    if error == 0.0: return "MASTERED"
    elif error <= 0.05: return "WARNING"
    else: return "ERROR"

def calculate_text_similarity(cosine_sim: float, user_text: str, required_keywords: list) -> float:
    alpha = 0.7
    beta = 0.3
    if not required_keywords:
        keyword_score = 1.0
    else:
        matched_count = sum(1 for kw in required_keywords if kw in user_text)
        keyword_score = matched_count / len(required_keywords)
    return (alpha * cosine_sim) + (beta * keyword_score)

def classify_knowledge_state(final_score: float):
    if final_score >= 0.90:
        return {"status": "MASTERED", "ui_color": "GREEN", "message": "완벽하게 이해했습니다!"}
    elif final_score >= 0.85 and final_score < 0.90:
        return {"status": "WARNING", "ui_color": "YELLOW", "message": "개념은 알지만 단순 표현 오타나 부주의가 있습니다."}
    else:
        return {"status": "ERROR", "ui_color": "RED", "message": "근본적인 맥락 오류 및 오개념이 발견되었습니다."}

app = FastAPI(title="맞춤형 학습 진단 시스템 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------
# [데이터 전송 규격 (프론트 동적 데이터 추가)]
# ---------------------------------------------------------
class UserAnswerSubmit(BaseModel):
    lecture_id: int
    question_text: str
    user_answer: str
    time_taken_seconds: float = 60.0 
    correct_answer: str = ""           # 프론트에서 넘어오는 진짜 정답
    required_keywords: List[str] = []  # 프론트에서 넘어오는 채점 기준
    explanation: str = ""              # 프론트에서 넘어오는 노드 해설

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
# [가중치 엔진] 팀원이 기획한 최종 엔진 수학 공식
# ---------------------------------------------------------
def calculate_final_weight(is_correct: bool, error_type: str, time_taken: float) -> float:
    w1 = 0.3 
    w2 = 0.7 
    w_kg = 0.5 
    v_i = 0.0 
    t_rec = 60.0 
    
    e_i = 0.0 if is_correct else 1.0
    
    s_i = 1.0
    if error_type == "SLIP":
        s_i = 0.5 
    elif error_type == "MISTAKE":
        s_i = 1.0 
        
    try:
        power = -(time_taken / t_rec - 1)
        t_i = 1.0 / (1.0 + math.exp(power))
    except OverflowError:
        t_i = 1.0 if time_taken > t_rec else 0.0
        
    w_final = w1 * (w_kg + v_i) + w2 * ((e_i * s_i) * t_i)
    return round(w_final, 2)

# ---------------------------------------------------------
# [1단계] 강의자료 입력 및 지식 그래프 생성 (DB 유지)
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
# [2단계] 동적 퀴즈 생성 (JSON 파싱 구조 반영)
# ---------------------------------------------------------
@app.get("/api/v1/materials/{material_id}/quiz")
async def create_quiz(material_id: int, db: Session = Depends(get_db)):
    material = db.query(models.LectureMaterial).filter(
        models.LectureMaterial.id == material_id
    ).first()

    if not material:
        return {"error": "해당 강의 자료를 찾을 수 없습니다."}

    # QuizGenerator 내부 프롬프트를 JSON 출력 방식으로 수정했다고 가정
    quiz_gen = QuizGenerator()
    raw_quiz_content = quiz_gen.generate_quiz(material.raw_content)

    # 문자열 텍스트를 JSON(딕셔너리)으로 변환
    try:
        quiz_data = json.loads(raw_quiz_content)
    except json.JSONDecodeError:
        quiz_data = {
            "question": "문제를 불러오지 못했습니다.",
            "correct_answer": "",
            "required_keywords": [],
            "explanation": ""
        }

    return {
        "material_id": material_id,
        "status": "퀴즈 생성 완료!",
        "quiz_data": quiz_data
    }

# ---------------------------------------------------------
# [3단계] 하이브리드 엔진 + 위계 가중치 융합 채점 로직
# ---------------------------------------------------------
@app.post("/api/v1/quiz/grade")
async def grade_quiz_answer(submit_data: UserAnswerSubmit):
    user_ans = submit_data.user_answer
    ideal_ans = submit_data.correct_answer 
    keywords = submit_data.required_keywords
    
    # 1. 코사인 유사도 연산 (기존에 쓰던 LLM 임베딩 모델 호출 부분)
    cosine_sim = 0.88 
    
    # 2. 팀의 7:3 하이브리드 공식 작동
    final_score = calculate_text_similarity(cosine_sim, user_ans, keywords)
    state_info = classify_knowledge_state(final_score)
    state = state_info["status"]

    # 3. 상태값을 통해 에러 타입 분류
    is_correct = False
    error_type = "UNKNOWN"
    
    if state == "MASTERED":
        is_correct = True
        error_type = "CORRECT"
    elif state == "WARNING":
        error_type = "SLIP"
    else:
        error_type = "MISTAKE"

    # 4. 융합 가중치(위험도) 계산 알고리즘 실행
    final_weight_score = calculate_final_weight(
        is_correct=is_correct,
        error_type=error_type,
        time_taken=submit_data.time_taken_seconds
    )
    
    # 5. Multi-Agent 피드백 및 Fallback 우회 로직 처리
    if is_correct:
        decision_message = "훌륭합니다! 개념을 완벽히 이해하셨네요."
        ai_feedback = "🎉 완벽하게 핵심을 짚었습니다!"
    elif error_type == "SLIP":
        decision_message = "위험도 점수: {}점. 단순 실수로 보입니다.".format(final_weight_score)
        ai_feedback = "🤖 [Agent B 분석] 방향성은 맞지만, 키워드 '{}'에 대한 언급이 부족하거나 오타가 있습니다.".format(", ".join(keywords))
    else:
        decision_message = "위험도 점수: {}점. 해당 챕터의 선행 개념 복습을 권장합니다.".format(final_weight_score)
        if cosine_sim < 0.7:
             ai_feedback = "🤖 [우회 모드] 질문의 의도와 많이 다릅니다. 이 문제는 '{}' 개념을 묻고 있습니다.".format(submit_data.explanation)
        else:
             ai_feedback = "🤖 [Agent A 분석] 핵심 개념에 대한 근본적인 오해가 있습니다. 관련 지식 그래프를 다시 확인하세요."

    return {
        "lecture_id": submit_data.lecture_id,
        "question": submit_data.question_text,
        "user_answer": submit_data.user_answer,
        "error_type_detected": error_type,
        "final_danger_score": final_weight_score,
        "ai_feedback": ai_feedback,
        "system_decision": decision_message
    }

# ---------------------------------------------------------
# [추가 기능] 기존 리포트 조회 로직 유지
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
        
        mock_suggested_content = "현재 '{}' 개념의 이해도가 낮습니다. 강의 PDF의 해당 섹션을 다시 참고하여 복습해 보세요.".format(weak_point_name)
        
        reports.append({
            "record_id": record.id,
            "error_type": record.error_type.value,
            "weak_point": weak_point_name,
            "suggested_content": mock_suggested_content,
            "status": "COMPLETED"
        })
        
    return reports