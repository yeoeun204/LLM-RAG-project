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
        return {"status": "MASTERED"}
    elif final_score >= 0.85 and final_score < 0.90:
        return {"status": "WARNING"}
    else:
        return {"status": "ERROR"}

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

class UserAnswerSubmit(BaseModel):
    lecture_id: int
    question_text: str
    user_answer: str
    time_taken_seconds: float = 60.0 
    correct_answer: str = ""           
    required_keywords: List[str] = []  
    explanation: str = ""              

def calculate_final_weight(is_correct: bool, error_type: str, time_taken: float) -> float:
    w1 = 0.3 
    w2 = 0.7 
    w_kg = 0.5 
    v_i = 0.0 
    t_rec = 60.0 
    e_i = 0.0 if is_correct else 1.0
    s_i = 1.0
    if error_type == "SLIP": s_i = 0.5 
    elif error_type == "MISTAKE": s_i = 1.0 
    try:
        power = -(time_taken / t_rec - 1)
        t_i = 1.0 / (1.0 + math.exp(power))
    except OverflowError:
        t_i = 1.0 if time_taken > t_rec else 0.0
    w_final = w1 * (w_kg + v_i) + w2 * ((e_i * s_i) * t_i)
    return round(w_final, 2)

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
        "message": "자료 업로드 완료",
        "lecture_id": new_material.id,
        "extracted_concepts": extracted_concepts
    }

# =========================================================
# 👇 바로 이 부분이 완벽하게 수정된 '퀴즈 생성 + 좀비 파서' 코드입니다! 
# =========================================================
@app.get("/api/v1/materials/{material_id}/quiz")
async def create_quiz(material_id: int, db: Session = Depends(get_db)):
    material = db.query(models.LectureMaterial).filter(models.LectureMaterial.id == material_id).first()
    if not material: return {"error": "강의 자료 없음"}
    
    import re # 데이터 강제 추출을 위한 정규표현식 라이브러리 추가
    
    quiz_gen = QuizGenerator()
    raw_quiz_content = quiz_gen.generate_quiz(material.raw_content)
    
    # 1. AI가 눈치 없이 붙인 마크다운 껍데기 벗기기
    cleaned_content = raw_quiz_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[3:]
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]
    cleaned_content = cleaned_content.strip()

    try:
        # 1차 시도: 깔끔하게 JSON으로 변환해보기
        quiz_data = json.loads(cleaned_content)
    except json.JSONDecodeError:
        # 2차 시도: AI가 형식을 엉망으로 보냈어도 멱살 잡고 핵심 데이터만 강제로 뜯어냄!
        try:
            q_match = re.search(r'"question"\s*:\s*"([^"]+)"', raw_quiz_content)
            a_match = re.search(r'"correct_answer"\s*:\s*"([^"]+)"', raw_quiz_content)
            k_match = re.search(r'"required_keywords"\s*:\s*\[(.*?)\]', raw_quiz_content, re.DOTALL)
            e_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', raw_quiz_content)

            question = q_match.group(1) if q_match else "문제를 생성했으나 형식이 깨졌습니다. AI의 답변: " + raw_quiz_content[:50]
            answer = a_match.group(1) if a_match else "정답 데이터 추출 실패"
            explanation = e_match.group(1) if e_match else "해설 추출 실패"

            keywords = []
            if k_match:
                keywords = re.findall(r'"([^"]+)"', k_match.group(1))

            quiz_data = {
                "question": question,
                "correct_answer": answer,
                "required_keywords": keywords,
                "explanation": explanation
            }
        except Exception:
            # 3차 최후의 수단: 무슨 짓을 해도 실패하면 AI가 한 말을 통째로 문제로 띄워버림
            quiz_data = {
                "question": raw_quiz_content,
                "correct_answer": "AI 답변 형식 오류로 자동 채점 불가",
                "required_keywords": [],
                "explanation": "해석 불가"
            }

    return {"material_id": material_id, "status": "퀴즈 생성 완료!", "quiz_data": quiz_data}
# =========================================================

@app.post("/api/v1/quiz/grade")
async def grade_quiz_answer(submit_data: UserAnswerSubmit):
    user_ans = submit_data.user_answer
    ideal_ans = submit_data.correct_answer 
    keywords = submit_data.required_keywords
    
    # 💡 임시 코사인 유사도 0.88 부여 (나중에 LLM 임베딩 연결 시 이 부분 수정)
    cosine_sim = 0.88 
    
    final_score = calculate_text_similarity(cosine_sim, user_ans, keywords)
    state_info = classify_knowledge_state(final_score)
    state = state_info["status"]

    is_correct = False
    error_type = "UNKNOWN"
    
    if state == "MASTERED":
        is_correct = True
        error_type = "CORRECT"
    elif state == "WARNING":
        error_type = "SLIP"
    else:
        error_type = "MISTAKE"

    final_weight_score = calculate_final_weight(
        is_correct=is_correct,
        error_type=error_type,
        time_taken=submit_data.time_taken_seconds
    )
    
    if is_correct:
        decision_message = "훌륭합니다! 개념을 완벽히 이해하셨네요."
        ai_feedback = "🎉 완벽하게 핵심을 짚었습니다!"
    elif error_type == "SLIP":
        decision_message = "위험도 점수: {}점. 단순 실수로 보입니다.".format(final_weight_score)
        ai_feedback = "🤖 [Agent B 분석] 방향성은 맞지만, 키워드 '{}'에 대한 언급이 부족하거나 오타가 있습니다.".format(", ".join(keywords))
    else:
        decision_message = "위험도 점수: {}점. 선행 개념 복습을 권장합니다.".format(final_weight_score)
        if cosine_sim < 0.7:
             ai_feedback = "🤖 [우회 모드] 질문의 의도와 많이 다릅니다. 이 문제는 '{}' 개념을 묻고 있습니다.".format(submit_data.explanation)
        else:
             ai_feedback = "🤖 [Agent A 분석] 핵심 개념에 대한 근본적인 오해가 있습니다. 지식 그래프를 다시 확인하세요."

    return {
        "lecture_id": submit_data.lecture_id,
        "question": submit_data.question_text,
        "user_answer": submit_data.user_answer,
        "error_type_detected": error_type,
        "final_danger_score": final_weight_score,
        "ai_feedback": ai_feedback,
        "system_decision": decision_message
    }