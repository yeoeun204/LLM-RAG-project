from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
import math
from fastapi.middleware.cors import CORSMiddleware

import models
from database import engine, SessionLocal
from preprocessor import DataPreprocessor
from graph_builder import KnowledgeGraphBuilder
from analyzer import AnswerAnalyzer
from quiz_generator import QuizGenerator

models.Base.metadata.create_all(bind=engine)

# 1. 앱 생성 및 CORS 허용 (순서 완벽 고정!)
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
# [데이터 전송 규격 (그릇)]
# ---------------------------------------------------------
class UserAnswerSubmit(BaseModel):
    lecture_id: int
    question_text: str
    user_answer: str
    time_taken_seconds: float = 60.0  # (추가됨) 문제 푸는 데 걸린 시간! 기본값 60초

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
    # 1. 기본 설정값 (팀원 기획서 기준)
    w1 = 0.3  # 지식구조 가중치 비율
    w2 = 0.7  # 동적 반응 가중치 비율
    w_kg = 0.5  # 지식 그래프 기본 점수 (임시 평균값)
    v_i = 0.0   # 역량 노드 가산점
    t_rec = 60.0 # 권장 문제 풀이 시간 (초)
    
    # 정답을 맞췄다면 위험도(오답률 E_i)는 0, 틀렸다면 1
    e_i = 0.0 if is_correct else 1.0
    
    # 2. S_i: 단순 실수(Slip) vs 근본적 오해(Mistake) 필터
    s_i = 1.0
    if error_type == "SLIP":
        s_i = 0.5  # 단순 실수면 가중치 감쇄
    elif error_type == "MISTAKE":
        s_i = 1.0  # 근본적 오해면 가중치 유지
        
    # 3. T_i: 인지 과부하 지수 (시그모이드 함수 적용)
    # 수식: 1 / (1 + e^(-(t_taken / t_rec - 1)))
    try:
        power = -(time_taken / t_rec - 1)
        t_i = 1.0 / (1.0 + math.exp(power))
    except OverflowError:
        t_i = 1.0 if time_taken > t_rec else 0.0
        
    # 4. W_final: 최종 심각도 가중치 계산
    w_final = w1 * (w_kg + v_i) + w2 * ((e_i * s_i) * t_i)
    
    # 소수점 둘째 자리까지 반올림하여 반환
    return round(w_final, 2)

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
    material = db.query(models.LectureMaterial).filter(
        models.LectureMaterial.id == material_id
    ).first()

    if not material:
        return {"error": "해당 강의 자료를 찾을 수 없습니다."}

    quiz_gen = QuizGenerator()
    quiz_content = quiz_gen.generate_quiz(material.raw_content)

    return {
        "material_id": material_id,
        "status": "퀴즈 생성 완료!",
        "quiz_data": quiz_content.split("\n")
    }

# ---------------------------------------------------------
# [3단계] 정답 해설 및 피드백 (팀원의 가중치 엔진 적용!)
# ---------------------------------------------------------
@app.post("/api/v1/quiz/grade")
async def grade_quiz_answer(submit_data: UserAnswerSubmit):
    # 1. AI 조교에게 채점과 동시에 '실수'인지 '오해'인지 분류를 지시하는 프롬프트
    evaluation_prompt = (
        "너는 친절한 대학 전공 과목 조교야. 사용자가 문제에 대한 답을 제출했어.\n"
        "문제: {0}\n"
        "사용자 답안: {1}\n\n"
        "이 답안이 정답인지 오답인지 상세히 해설해줘.\n"
        "단, 마지막 줄에는 반드시 다음 중 하나를 판별해서 텍스트로 적어줘:\n"
        "- 단순 계산 실수나 기호 혼동(Slip)으로 틀렸다면: [SLIP]\n"
        "- 근본적인 개념을 몰라서(Mistake) 틀렸다면: [MISTAKE]\n"
        "- 정답을 맞췄다면: [CORRECT]"
    ).format(submit_data.question_text, submit_data.user_answer)
    
    # 임시 AI 응답 (나중에 실제 랭체인 연결 시 대체될 부분)
    # 테스트를 위해 임의로 MISTAKE 상황을 가정해두었어!
    mock_ai_response = "사용자님, 공식은 맞게 접근하셨으나 핵심 개념의 적용이 잘못되었습니다.\n[MISTAKE]"
    
    # 2. AI 응답에서 에러 타입(Slip vs Mistake) 추출하기
    error_type = "UNKNOWN"
    is_correct = False
    if "[SLIP]" in mock_ai_response:
        error_type = "SLIP"
    elif "[MISTAKE]" in mock_ai_response:
        error_type = "MISTAKE"
    elif "[CORRECT]" in mock_ai_response:
        error_type = "CORRECT"
        is_correct = True
        
    # 3. 팀원의 최종 가중치 엔진 가동!
    final_weight_score = calculate_final_weight(
        is_correct=is_correct,
        error_type=error_type,
        time_taken=submit_data.time_taken_seconds
    )
    
    # 4. 점수에 따른 조건부 의사결정 (제어 논리)
    decision_message = ""
    if is_correct:
        decision_message = "훌륭합니다! 개념을 완벽히 이해하셨네요."
    elif final_weight_score >= 0.5:
        decision_message = "위험도 점수: " + str(final_weight_score) + "점. 심각한 인지 마비 상태입니다. 해당 챕터의 선행 개념부터 다시 학습할 것을 권장합니다."
    else:
        decision_message = "위험도 점수: " + str(final_weight_score) + "점. 단순 실수로 보입니다. 오답 노트만 확인하고 다음 단계로 넘어가도 좋습니다."

    return {
        "lecture_id": submit_data.lecture_id,
        "question": submit_data.question_text,
        "user_answer": submit_data.user_answer,
        "error_type_detected": error_type,
        "final_danger_score": final_weight_score,
        "ai_feedback": mock_ai_response,
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
        
        mock_suggested_content = "현재 '" + weak_point_name + "' 개념의 이해도가 낮습니다. 강의 PDF의 해당 섹션을 다시 참고하여 복습해 보세요."
        
        reports.append({
            "record_id": record.id,
            "error_type": record.error_type.value,
            "weak_point": weak_point_name,
            "suggested_content": mock_suggested_content,
            "status": "COMPLETED"
        })
        
    return reports