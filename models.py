from sqlalchemy import Column, Integer, String, ForeignKey, Enum, Text, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from database import Base

# 오답 패턴 분류용 Enum [cite: 162-165]
class ErrorPattern(enum.Enum):
    SIMPLE_MISTAKE = "단순 계산 실수"
    CONCEPT_UNKNOWN = "개념 미숙지"
    HIGHER_CONCEPT_MISSING = "상위 개념 부재"

# 1. 사용자 테이블 [cite: 166-170]
class Member(Base):
    __tablename__ = "members"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)

# 2. 강의 자료 테이블 [cite: 171-173]
class LectureMaterial(Base):
    __tablename__ = "lecture_materials"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    raw_content = Column(Text, nullable=True) # 비정형 강의 데이터
    created_at = Column(DateTime, default=datetime.utcnow)

# 3. 지식 그래프 노드 테이블 [cite: 174-189]
class KnowledgeNode(Base):
    __tablename__ = "knowledge_graph"
    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("lecture_materials.id", ondelete="CASCADE"))
    concept_name = Column(String(100), nullable=False) # 핵심 학습 개념
    
    # 개념 간 선후 관계 보장을 위한 자기 참조 (Self-Reference)
    parent_id = Column(Integer, ForeignKey("knowledge_graph.id"), nullable=True)
    
    material = relationship("LectureMaterial")
    children = relationship("KnowledgeNode")

# 4. 학습 기록 및 피드백 루프 테이블 [cite: 190-203]
class FeedbackLoop(Base):
    __tablename__ = "feedback_loops"
    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False)
    
    # 추론된 취약 지점 노드 연결
    weak_node_id = Column(Integer, ForeignKey("knowledge_graph.id"), nullable=False)
    
    error_type = Column(Enum(ErrorPattern), nullable=False) # 오답 패턴
    user_answer = Column(Text, nullable=False)
    customized_feedback = Column(Text) # RAG 연관 최적 강의 구간 피드백
    created_at = Column(DateTime, default=datetime.utcnow)