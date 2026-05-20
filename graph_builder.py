import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from sqlalchemy.orm import Session
from models import KnowledgeNode, LectureMaterial


class KnowledgeGraphBuilder:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def extract_and_save_concepts(self, text_chunks, db: Session, material_id: int):
        # 1. 텍스트 합치기 (비용과 속도를 위해 첫 3페이지만 샘플로 분석)
        combined_text = " ".join([chunk.content for chunk in text_chunks[:3]])
        
        # 2. LLM에게 지시할 프롬프트(명령어) 설정
        template_str = "다음 대학교 강의 자료에서 가장 중요한 핵심 개념 3가지만 쉼표로 구분해서 단어로 알려줘.\n\n강의내용: {text}"
        prompt = PromptTemplate.from_template(template_str)
        
        # 3. 인공지능 호출 및 답변 받아오기
        chain = prompt | self.llm
        response = chain.invoke({"text": combined_text})
        
        # 4. 답변 결과(문자열)를 쉼표 기준으로 쪼개서 리스트로 만들기
        concepts = [c.strip() for c in response.content.split(",")]
        
        # 5. DB에 추출된 개념 저장 (부모-자식 관계로 연결)
        parent_node = None
        for concept in concepts:
            new_node = KnowledgeNode(
                material_id=material_id,
                concept_name=concept,
                parent_id=parent_node.id if parent_node else None 
            )
            db.add(new_node)
            db.commit()
            db.refresh(new_node)
            parent_node = new_node
            
        return concepts