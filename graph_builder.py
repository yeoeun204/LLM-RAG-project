import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from sqlalchemy.orm import Session
from models import KnowledgeNode, LectureMaterial

class KnowledgeGraphBuilder:
    def __init__(self):
        # 여기도 깔끔하게 모델만 남깁니다.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0
            api_key=os.environ.get("GOOGLE_API_KEY"),
            api_version="v1"
        )

    def extract_and_save_concepts(self, text_chunks, db: Session, material_id: int):
        combined_text = " ".join([chunk.content for chunk in text_chunks[:3]])
        
        template_str = "다음 대학교 강의 자료에서 가장 중요한 핵심 개념 3가지만 쉼표로 구분해서 단어로 알려줘.\n\n강의내용: {text}"
        prompt = PromptTemplate.from_template(template_str)
        
        chain = prompt | self.llm
        response = chain.invoke({"text": combined_text})
        
        concepts = [c.strip() for c in response.content.split(",")]
        
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