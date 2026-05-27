import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from sqlalchemy.orm import Session
from models import KnowledgeNode, LectureMaterial
from google.api_core import client_options as client_options_lib

class KnowledgeGraphBuilder:
    def __init__(self):
        # API 경로를 강제로 고정하여 v1beta 에러를 방지합니다.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            client_options=client_options_lib.ClientOptions(api_endpoint="generativelanguage.googleapis.com")
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