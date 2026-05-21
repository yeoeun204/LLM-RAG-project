import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class QuizGenerator:
    def __init__(self):
        # 우리가 드디어 성공시킨 바로 그 최신 모델! (퀴즈니까 창의력을 위해 temperature를 0.7로 살짝 올림)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    def generate_quiz(self, text_content: str):
        # 1. 너무 길면 구글이 힘들어하니까 앞부분 3000자만 잘라서 줌
        sample_text = text_content[:15000]
        
        # 2. LLM에게 퀴즈를 만들어달라고 하는 강력한 프롬프트(명령어)
        template_str = """
        다음 대학교 강의 자료를 바탕으로, 학생들이 배운 내용을 점검할 수 있는 핵심 객관식 퀴즈 3문제를 만들어주세요.
        반드시 아래의 출력 양식을 엄격하게 지켜주세요.

        [출력 양식]
        Q1. (문제 내용)
        - A) (보기 1)
        - B) (보기 2)
        - C) (보기 3)
        - D) (보기 4)
        👉 정답: (A, B, C, D 중 하나)
        💡 해설: (왜 이 정답인지 1~2줄로 설명)

        강의내용: {text}
        """
        
        prompt = PromptTemplate.from_template(template_str)
        chain = prompt | self.llm
        
        # 3. AI 퀴즈 생성!
        response = chain.invoke({"text": sample_text})
        
        return response.content