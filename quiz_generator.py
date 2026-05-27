import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class QuizGenerator:
    def __init__(self):
        # 우리가 드디어 성공시킨 바로 그 최신 모델! (퀴즈니까 창의력을 위해 temperature를 0.7로 살짝 올림)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    def generate_quiz(self, text_content: str):
        # 1. 너무 길면 구글이 힘들어하니까 앞부분 15000자만 잘라서 줌
        sample_text = text_content[:15000]
        
        # 2. 범용 퀴즈 생성 프롬프트 (어떤 과목이든 완벽 대응, 100% JSON 강제)
        template_str = """
        너는 대학생을 위한 AI 학습 조교야. 다음 대학교 강의 자료를 바탕으로, 배운 '핵심 전공 지식'을 깊이 있게 점검할 수 있는 [주관식 단답/서술형 퀴즈 딱 1문제]만 출제해.

        [절대 주의사항]
        1. 인사말, 부연 설명, 앞뒤의 마크다운 기호(```json 등)는 절대 출력하지 마. 
        2. 무조건 아래의 순수 JSON 딕셔너리 형식으로만 출력해. 형식을 어기면 서버가 붕괴되니 반드시 지켜.
        3. 객관식(A,B,C,D)은 절대 내지 마. 사용자가 직접 텍스트로 타이핑해서 논리를 설명할 수 있는 주관식/서술형 문제를 내야 해.
        4. 저작권 경고문, 출판사 출처, 교수명, 목차 등은 철저히 무시하고 오직 '학문적 핵심 개념'에 대해서만 출제해.

        [출력 양식 (반드시 이 JSON 구조를 그대로 복사해서 내용만 채울 것)]
        {{
            "question": "출제할 주관식/서술형 문제 내용 (예: 이산수학에서 그래프의 차수(Degree)란 무엇인지 설명하시오.)",
            "correct_answer": "해당 문제의 완벽한 모범 정답",
            "required_keywords": ["정답에 반드시 들어가야 할 핵심 단어1", "핵심 단어2", "핵심 단어3"],
            "explanation": "이 문제와 관련된 지식 그래프의 노드(개념) 설명 및 정답 해설"
        }}

        강의내용: {text}
        """
        
        prompt = PromptTemplate.from_template(template_str)
        chain = prompt | self.llm
        
        # 3. AI 퀴즈 생성!
        response = chain.invoke({"text": sample_text})
        
        return response.content