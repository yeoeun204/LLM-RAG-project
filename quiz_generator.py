import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class QuizGenerator:
    def __init__(self):
        # 우리가 드디어 성공시킨 바로 그 최신 모델! (퀴즈니까 창의력을 위해 temperature를 0.7로 살짝 올림)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    def generate_quiz(self, text_content: str):
        # 1. 너무 길면 구글이 힘들어하니까 앞부분 15000자만 잘라서 줌
        sample_text = text_content[:15000]
        
        # 2. 범용 퀴즈 생성 프롬프트 (어떤 과목이든 완벽 대응)
        template_str = """
        다음 대학교 강의 자료를 바탕으로, 대학생 1학년 수준에서 배운 '핵심 전공 지식'을 점검할 수 있는 객관식 퀴즈 3문제를 만들어주세요.

        [절대 주의사항]
        1. 문서의 1~2페이지에 주로 등장하는 저작권 경고문, 출판사 출처, 교수님 성함, 목차 등 학문적 내용과 무관한 부분은 철저히 무시하세요. 반드시 '수업 핵심 개념'에 대해서만 출제해야 합니다.
        2. 프론트엔드 시스템이 문제를 분리할 수 있도록, 문제 번호(Q1, Q2, Q3)에 마크다운 기호(** 등)를 절대 사용하지 마세요.
        3. 반드시 아래의 [출력 양식]을 토씨 하나 틀리지 않고 엄격하게 지켜주세요.

        [출력 양식]
        Q1. (문제 내용)
        - A) (보기 1)
        - B) (보기 2)
        - C) (보기 3)
        - D) (보기 4)
        👉 정답: (A, B, C, D 중 하나)
        💡 해설: (왜 이 정답인지 1~2줄로 설명)

        Q2. (문제 내용)
        - A) (보기 1)
        - B) (보기 2)
        - C) (보기 3)
        - D) (보기 4)
        👉 정답: (A, B, C, D 중 하나)
        💡 해설: (왜 이 정답인지 1~2줄로 설명)

        Q3. (문제 내용)
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