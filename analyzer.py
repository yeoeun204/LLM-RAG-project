from models import ErrorPattern

class AnswerAnalyzer:
    def __init__(self):
        # 나중에 여기에 진짜 코사인 유사도 계산 모델(KoNLPy 등)이 들어갈 자리입니다.
        pass
        
    def analyze_error(self, user_answer: str, question_context: str) -> ErrorPattern:
        """
        사용자의 답변을 분석하여 오답 패턴을 반환합니다. [cite: 262-271]
        (현재는 테스트를 위한 조건문 모킹 상태)
        """
        answer_lower = user_answer.lower()
        
        # 1. 단순 계산 실수 판별 (예: 숫자나 단위 오타)
        if "0.385" in answer_lower or "오타" in answer_lower:
            return ErrorPattern.SIMPLE_MISTAKE
            
        # 2. 개념 미숙지 판별 (핵심 키워드를 잘못 사용한 경우)
        elif "잘 모르겠" in answer_lower or "헷갈" in answer_lower:
            return ErrorPattern.CONCEPT_UNKNOWN
            
        # 3. 상위 개념 부재 (아예 맥락을 파악하지 못한 경우)
        else:
            return ErrorPattern.HIGHER_CONCEPT_MISSING