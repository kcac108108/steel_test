# Steel Test - 철강 수입신고 자동 분류 시스템

## 개요
수입신고 모델규격(영문 자유형식 텍스트)에서 강종과 사이즈를 자동으로 분류하는 시스템

## 프로젝트 배경
- 매일 수입된 철강제품 자료를 엑셀로 변환 후 사람이 수동으로 강종/사이즈 분류 중
- 10년치 월별 분류 데이터 보유
- 룰베이스 DB 1만여개 (회사 내부 Oracle DB)
- 규격품명은 전부 영문 자유형식 텍스트 (신고인이 임의로 작성)

## 분류 방식 (4단계)
1. **룰베이스 매칭**: 기존 1만여개 규칙 DB 활용 → 즉시 분류
2. **RAG 유사도 검색**: 10년치 분류 데이터 벡터 검색 → 유사도 높으면 분류
3. **LLM 판단**: OpenAI GPT → 어느 정도 근거 있을 때만 분류
4. **미분류**: LLM도 판단 불가 시 → 건드리지 않고 미분류 표시

## 기술 스택
- **언어**: Python 3.11
- **AI API**: OpenAI (임베딩: text-embedding-3-small, LLM: GPT)
- **벡터DB**: ChromaDB
- **DB**: Oracle (도커) - 룰베이스 + 분류 이력 저장
- **사용 방식**: 스크립트 실행 → 결과 엑셀 파일 저장

## 결과 저장
- 입력 엑셀에 강종/사이즈 컬럼 추가 후 저장
- 분류 이력은 Oracle DB에도 저장 (재학습 데이터 활용)

## 규격품명 예시
```
# 잘 된 케이스
CRC SUS430 2B SLIT 2.00X1524XC PS: COLD ROLLED, MT: STAINLESS STEEL, SHAPE: COILS, SIZE: T 2.0MM X 2 1524MM X L C

# 분류 어려운 케이스 (미분류 처리)
HARDWARE ACCESSORIES
DKSLAUS E17056470TUBE,COOLING
```

## 진행 상황
- [x] GitHub 연결 (https://github.com/kcac108108/steel_test)
- [x] 가상환경 생성 (venv, Python 3.11)
- [x] 라이브러리 설치 완료
- [x] 프로젝트 구조 생성
- [x] Oracle DB 연결 설정 (`app/core/database.py`)
- [x] 룰베이스 매칭 로직 구현 (`app/services/rule_matcher.py`)
- [x] RAG 유사도 검색 구현 (`app/services/rag_service.py`)
- [x] LLM 판단 로직 구현 (`app/services/llm_service.py`)
- [x] 메인 분류 스크립트 작성 (`scripts/classify.py`)
- [ ] Oracle DB 테이블 생성 및 룰베이스 데이터 입력
- [ ] 이력 데이터 RAG 인덱싱
- [ ] 테스트

## 프로젝트 구조
```
steel_test/
├── app/
│   ├── api/          # API 엔드포인트
│   ├── core/         # 핵심 설정
│   ├── services/     # 분류 서비스 로직
│   └── models/       # 데이터 모델
├── data/
│   ├── raw/          # 원본 엑셀 파일 (gitignore)
│   ├── processed/    # 결과 엑셀 파일
│   └── rules/        # 룰베이스 데이터
├── chroma_db/        # 벡터 DB
├── scripts/          # 분류 실행 스크립트
├── static/
├── templates/
├── tests/
├── venv/             # 가상환경 (Python 3.11)
├── .env              # API 키 (OPENAI_API_KEY 등)
├── .gitignore
├── requirements.txt
└── README.md
```
