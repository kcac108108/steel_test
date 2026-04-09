# Steel Test - 철강 수입신고 자동 분류 시스템

## 개요
수입신고 모델규격 텍스트에서 강종과 사이즈를 자동으로 분류하는 시스템

## 분류 방식 (3단계)
1. **룰베이스 매칭**: 기존 1만여개 규칙 DB 활용
2. **RAG 유사도 검색**: 기존 분류 데이터 벡터 검색
3. **LLM 판단**: 저신뢰도 데이터 AI 분류

## 프로젝트 구조
```
steel_test/
├── app/
│   ├── api/          # API 엔드포인트
│   ├── core/         # 핵심 설정
│   ├── services/     # 분류 서비스 로직
│   └── models/       # 데이터 모델
├── data/
│   ├── raw/          # 원본 엑셀 파일
│   ├── processed/    # 전처리된 데이터
│   └── rules/        # 룰베이스 DB
├── chroma_db/        # 벡터 DB
├── scripts/          # 데이터 처리 스크립트
├── static/           # 정적 파일
├── templates/        # HTML 템플릿
└── tests/            # 테스트
```
