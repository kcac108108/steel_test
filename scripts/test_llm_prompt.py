import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.llm_service import LLMService

samples = [
    ("METAL STAINLESS STEELROUND BARAMS5659T 0.3125\"DIA X 144\" R/L HEAT:595161", "AMS5659"),
    ("METAL STAINLESS STEELROUND BARAMS5630M 0.375\"DIA X 144\"RL HEAT:438255", "AMS5630"),
    ("ALLOY 4340, CF ANNL, AMS6415,AMS.S5000 COND C4, ASTM A1080.875\"DIA X 12 R/L", "AMS6415"),
    ("304/304L AMS5511L,AMS5513L,ASTM A240 T2.15\" X 2.9528\" X 3.5433\"", "ASTM A240 304/304L"),
    ("AMS5599-0.059TX96 0.059 X 48 X 96 SHEET, INC (INCONEL625)", "AMS5599"),
    ("9310, AMS-6265 VAR / NORMALIZED TEMPERED 5.250\"X2.56\" CUTLENGTH, MT: ALLOY STEEL", "AMS6265"),
    ("SGCC 0.5T X 1219W X C", "JIS G3302 SGCC"),
    ("SKD11 100 X 100 X 300", "JIS G4404 SKD11"),
    ("SS400 PL 9.0T X 1500W X 6000L", "JIS G3101 SS400"),
    ("MODEL:SA789(UNS531803)MT:STAINLESS STEEL,SHAPE: SEAMLESS TUBE 25.4OD X 2.11T", "ASME SA789 UNS S31803"),
    ("17-4 PH BAR,AMS 5643 H1150D0.5\" X L 20\" HEAT NO.292985", "AMS5643"),
    ("ALLOY X BAR,AMS 5754D 2.0\" X L 20\" HEAT NO.200840", "AMS5754"),
    ("CARBON STEEL TUBE 1026 DOM ASTM A513 6.25\"OD X 5.00\"ID X 10FT", "ASTM A513"),
    ("0009260190000DR EN10270-1-0,4-DH LV", "EN 10270-1"),
]

llm = LLMService()
correct = 0
print(f"{'규격(앞45자)':<45} | {'담당자':<25} | {'LLM결과':<25} | 일치")
print("-" * 110)
for spec, expected in samples:
    result = llm.classify(spec)
    got = result.steel_grade if result else "(미분류)"
    match = got.upper().replace(" ", "") == expected.upper().replace(" ", "")
    if match:
        correct += 1
    print(f"{spec[:45]:<45} | {expected:<25} | {got:<25} | {'O' if match else 'X'}")

print()
print(f"결과: {correct}/{len(samples)}건 정확 ({correct/len(samples)*100:.0f}%)")
