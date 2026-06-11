from app.core.database import db_cursor

pattern    = "JIS SPHC, JIS SS400"
steel_grade = "JIS G3101 SS400"
rule_id    = "R_SPHC_SS400"

with db_cursor() as cursor:
    cursor.execute(
        "SELECT COUNT(*) FROM rule_base WHERE rule_id = :1",
        (rule_id,)
    )
    exists = cursor.fetchone()[0]
    if exists:
        print(f"이미 존재: {rule_id}")
    else:
        cursor.execute(
            "INSERT INTO rule_base (rule_id, pattern, steel_grade, size_val) VALUES (:1, :2, :3, NULL)",
            (rule_id, pattern, steel_grade)
        )
        print(f"INSERT 완료: {pattern} → {steel_grade}")
