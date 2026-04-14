# app/schemas/analyze_schema.py
from pydantic import BaseModel

class AnalyzeFromS3Request(BaseModel):
    video_url: str
    s3_key: str
    presentation_type: str

    topic_summary: str
    topic_desc: str
    topic_tags: list

