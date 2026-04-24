from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Link(BaseModel):
    url: str
    type: Literal["hypertext", "youtube"]

class Option(BaseModel):
    text: str
    isCorrect: bool

class Question(BaseModel):
    questionText: str = Field(description="Markdown formatted question")
    type: Literal["MultiChoice", "ShortAnswer", "Essay"]
    isJustification: bool = False
    grade: int = 1
    links: List[Link] = []
    options: List[Option] = []
    sampleAnswer: Optional[str] = Field(None, description="The 'perfect' answer for Essays/ShortAnswer")
    context: str = Field(description="Markdown formatted context/explanation")

class QuizDetails(BaseModel):
    questions: List[Question]

class FullQuizSchema(BaseModel):
    title: str
    description: str = Field(description="Markdown formatted description")
    state: Literal["active", "draft"]
    subject: str
    type: Literal["Quiz", "Essay"]
    quiz: QuizDetails