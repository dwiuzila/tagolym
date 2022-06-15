from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    text: str = Query(None, min_length=1)


class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {
                        "text": "Let $n \\geqslant 100$ be an integer. Ivan writes the numbers $n, n+1, \\ldots, 2 n$ each on different cards. He then shuffles these $n+1$ cards, and divides them into two piles. Prove that at least one of the piles contains two cards such that the sum of their numbers is a perfect square."
                    },
                    {
                        "text": "Show that the inequality\\[\\sum_{i=1}^n \\sum_{j=1}^n \\sqrt{|x_i-x_j|}\\leqslant \\sum_{i=1}^n \\sum_{j=1}^n \\sqrt{|x_i+x_j|}\\]holds for all real numbers $x_1,\\ldots x_n.$"
                    },
                    {
                        "text": "Let $D$ be an interior point of the acute triangle $ABC$ with $AB > AC$ so that $\\angle DAB = \\angle CAD.$ The point $E$ on the segment $AC$ satisfies $\\angle ADE =\\angle BCD,$ the point $F$ on the segment $AB$ satisfies $\\angle FDA =\\angle DBC,$ and the point $X$ on the line $AC$ satisfies $CX = BX.$ Let $O_1$ and $O_2$ be the circumcenters of the triangles $ADC$ and $EXD,$ respectively. Prove that the lines $BC, EF,$ and $O_1O_2$ are concurrent."
                    },
                ]
            }
        }
