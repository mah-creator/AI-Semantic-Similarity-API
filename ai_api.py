from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

class RequiredSkill(BaseModel):
    name: str
    weight: int


class Payload(BaseModel):
    list_A: List[RequiredSkill]
    list_B: List[str]
    threshold: float

@app.post("/")
async def root(payload: Payload):
    return compare_lists(payload.list_A, payload.list_B, payload.threshold)


class PhraseItem(BaseModel):
    odata_type: str = Field(..., alias='@odata.type')
    phrase: str

class PhraseList(BaseModel):
    phrases: List[PhraseItem]

@app.post("/phrase_names")
async def phrase_names(phrases: PhraseList):
    phrase_names = [item.phrase for item in phrases.phrases]
    return phrase_names

def compare_lists(list_A, list_B, threshold: float):
    list_A_names = [skill.name for skill in list_A]
    skill_score = 0
    matches = []
    # Encode both lists
    embeddings_A = model.encode(list_A_names, convert_to_tensor=True)
    embeddings_B = model.encode(list_B, convert_to_tensor=True)

    # Find highest match for each item in A
    for idx, emb_A in enumerate(embeddings_A):
        # Compute cosine similarity with all B
        cosine_scores = util.cos_sim(emb_A, embeddings_B)

        # Get the best match
        best_match_idx = cosine_scores.argmax()
        best_score = cosine_scores[0][best_match_idx]

        if(best_score >= threshold):
            matches.append(list_A_names[idx])
            skill_score += list_A[idx].weight

    return {"matches": matches, "matchScore": skill_score}
