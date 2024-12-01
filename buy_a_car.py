import numpy as np
import pandas as pd
from openai import OpenAI
import os

# client is an instance of a library or service
client = OpenAI()
# model: Literal["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding
    #The create method requires input as a list and model and I am returning the embedding(numerical vector value) as a list

def suggestion_to_buy_a_car(question):
    question_embedding = get_embedding(question)

    # The dot product measures the similarity between two vectors in high-dimensional space.
    # Larger values indicate higher similarity.
    # page_embedding = an embedding from the DataFrame
    # def fn(page_embedding):
    #     return np.dot(page_embedding, question_embedding)

    distance_score = df['embeddings'].apply(lambda page_embedding: np.dot(page_embedding, question_embedding))  # based on distance score
    print(distance_score)
    # distance_score = df['embeddings'].apply(fn)#based on distance score

    top_five = distance_score.sort_values(ascending=False).index[:5]

    text_series = df.loc[top_five]['text']
    context = "\n\n".join(text_series)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant who is helping based on the German or Japanese make and model. Please help in better answer questions as poem."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Use this input file as context to answer the question: {context}. Please stick to this context to answering the question"}
        ]
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    # The below code needs to run once to avoid embedding costs
    df = pd.read_csv('./ExternalData/CarListing_2023Septemberdata_1.csv')
    df['embeddings'] = df['text'].apply(get_embedding)
    df.to_csv('./CarListing_2023Septemberdata_1.csv', index=False)
    # Due to fast processing I am converting the .csv to .pkl file and the pkl format is binary representation
    # and it avoids the overhead of parsing and type inference
    # The pkl file preserves the exact data types and structure including index, timestamps and custom python objects or nested data structures
    # where as in csv, numeric columns may lose precision and timestamps converted as Strings and custom objects cant store in CSV.
    # Also pkl files can be serialized and it is more compact because of binary representation
    df.to_pickle('./CarListing_2023Septemberdata_1.pkl')
    df = pd.read_pickle('./CarListing_2023Septemberdata_1.pkl')
    print(suggestion_to_buy_a_car("I want to buy a car in December. Help me choose one under $30,000."))

# Output:
# ===============
# C:\Users\ACHAND27\AppData\Local\Programs\Python\Python310\python.exe C:\PythonMicroService\RAGImplementation\buy_a_car.py
# Based on the provided context, here are some options for cars under $30,000 that you may consider:
#
# 1. 2019 Toyota Vios 1.5 (A) E Spec or G Spec: These models are priced between RM86,900 and RM93,000, making them affordable choices for your budget. They offer comfort, ample space, and advanced safety features.

# 2. 2020 Toyota Alphard 2.5 X: Even though this luxury model is priced at RM212,000, you may negotiate for a lower price within your budget. It provides a spacious interior, advanced safety features like pre-crash and lane-keeping assist, and a luxurious driving experience.

# Remember to consider factors such as your preferences in terms of space, features, and budget when making your final decision.

# Process finished with exit code 0


