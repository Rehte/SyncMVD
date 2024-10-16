import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OPENAI_API_KEY from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=openai_api_key)

def get_styles(prompt):
    outside_style = prompt
    inside_style = prompt
    
    system_message = """
    You are a helpful assistant for parsing the prompt which is designed to specify the style of a 3d mesh, and distinguish and separate the outside style from the inside style if specified.
    """
    request_message = f"""
    In the following prompt if the outside style and inside style are specified separately, identify and separate the two styles (outside and inside styles respectively; which must be distinct) separated by a newline, and if the inside or outside styles are not specified / omitted in the prompt just return \"DNE\" and try to adhere to the original description as much as possible. 
    <Example Prompt1>
    Halloween style on the outside and christmas style inside
    
    <Example Output1>
    Outside: Halloween style
    Inside: Christmas style
    
    <Example Prompt2>
    Christmas style
    
    <Example Output2>
    DNE
    
    <Prompt>
    \"{prompt}\"
    
    <Output>
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": request_message}
        ]
    )

    response = completion.choices[0].message.content
    if '\n' in response:
        outside_style, inside_style = response.split("\n")
        outside_style, inside_style = outside_style.split(':')[1], inside_style.split(':')[1]
    
    return outside_style, inside_style

if __name__ == "__main__":
    prompt = "Halloween style on the outside and christmas style inside"
    get_styles(prompt)