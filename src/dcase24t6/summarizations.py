import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
import vertexai.preview.generative_models as generative_models

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]


def generate_summarization(captions: list[str]):
    """
    Generate a summarization of the given captions.

    Args:
        captions (List[str]): A list of captions to summarize.
    """
    vertexai.init(project="ancient-script-432010-j0", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-pro-001",
    )

    captions = ', '.join(captions)

    # The following prompt is a slight modification of Jee-weon Jung, Dong Zhang, Huck C.-H. Yang, Shih-Lun Wu, 
    # David M. Chan,  Zhifeng Kong, Deng Ruifan, Zhou Yaqian, Valle Rafael, and Shinji Watanabe. Automatic audio 
    # captioning  with encoder fusion, multi-layer aggregation, and large language model enriched summarization. 
    # Technical report, DCASE2024 Challenge, May 2024.
    responses = model.generate_content(
        [f"This is a hard problem. Carefully summarize in ONE detailed sentence the following captions by different 
         (possibly incorrect) people describing the same audio. Be sure to describe everything, including the source 
         and background of the sounds, identify when you’re not sure. Do not allude to the existence of the multiple 
         captions. Do not start your summary with sentence like “The audio (likely) features”, “The audio (likely) 
         captures” and so on. Focus on describing the content of the audio. Note that your summary MUST be shorter 
         than twenty words and use subject-predicate-object structure. Your summary NEEDS to use present continuous 
         tense whenever possible. HERE is the question, Captions: {captions}."],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    return ''.join([response.text for response in responses]).replace(' \n', '')
