from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_new_tokens": 150}
        

        pipe = pipeline("text-generation", model=self.config.model_path,tokenizer=tokenizer)
        

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)
        print("\nModel Summary:")
        print(output)

        return output