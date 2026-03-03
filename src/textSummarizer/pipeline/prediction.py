from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self,text):
        #tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_new_tokens": 150}
        #pipe = pipeline("text-generation", model=self.config.model_path,tokenizer=tokenizer)


        model_ckpt = "google/pegasus-cnn_dailymail"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
        pipe = pipeline("text-generation", model_pegasus,tokenizer=tokenizer)
        
        

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)
        print("\nModel Summary:")
        print(output)

        return output