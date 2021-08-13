
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer
from transformers.generation_utils import GenerationMixin


class GPTContext:
    model: GenerationMixin
    tokenizer: PreTrainedTokenizer
    max_position_embeddings: int
    device: str

    def __init__(self, model: GenerationMixin, tokenizer: PreTrainedTokenizer, max_position_embeddings: int, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.max_position_embeddings = max_position_embeddings
        self.device = device

    def create(ModelType: GenerationMixin, model_name: str, tokenizer_name: str = "", device="cuda:0", half=True):
        model = ModelType.from_pretrained(model_name)
        if half:
            model = model.half()
        model = model.to(device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if len(tokenizer_name) else model_name)
        config = AutoConfig.from_pretrained(model_name)
        try:
            # GPT2Config
            max_position_embeddings = config.n_ctx
        except AttributeError:
            # GPTNeoConfig
            max_position_embeddings = config.max_position_embeddings
        return GPTContext(model, tokenizer, max_position_embeddings, device)


class GenerationOptions(object):
    number_generated_tokens = 60
    temperature = 0.8
    tfs = None
    top_k = 60
    top_p = 0.9
    repetition_penalty = 2.5
    repetition_penalty_range = 512
    repetition_penalty_slope = 3.33
    prevent_square_brackets = True
    prevent_angle_brackets = True
    prevent_curly_brackets = True

    def __init__(self, number_generated_tokens=60, temperature=0.8, tfs=None, top_k=60, top_p=0.9,
                 repetition_penalty=2.5, repetition_penalty_range=512, repetition_penalty_slope=3.33,
                 prevent_square_brackets=True, prevent_angle_brackets=True, prevent_curly_brackets=True):
        self.number_generated_tokens = number_generated_tokens
        self.temperature = temperature
        self.tfs = tfs
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_range = repetition_penalty_range
        self.repetition_penalty_slope = repetition_penalty_slope
        self.prevent_square_brackets = prevent_square_brackets
        self.prevent_angle_brackets = prevent_angle_brackets
        self.prevent_curly_brackets = prevent_curly_brackets
