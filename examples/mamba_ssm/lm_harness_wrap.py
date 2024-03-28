import torch

import transformers
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments


from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from ofm import OFM


@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        pretrained="state-spaces/mamba-1.4b-hf",
        max_length=1024,
        batch_size=None,
        device="cuda",
        dtype=torch.float16,
    ):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(pretrained)

        if model.elastic_config is None:
            elastic_config = {
                "atten_out_space": [768],
                "inter_hidden_space": [4096, 3072, 1920, 1280],
                "residual_hidden_space": [768],
            }
            model.elastic_config = elastic_config

        supernet = OFM(model=model)

        self._model, param, config = supernet.smallest_model()

        self._model.to("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self.add_bos_token = False

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
