# MedBot: Medical Chatbot with Instruction Fine-Tuning and Conversational Memory


<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/medbot-instruct-conversational/blob/master/MedBot_Medical_Chatbot_Instruction_Fine_Tuning_Conversational_Memory.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
</div>


Language can be thought of as a conduit through which our abstract mind is exposed. Recently, the growth of AI has followed this notion, and almost everyone in the world benefits from it on a regular basis, particularly in the medical field. The language-based AI technology has the potential to alter the medical industry by allowing patients to connect with conversational machines. It allows anyone to get medical information from a computer in a more natural way. We can create a chatbot that plays the role of a medical practitioner. MedBot, a chatbot, is built on the well-known LLM, LLaMA 3, with instruction fine-tuning. It follows the directions in the prompt. To alleviate the effort of fine-tuning, the notoriously heavyweight LLM is quantized using 4-bit quantization. Additionally, LoRA (low-rank adaptation) is applied. These methods are collectively referred to as QLoRA. To maximize the fine-tuning efficiency, we have to load the LLM from the Unsloth library. Once the LLM model is fine-tuned, we can funnel it to LangChain, rendering a chatbot with conversational memory. We can converse with the chatbot via the Telegram bot. Last but not least, the fine-tuned LLM is trained and tested on ``Shekswess/medical_llama3_instruct_dataset_short``. Finally, ROUGE is used to measure its quantitative performance.



## Experiment

Converse with the bot by running this [notebook](https://github.com/reshalfahsi/medbot-instruct-conversational/blob/master/MedBot_Medical_Chatbot_Instruction_Fine_Tuning_Conversational_Memory.ipynb).


## Result

## Quantitative Result

Here is the overview of the bot's quantitative performance.

Test Metric | Score  |
----------- | -----  |
ROUGE-1     | 35.98% |
ROUGE-1     | 19.67% |
ROUGE-L     | 27.69% |
ROUGE-L Sum | 28.42% |


## Qualitative Result

Below is a snapshot of the bot's conversational performance.

<p align="center"> <img src="https://github.com/reshalfahsi/medbot-instruct-conversational/blob/master/assets/qualitative.png" alt="qualitative" > <br /> Conversation about Epilepsy with MedBot. </p>


## Credit

- [LLM-Medical-Finetuning](https://github.com/Shekswess/LLM-Medical-Finetuning)
- [Chatbot Project](https://github.com/areebniyas/chat-bot)
- [Conversational Memory for LLMs with Langchain](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)
- [Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
- [Fine Tuning LLAMAv2 with QLora on Google Colab for Free](https://www.kdnuggets.com/fine-tuning-llamav2-with-qlora-on-google-colab-for-free)
- [QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- [Cross-Task Generalization via Natural Language Crowdsourcing Instructions](https://aclanthology.org/2022.acl-long.244.pdf)
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/pdf/2109.01652)
- [Shekswess/medical_llama3_instruct_dataset_short](https://huggingface.co/datasets/Shekswess/medical_llama3_instruct_dataset_short)
- [unsloth/llama-3-8b-Instruct-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit)
- [Unsloth](https://github.com/unslothai/unsloth)
- [🤗 Evaluate](https://github.com/huggingface/evaluate)
- [🤗 Datasets](https://github.com/huggingface/datasets)
- [🦜️🔗 LangChain](https://github.com/langchain-ai/langchain)
