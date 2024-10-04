## My Mini GPT
### Summary
Ongoing work to re-create, from scratch, the GPT2 (124M) model. I followed the work outlined in "Attention is All You Need" - the seminal transformer architecture paper. With this as a starting point, I made many design alterations and customized the training loop and architecture based on specifications outlined in the GPT2 and GPT3 papers released by OpenAI. These enhancements reduced training time, increased memory efficiency, and eventually increased model accuracy. The dataset I used for this project was filtered Common Crawl.\
\
I encountered a number of challenges when working on this project including increasing loss during pre-training (oof), running out of memory locally and on Google Colab (obvious), and a high cost of training on GPU clusters from inefficiently written training loops. The answers to most of these problems I found hints to in the literature, and was able to figure out how to fix them. After trying a bunch of things and reading through papers I decided on replicating the GPT2 / small GPT3 approach of training for tens of thousands of steps with ~.5M tokens per batch and using the Byte-Per-Encoding tokenization scheme to tokenize my input text into a sequence of 50,257 tokens. I experimented with the use of weight tying, weight initialization, gradient clipping, and a cosine decay based learning rate scheduler to increase training efficiency, training stability, and model performance. \
\
One bottleneck I encountered was when I tried to stream my dataset instead of loading the whole thing into memory - I think this greatly increased the training time as it required that we load an example each time we needed to retrieve a batch. 
Currently I am at a stage where I have achieved limited success with pre-training. The next step is to try sharding my dataset - loading it in and tokenizing it before starting training. Once that is done - I anticipate a fast, efficient training loop.\
\
### Next Steps
Once I have completed my pre-training, my next goal is to fine-tune the model to a dataset containing the full published works of the philosopher Nietzsche. The goal is to create a next character prediction model based on Nietzsche's works.\
\
Another and more interesting idea I have for future direction is to create a dataset of math problems (just simple calculator operations +, -, *, /) and train my model to perform Chain-Of-Thought Reasoning to learn how to perform these operations. This will involve constructing a large dataset of math questions (something like 35 + 42 = ?) with output consisting of not just the answer but the steps needed to get the answer. The idea is the model learns not just how to produce an answer, but also learns the steps necessary to obtain the answer. 
