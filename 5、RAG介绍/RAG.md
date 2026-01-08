# 背景
> LLM在生成内容的时候，具有强大的能力，但是也会面临一些`**问题**`：“`**幻觉**`”问题，即**胡编乱造**，**捏造事实**的问题。除此之外，LLM的`**训练数据可能过时**`，因此在生成任务的时候`**准确性**`和`**精度**`难以保证。
>

`**方案**`：检索增强生成（Retrieval-Augmented Generation，RAG）。具体：

RAG在回答之前，会从外部的数据库中检索相关信息，并将信息拼接到上下文中，从而指导模型的输出。

`**核心原理**`：

+ 检索。当用户提出查询时，系统通过检索找到问题的相关文本片段，作为上下文信息传递给模型，模型据此生成更为精准和可靠的回复。
+ 生成。



# 搭建RAG框架
本章节仅仅介绍如何搭建RAG的大致框架，对于具体代码，请参考TinyRAG目录。

## 基本结构
+ 向量模块：将文本进行向量化。之后通过向量检索，会更快。
+ 文档加载与切分模块：用于加载文档并且分成文档片段。
+ 数据库：存储文档与对应向量。
+ 检索模块：根据query，来获取相关的文档片段。
+ 大模型模块：提供LLM服务。

## RAG的流程
+ 索引：将文档分割成比较小的片段，并通过编码器构建向量索引。
+ 检索：根据问题和片段，计算相似度检索相关文档片段。
+ 生成：以检索到的上下文为条件，生成问题的回答。

流程如下：（图片出处<font style="color:rgb(52, 73, 94);"> </font>[_**<font style="color:rgb(66, 185, 131);">Retrieval-Augmented Generation for Large Language Models: A Survey</font>**_](https://arxiv.org/pdf/2312.10997.pdf)_**<font style="color:rgb(66, 185, 131);">）</font>**_

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/35251293/1767868813100-b8f32bea-e17a-4bb4-80bc-c59d4ce6b1c6.png)

流程图

## 文档加载和切分
实现一个class，用于文档的加载和切分。

```python
# 读取文档
def read_file_content(cls, file_path: str):
    # 根据文件扩展名选择读取方法
    if file_path.endswith('.pdf'):
        return cls.read_pdf(file_path)
    elif file_path.endswith('.md'):
        return cls.read_markdown(file_path)
    elif file_path.endswith('.txt'):
        return cls.read_text(file_path)
    else:
        raise ValueError("Unsupported file type")


# 文档切分
# 我们可以设置一个最大的Token长度，然后根据这个最大长度来切分文档
# 但实际上，优先保证完整句子，以及片段间有些重叠，提交检索效率。
def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    token_len = max_token_len - cover_content
    lines = text.splitlines()  # 假设以换行符分割文本为行

    for line in lines:
        # 保留空格，只移除行首行尾空格
        line = line.strip()
        line_len = len(enc.encode(line))
        
        if line_len > max_token_len:
            # 如果单行长度就超过限制，则将其分割成多个块
            # 先保存当前块（如果有内容）
            if curr_chunk:
                chunk_text.append(curr_chunk)
                curr_chunk = ''
                curr_len = 0
            
            # 将长行按token长度分割
            line_tokens = enc.encode(line)
            num_chunks = (len(line_tokens) + token_len - 1) // token_len
            
            for i in range(num_chunks):
                start_token = i * token_len
                end_token = min(start_token + token_len, len(line_tokens))
                
                # 解码token片段回文本
                chunk_tokens = line_tokens[start_token:end_token]
                chunk_part = enc.decode(chunk_tokens)
                
                # 添加覆盖内容（除了第一个块）
                if i > 0 and chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    chunk_part = cover_part + chunk_part
                
                chunk_text.append(chunk_part)
            
            # 重置当前块状态
            curr_chunk = ''
            curr_len = 0
            
        elif curr_len + line_len + 1 <= token_len:  # +1 for newline
            # 当前行可以加入当前块
            if curr_chunk:
                curr_chunk += '\n'
                curr_len += 1
            curr_chunk += line
            curr_len += line_len
        else:
            # 当前行无法加入当前块，开始新块
            if curr_chunk:
                chunk_text.append(curr_chunk)
            
            # 开始新块，添加覆盖内容
            if chunk_text:
                prev_chunk = chunk_text[-1]
                cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                curr_chunk = cover_part + '\n' + line
                curr_len = len(enc.encode(cover_part)) + 1 + line_len
            else:
                curr_chunk = line
                curr_len = line_len

    # 添加最后一个块（如果有内容）
    if curr_chunk:
        chunk_text.append(curr_chunk)

    return chunk_text

```

## 向量化
向量化的目标是：将切分好的文档片段转化成向量。

我们这里定义一个`**BaseEmbedding**`基类，这样我们在使用其他模型的时候，只需要继承这个基类，在基类上进行修改即可。

```python
class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        """
        初始化嵌入基类
        Args:
            path (str): 模型或数据的路径
            is_api (bool): 是否使用API方式。True表示使用在线API服务，False表示使用本地模型
        """
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        获取文本的嵌入向量表示
        Args:
            text (str): 输入文本
            model (str): 使用的模型名称
        Returns:
            List[float]: 文本的嵌入向量
        Raises:
            NotImplementedError: 该方法需要在子类中实现
        """
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        Args:
            vector1 (List[float]): 第一个向量
            vector2 (List[float]): 第二个向量
        Returns:
            float: 两个向量的余弦相似度，范围在[-1,1]之间
        """
        # 将输入列表转换为numpy数组，并指定数据类型为float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # 检查向量中是否包含无穷大或NaN值
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0

        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        # 计算向量的范数（长度）
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算分母（两个向量范数的乘积）
        magnitude = norm_v1 * norm_v2
        # 处理分母为0的特殊情况
        if magnitude == 0:
            return 0.0
            
        # 返回余弦相似度
        return dot_product / magnitude

```

基类主要有两个方法：`**获取文本片段的embedding**`和`**计算两个embedding的相似度**` 。`<font style="color:#ED740C;">get_embedding</font>`用于获取文本片段的向量表示，`<font style="color:#ED740C;">cosine_similarity</font>`用于计算两个向量的余弦相似度。当然，我们在基类设置了是否适用api进行向量化的参数：`<font style="color:#ED740C;">self.is_api</font>`。

继承`**BaseEmbedding**`类只需要实现`<font style="color:#ED740C;">get_embedding</font>`方法即可，如果需要，可以复写父类的方法。具体例子：

```python
class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            # 从环境变量中获取 OPEN_API_KEY 密钥
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            # 从环境变量中获取 OPEN_API_KEY 的基础URL
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """
        此处默认使用 Silicon Flow 的免费嵌入模型 BAAI/bge-m3
        """
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError

```

## 数据库与向量检索
我们使用向量数据库来存储文档片段对应的向量表示，以及设计一个检索模块，用于使用query向量来检索相关文档片段。

向量数据库的功能包括：

+ `persist`：数据库持久化保存
+ `load_vector`：从本地加载数据库
+ `get_vector`：获取文档的向量表示
+ `query`：根据问题检索相关的文档片段

整体框架如下：

```python
class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # 获得文档的向量表示
        pass

    def persist(self, path: str = 'storage'):
        # 数据库持久化保存
        pass

    def load_vector(self, path: str = 'storage'):
        # 从本地加载数据库
        pass

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # 根据问题检索相关文档片段
        pass

```

## LLM模块
该模块比较简单，即将从向量数据库中检索到的相关文本片段拼接到query的上下文中，输入LLM即可。

这里实现一个基类：

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

```

我们使用LLM的api来构建一个LLM模块：

```python
from openai import OpenAI

class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")   
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append({'role': 'user', 'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)})
        response = client.chat.completions.create(
                model=self.model,
                messages=history,
                max_tokens=2048,
                temperature=0.1
            )
        return response.choices[0].message.content

```

设计一个专用于RAG的大模型提示词，如下：

```python
RAG_PROMPT_TEMPLATE="""
使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:
"""

```

## Tiny-RAG Demo
```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# 没有保存数据库
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
vector = VectorStore(docs)
embedding = OpenAIEmbedding() # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# vector.load_vector('./storage') # 加载本地的数据库

question = 'RAG的原理是什么？'

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))

```

也可以从本地加载已处理好的数据库:

```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# 保存数据库之后
vector = VectorStore()

vector.load_vector('./storage') # 加载本地的数据库

question = 'RAG的原理是什么？'

embedding = ZhipuEmbedding() # 创建EmbeddingModel

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
```

