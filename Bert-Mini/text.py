import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import logging
import time
import numpy as np

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从CSV文件中读取数据
csv_file = "C:/Users/86176/Desktop/计算机设计大赛/fine-tuning-Bert-for-sentiment-analysis-master/抖音数据/哪吒.csv"  # 更新为你的CSV文件路径
data = pd.read_csv(csv_file)

# 初始化情感分析管道
model_dir = "C:/Users/86176/Desktop/计算机设计大赛/fine-tuning-Bert-for-sentiment-analysis-master/uncased_L-12_H-768_A-12"  # 更新为你的预训练模型所在的路径

try:
    logger.info("加载情感分析模型...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
except Exception as e:
    logger.error("加载情感分析模型时出错：{}".format(str(e)))
    exit()

# 对每条数据进行情感分析并写入CSV文件
sentiments = []
logger.info("开始情感分析...")
for index, row in tqdm(data.iterrows(), total=len(data)):
    text = str(row['content']) if isinstance(row['content'], str) else ""
    if not text:
        logger.warning("第{}条评论为空，跳过情感分析。".format(index))
        sentiments.append("")  # 将空字符串作为标记
        continue

    # 多次运行模型并取中位数作为结果
    num_runs = 1  # 设置运行次数
    results = []
    for _ in range(num_runs):
        try:
            result = classifier([text])  # 将文本转换为列表以符合预期的输入格式
            sentiment_score = result[0]['score']
            results.append(sentiment_score)
        except Exception as e:
            logger.error("处理第{}条评论时出错：{}".format(index, str(e)))

    # 如果结果为空，则将空字符串作为标记
    if not results:
        sentiments.append("")
        continue

    # 使用中位数作为最终结果
    median_score = np.median(results)
    if median_score > 0.55:
        sentiments.append("positive")
    else:
        sentiments.append("negative")

# 将情感结果添加到DataFrame中
data['result'] = sentiments

# 创建新的文件名
timestamp = time.strftime("%Y%m%d%H%M%S")
output_csv = "C:/Users/86176/Desktop/计算机设计大赛/fine-tuning-Bert-for-sentiment-analysis-master/sentiment_results_{}.csv".format(
    timestamp)

# 将DataFrame写入到CSV文件中
data.to_csv(output_csv, index=False)

logger.info("情感分析结果已保存至: {}".format(output_csv))
