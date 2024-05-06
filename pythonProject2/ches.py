import pandas as pd
import re


def contains_chinese(s):
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(pattern.search(s))


def keep_english_numbers_punctuation(text):
    # 使用正则表达式替换非匹配字符为空字符串
    if isinstance(text, str):  # 确保text是字符串类型
        return re.sub(r'[^A-Za-z0-9\s\.,!?;:\'\"-]', '', text)
    else:
        return text


def remove_chinese(text):
    if isinstance(text, str):  # 确保text是字符串类型
        return re.sub(r'[\u4e00-\u9fff]+', '', text)
    else:
        return text


# 保留中文
def extract_chinese_and_punctuation(text):
    # 汉字和中文标点符号的正则表达式
    pattern = re.compile(r'[\u4e00-\u9fff\u3002\uff0c\uff1b\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]')
    if isinstance(text, str):
        chinese_and_punctuation = pattern.findall(text)
        return ''.join(chinese_and_punctuation)
    else:
        return text


df = pd.read_csv('哪吒jkc.csv', delimiter=',', encoding='utf-8-sig')
df1 = pd.read_csv('哪吒lr.csv', delimiter=',', encoding='utf-8-sig')
df2 = pd.read_csv('哪吒pb.csv', delimiter=',', encoding='utf-8-sig')

# 合并三个文件
ret = pd.concat([df, df1, df2])
# 保留指定列
ret = ret[['用户昵称', '微博正文', '发布位置', '发布时间']]
# 列名格式化
ret.columns = ['nickname', 'content', 'ip_location', 'create_date_time']

# 去除英文
for i in range(ret.shape[0]):
    ret.content.values[i] = extract_chinese_and_punctuation(ret.content.values[i])

ret.to_csv("处理后哪吒.csv", index=False, encoding='utf-8-sig')
