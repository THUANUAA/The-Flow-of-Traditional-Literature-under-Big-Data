import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import argparse
import pandas as pd
import os
from sqlalchemy import create_engine

# 模型文件所在目录
model_dir = "C:\\Users\\86189\\Desktop\\moxing"




# engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/mysql-base')


# 加载模型和分词器
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 接收输入函数
def Infer(input_review):
    encoded_input = tokenizer.encode(input_review, return_tensors='pt')
    result = model(encoded_input)['logits']
    label = torch.argmax(result, dim=-1)
    return int(label)


def InferMode():
    print("InferMode.Press Q to exit.")
    while True:
        input_review = input("Input Review: ")
        if input_review == 'Q':
            print("Exit.")
            break
        label = Infer(input_review)
        print("Positive." if label else "Negative.")

#df = pd.read_sql_table('table_name', engine)
df = pd.read_csv("哪吒.csv", delimiter=',', encoding='utf-8-sig')

for i in range(df.shape[0]):
    if len(str(df.content.values[i]))>250:
        df.result.values[i] = 1
        continue
    parser = argparse.ArgumentParser(description='Quick Inference')
    parser.add_argument("-s", "--sentence", metavar=None, type=str,
                        default=str(df.content.values[i]))
    parser.add_argument("-i", "--infer_mode", metavar=None, type=bool, default=False)
    args = parser.parse_args()
    if __name__ == '__main__':
        if args.infer_mode:
            InferMode()
        else:
            infer_label = Infer(args.sentence)
            # print("Your Input: " + args.sentence)
            # print("Infer Result: " + ("Positive." if infer_label else "Negative."))
            # df.喜厌.values[i] = 2 if infer_label else 1
            if infer_label:
                df.result.values[i] = 0
            else:
                df.result.values[i] = 1
            print(i,df.result.values[i])
# df_to_update = df  # 假设 df 是你要更新的数据框
# df_to_update.to_sql('table_name', engine, if_exists='replace', index=False)
# os.remove('ggb1.csv')
df.to_csv("B站哪吒5.csv",index=False,encoding='utf-8-sig')
# 使用argparse
# parser = argparse.ArgumentParser(description='Quick Inference')
# parser.add_argument("-s", "--sentence", metavar=None, type=str,
#                     default="谁家好人买汉服呀")
# parser.add_argument("-i", "--infer_mode", metavar=None, type=bool, default=False)
# args = parser.parse_args()
#
# if __name__ == '__main__':
#     if args.infer_mode:
#         InferMode()
#     else:
#         infer_label = Infer(args.sentence)
#         print("Your Input: " + args.sentence)
#         print("Infer Result: " + ("Positive." if infer_label else "Negative."))
