import re


s ='  Ab我, love. '
print("so-->", s)
s = s.lower().strip()
print("s1-->", s)

# sub(pattern, repl, string)
# pattern: 正则表达式
# repl: 替换的字符串
# string: 原文本

#含义: 在string中,如果能使用pattern匹配,则替换匹配的内容为repl
# (): 分组,分组后,组号从1开始
# \1: 匹配第一个分组

s = re.sub(r"([.!?])",r" \1",s)
print("s2-->", s)