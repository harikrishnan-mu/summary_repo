from cluster import cluster_based_summary
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from cluster import cluster_based_summary
loader = PyPDFLoader('data/Joanne-K.-Rowling-Harry-Potter-Book-1-Harry-Potter-and-the-Philosophers-Stone.pdf')
docs = loader.load()
total_text_data = ''
for i in docs:
    total_text_data= total_text_data + '\n' + i.page_content
cluster_summarizer = cluster_based_summary(total_text_data)
print(cluster_summarizer)
print('***********************')
result_dict = cluster_summarizer(total_text_data)

print(result_dict)

