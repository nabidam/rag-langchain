from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pickle

load_dotenv()

# base_url = "https://python.langchain.com"
# home_url = "https://python.langchain.com/docs/get_started/introduction"

# proxies = {"http": "http://127.0.0.1:2080", "http": "http://127.0.0.1:2080"}

# page = requests.get(home_url, proxies=proxies)

# bs = BeautifulSoup(page.content, "html.parser")

# nav_hrefs = []

# nav_lis = bs.find_all(class_="menu__link")

# # init loop
# for li in nav_lis:
#     nav_hrefs.append(li["href"])

# raw_data = []

# idx = 0
# while idx < len(nav_hrefs):
#     href = nav_hrefs[idx]

#     this_url = base_url + href
#     print(f"[INFO] url: {this_url} is going to be processed")

#     # add new links to list
#     this_page = requests.get(this_url, proxies=proxies)
#     this_bs = BeautifulSoup(this_page.content, "html.parser")
#     this_nav_lis = this_bs.find_all(class_="menu__link")
#     # init loop
#     for li in this_nav_lis:
#         if li["href"] not in nav_hrefs:
#             nav_hrefs.append(li["href"])

#     # use loader to load data
#     loader = WebBaseLoader(this_url)
#     data = loader.load()

#     raw_data.append(data)

#     print(f"[INFO] url: {this_url} is processed")

#     idx += 1

file_path = "links.pickle"

# with open(file_path, "wb") as file:
#     # Serialize and write the variable to the file
#     pickle.dump(nav_hrefs, file)

with open(file_path, "rb") as file:
    # Serialize and write the variable to the file
    nav_hrefs = pickle.load(file)

file_path = "raw_data.pickle"

# # Open the file in binary mode
# with open(file_path, "wb") as file:
#     # Serialize and write the variable to the file
#     pickle.dump(raw_data, file)

with open(file_path, "rb") as file:
    # Serialize and write the variable to the file
    raw_data = pickle.load(file)

vectorstore = Chroma(persist_directory="db", embedding_function=OpenAIEmbeddings())

for idx, data in enumerate(raw_data):
    print(f"[INFO] idx: {idx}/{len(raw_data)} is going to be embedded and stored.")

    # split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # store splits
    vectorstore.add_documents(documents=all_splits)
    print(f"[INFO] idx: {idx}/{len(raw_data)} is processed")

vectorstore.persist()
