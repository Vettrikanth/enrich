import re
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# Define a custom Document class
class Document:
    def _init_(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Initialize the ChatOllama model
model_local = ChatOllama(model="mistral")

# Function to set up Selenium WebDriver
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# Function to fetch content from URL using Selenium
def fetch_content_with_selenium(url):
    driver = setup_driver()
    driver.get(url)
    content = driver.page_source
    driver.quit()
    return content

# Enhanced function to extract emails and phone numbers using regex
def extract_emails_and_phones(text):
    # Regular expression for standard emails
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    # Regular expression for obfuscated emails
    obfuscated_email_pattern = re.compile(r'\b(mail|email|gmail)\b[\w\s:]*[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}')
    
    # Regular expression for phone numbers (international pattern) with descriptors
    phone_pattern = re.compile(
        r'\b(fax|tel|phone|telephone)\b[\w\s:]*((\+\d{1,3}[-.\s]?)?(\(?\d{1,4}\)?[-.\s]?)?(\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}))',
        re.IGNORECASE
    )
    
    # Find all emails
    emails = email_pattern.findall(text)
    obfuscated_emails = obfuscated_email_pattern.findall(text)
    
    # Find all phone numbers with descriptors
    phones = phone_pattern.findall(text)
    
    # Flatten the list of tuples for phone numbers and remove descriptor
    phones = [''.join(phone[1:]) for phone in phones]
    
    return emails + obfuscated_emails, phones

# Function to fetch content using WebBaseLoader
def fetch_content_with_webbase(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    text = " ".join(doc.page_content for doc in documents)
    return text

# Define URLs and choose the method to fetch content
url = "https://ngmc.edu.np/"
use_selenium = True  # Set to False if you want to use WebBaseLoader instead of Selenium-

if use_selenium:
    html_content = fetch_content_with_selenium(url)
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
else:
    text = fetch_content_with_webbase(url)

# Logging the fetched text for debugging
logger.info("Fetched text: %s", text[:500])  # Print first 500 characters for debugging

# Create a Document object
doc = Document(page_content=text)

# Split the document into smaller chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents([doc])

# Create a vector store from the documents
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Define the template and prompt for extracting email and contact information
after_rag_template = """Extract email addresses and phone numbers based on the following context:
{context}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "text": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)

# Invoke the chain with the context from the retriever
response = after_rag_chain.invoke({"text": text})

# Logging the response for debugging
logger.info("Processed text: %s", response)

# Extract emails and phone numbers from the processed text
emails, phones = extract_emails_and_phones(response)
print("Extracted Emails:", emails)
print("Extracted Phone Numbers:", phones)
