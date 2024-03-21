import urllib.request
import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import time


class GenAI_Wrpapper:
	def __init__(self, chat_client='chatgpt3.5'):
		self.is_gemini = False
		self.vectordb = None
		if chat_client == 'chatgpt3.5':
			self.chat_client = ChatOpenAI(model="gpt-3.5-turbo")
		elif chat_client == 'chatgpt3.5turbo':
			self.chat_client = ChatOpenAI(model="gpt-3.5-turbo-instruct")
		elif chat_client == 'chatgpt4':
			self.chat_client = ChatOpenAI(model="gpt-4")
		elif chat_client == 'gemini':
			genai.configure()
			self.chat_client = genai.GenerativeModel('gemini-pro')
			self.is_gemini = True
		self.embedding = OpenAIEmbeddings()
		self.sizes = {
			"abstract": 50,
			"introduction": 300,
			"methodology": 500,
			"results": 200,
			"conclusion": 100
					}
		self.queries = {
			"abstract": "abstract",
			"introduction": "Extract the introduction section discussing background information and research objectives.",
			"methodology": "Get the methodology section detailing experimental design, data collection, and analysis techniques.",
			"results": "results",
			"conclusion": "conclusion"
					}
	
	def get_document_splits(self, file_data, chunk_size=1500, chunk_overlap=150):
		splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
		splits = splitter.split_text(file_data)
		return splits

	def create_vectordb_from_document_splits(self, document_splits):
		return FAISS.from_texts(document_splits, embedding=self.embedding)
	  
	def get_component_output_parser(self, component):
		summary = ResponseSchema(name=f"{component}", description=f"Very brief summary of the {component} in less than {self.sizes[component]} words.")
		return StructuredOutputParser.from_response_schemas([summary])

	def get_summary_output_parser(self):
		summary = ResponseSchema(name=f"summary", description=f"Summary of the research paper in 1000 words.")
		return StructuredOutputParser.from_response_schemas([summary])

	def get_qa_output_parser(self):
		answer = ResponseSchema(name=f"answer", description=f"Answer to the question asked in only one sentence.")
		return StructuredOutputParser.from_response_schemas([answer])

	def get_component_prompt_template(self):
		prompt_template_text = """
		You will be provided with {component} part of a research paper enclosed within {delimiter} delimiter.
		Please provide a summary of this {component}.
		Please make sure that the summary is concise and to the point.

		The {component} part of the research paper can have some information unrealated to the {component}. You must ignore it.

		{delimiter}{context}{delimiter}

		{instructions}
		"""
		return PromptTemplate(template=prompt_template_text, input_variables=["context", "delimiter", "instructions", "component"])
		
	def get_summary_prompt_template(self):
		prompt_template_text = """
		You will be provided with details of {components_list} from a research paper enclosed within {delimiter} delimiter.
		Please provide a combined summary from it.
		Please make sure that the summary is concise and to the point.

		{component_summary}

		{instructions}
		"""
		return PromptTemplate(template=prompt_template_text, input_variables=["component_summary", "delimiter", "instructions", "components_list"])

	def get_qa_prompt_template(self):
		prompt_template_text = """
		You will be provided with a question (delimited with {question_delimiter}) pertaining to a research paper.
		You will also be provided with a relevant context (delimited with {context_delimiter}) extracted from a research paper.

		Please answer the question keeping only the context in mind.
		Answer must be one sentence long.

		{question_delimiter}{question}{question_delimiter}

		{context_delimiter}{context}{context_delimiter}

		{instructions}

		Let me remind again. Answer must be one sentence long.
		"""
		return PromptTemplate(template=prompt_template_text, input_variables=["context", "delimiter", "instructions", "component"])

	def get_qa_chain(self, prompt):
		return load_qa_chain(llm=self.chat_client, chain_type="stuff", prompt=prompt)

	def get_llm_chain(self, prompt):
		return LLMChain(llm=self.chat_client, prompt=prompt)

	def run_component_qa_chain(self, component, vectordb):
		query = "What are the skills and educational qualifications of the candidate?"
		prompt = self.get_component_prompt_template()
		output_parser = self.get_component_output_parser(component)
		instructions = output_parser.get_format_instructions()
		context = vectordb.similarity_search(query, k=1)
		chain = self.get_qa_chain(prompt)
		prompt_inputs = {"input_documents": context, "delimiter": "###", "instructions": instructions,
					   "component": component, "words": self.sizes[component]}
		response = chain(prompt_inputs, return_only_outputs=True)
		response_dict = output_parser.parse(response["output_text"])
		return response_dict

	def run_summary_llm_chain(self, component_summary, components_list):
		prompt = self.get_summary_prompt_template()
		output_parser = self.get_summary_output_parser()
		instructions = output_parser.get_format_instructions()
		chain = self.get_llm_chain(prompt)
		response = chain({"component_summary": component_summary, "delimiter": "###", "instructions": instructions,
					   "components_list": ", ".join(components_list)}, return_only_outputs=True)
		response_dict = output_parser.parse(response['text'])
		return response_dict
		
	def run_qa_chain(self, question, vectordb):
		query = question
		prompt = self.get_qa_prompt_template()
		output_parser = self.get_qa_output_parser()
		instructions = output_parser.get_format_instructions()
		context = vectordb.similarity_search(query, k=1)
		chain = self.get_qa_chain(prompt)
		prompt_inputs = {"input_documents": context, "context_delimiter": "###", "question_delimiter": "$$$",
					   "instructions": instructions, "question": question}
		response = chain(prompt_inputs, return_only_outputs=True)
		response_dict = output_parser.parse(response["output_text"])
		return response_dict
		
	def get_summary(self, research_paper_content):
		document_splits = self.get_document_splits(research_paper_content, chunk_size=1500, chunk_overlap=150)
		self.vectordb = self.create_vectordb_from_document_splits(document_splits)
		components = ["abstract", "introduction", "methodology", "results", "conclusion"]
		summary = ""
		for component in components:
			response = self.run_component_qa_chain(component, self.vectordb)
			summary += "\n\n\n\n<" + component + ">\n\n"
			summary += "###" + response[component] + "###"
		response = self.run_summary_llm_chain(component_summary=summary, components_list=components)
		return response['summary'].replace(".", ".\n")

	def get_answer(self, question):
		response = self.run_qa_chain(question, self.vectordb)
		return response['answer']

class Page:
	def __init__(self):
		self.research_paper_url = None
		self.research_paper_content = ""
		self.submit_object = None
		self.submit_object2 = None
		self.genai_wrapper_object = None
		if 'genai_wrapper_object' not in st.session_state:
			st.session_state.genai_wrapper_object = None
		else:
			self.genai_wrapper_object = st.session_state.genai_wrapper_object
		self.research_paper_summary = None
		if 'research_paper_summary' not in st.session_state:
			st.session_state.research_paper_summary = None
		else:
			self.research_paper_summary = st.session_state.research_paper_summary
	
	def create_header(self, displayText="Header"):
		st.header(displayText)
	
	def create_subheader(self, displayText="Sub-Header"):
		st.subheader(displayText)
		
	def create_input_text(self, displayText="Input", height=150):
		return st.text_area(displayText, height=height)
	
	def create_error_message(self, displayText="Error"):
		st.error(displayText, icon="🚨")
	
	def create_file_widget(self, displayText="Choose a file...", fileType="pdf"):
		return st.file_uploader(displayText, type=fileType)
	
	def create_submit_button(self, displayText="Submit"):
		return st.button(displayText)
		
	def read_research_paper(self):
		local_file_name = self.research_paper_url.split("/")[-1]
		urllib.request.urlretrieve(self.research_paper_url, local_file_name)
		reader=pdf.PdfReader(local_file_name)
		self.research_paper_content = ""
		for page in range(len(reader.pages)):
			page = reader.pages[page]
			self.research_paper_content += str(page.extract_text())
		
	def display_research_paper_summary(self, results):
		st.write("### Research Paper Summary")
		i = 1
		for result in results.split('\n'):
			if len(result.strip()) > 0:
				st.write(f"##### {i}. {result}")
				i += 1
		
	def get_summary(self):
		return self.genai_wrapper_object.get_summary(self.research_paper_content)
		
	def get_answer(self, question):
		response = self.genai_wrapper_object.get_answer(question)
		if len(response.strip()) > 0:
			st.write("#### Answer")
			st.write(response)
		
	def create_page(self):
		alternative_model = {"chatgpt3.5": "gemini", "gemini": "chatgpt3.5"}
		chat_client = st.selectbox("Choose a model:", ("chatgpt3.5", "gemini"))
		chat_client='chatgpt3.5'
		self.create_header(displayText="Enter url of your research paper.")
		self.research_paper_url = self.create_input_text(displayText="Paste the url of research paper here...", height=1)
		self.submit_object = self.create_submit_button(displayText="Load")
		if self.submit_object:
			if self.research_paper_url is not None:
				try:
					self.read_research_paper()
				except Exception as e:
					self.create_error_message(displayText=f"Please provide valid url.")
					return
			if len(self.research_paper_content.strip()) == 0:
				self.create_error_message(displayText="Please provide a valid research paper.")
			elif len(self.research_paper_content.strip()) > 0:
				try:
					self.genai_wrapper_object = GenAI_Wrpapper(chat_client)
					self.research_paper_summary = self.get_summary()
					st.session_state.research_paper_summary = self.research_paper_summary
					st.session_state.genai_wrapper_object = self.genai_wrapper_object
				except Exception as e1:
					try:
						self.genai_wrapper_object = GenAI_Wrpapper(alternative_model[chat_client])
						self.research_paper_summary = self.get_summary()
						st.session_state.research_paper_summary = self.research_paper_summary
						st.session_state.genai_wrapper_object = self.genai_wrapper_object
					except Exception as e2:
						self.create_error_message(displayText=f"Unble to connect to ChatBot at his moment. Please try again later.")
		if self.research_paper_summary:
			self.display_research_paper_summary(self.research_paper_summary)
			self.create_header(displayText="Ask a question.")
			question = self.create_input_text(displayText="Paste your question here...", height=1)
			self.submit_object2 = self.create_submit_button(displayText="Ask")
			if self.submit_object2:
				if len(question.strip()) == 0:
					self.create_error_message(displayText=f"Please ask a valid question.")
				else:
					self.get_answer(question=question)
					try:
						self.get_answer(question=question)
					except Exception as e1:
						self.create_error_message(displayText=f"Unble to connect to ChatBot at his moment. Please try again later.{e1}")
page = Page()
page.create_page()
