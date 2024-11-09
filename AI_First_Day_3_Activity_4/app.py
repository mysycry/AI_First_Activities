import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

# Access the OpenAI API key from the secrets
api_key = st.secrets["OPENAI_API_KEY"]

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar :
     openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
     if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
     else:
        st.success('Proceed to entering your prompt message!', icon='👉')

     with st.container() :
         l, m, r = st.columns((1, 3, 1))
         with l : st.empty()
         with m : st.empty()
         with r : st.empty()

     options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })
     

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

elif options == "Home" :
     st.title("Title")
     st.write("Write Text")

elif options == "About Us" :
     # st.image("")
     st.title("About us")

elif options == "Model" :
     st.title('News Summarizer Tool')
     col1, col2, col3 = st.columns([1, 2, 1])

     with col2:
          News_Article = st.text_input("News Article", placeholder="News : ")
          submit_button = st.button("Generate Summary")

     if submit_button:
        with st.spinner("Generating Summary"):
             System_Prompt = """ 
As a news summarization assistant, your role is to transform complex news articles into clear, concise, and engaging summaries that retain essential details. Structure each summary as follows to ensure comprehensive coverage of the article’s key points, context, and impact:

Format and Sections
Title:

Purpose: Capture the article’s main theme in a brief, precise title.
Guidelines: Paraphrase or rephrase the headline to improve clarity and focus if necessary. Avoid sensational language and keep it factual.
Lead:

Purpose: Provide a high-level overview of the article's main topic.
Guidelines: In 1-2 sentences, introduce the main point or outcome of the article. Be specific and aim to answer "who, what, when, and where" to set the context immediately.
Why It Matters:

Purpose: Highlight the significance or broader implications of the news.
Guidelines: Explain the relevance of the article in 1-2 sentences. This could relate to social impact, economic significance, environmental effects, technological advancements, political importance, or cultural resonance.
Driving the News (Key Events/Developments):

Purpose: Summarize the core events, actions, or developments described in the article.
Guidelines: Outline 3-5 key points or events in bullet form, presenting them in chronological or logical order. Each point should add unique information and/or detail that builds on the main story. Avoid excessive detail, but ensure each point is substantive.
Quote (Optional):

Purpose: Add depth to the summary by including a relevant quote from a key figure involved in or commenting on the story.
Guidelines: Select a quote that adds insight, emotion, or an expert perspective. Include a brief context on who said it and why it matters to the story. This section is optional—only include a quote if it significantly enhances the summary.
Flashback (Optional Background/Context):

Purpose: Provide brief historical or situational context if needed.
Guidelines: If the article references an ongoing issue or event series, add a sentence or two summarizing relevant background details. Keep this concise and only include background information that directly aids in understanding the current article.
Next Steps:

Purpose: Summarize any forthcoming actions, decisions, or anticipated outcomes.
Guidelines: Conclude with 1-2 sentences on what can be expected next. This could include future meetings, planned initiatives, expected results, or statements from involved parties.
Constraints and Style Guidelines:
Word Limit: Keep each summary between 150-200 words.
Language: Use clear, accessible language, free from jargon or overly technical terms. The summary should be understandable by readers with diverse backgrounds.
Tone: Maintain an objective and neutral tone, focusing on factual reporting. Avoid personal opinions or unverified information.
Perspective: Use third-person perspective consistently, ensuring that the summary stays professional and factual.
Example Summaries
Example 1:

Title: Scientists discover 80million-year-old 'fully intact' dinosaur eggs from previously-unknown species


The smallest fully intact dinosaur egg ever discovered has been unearthed in a remote area of China.

The egg, which measures just over an inch long, was one of six dinosaur eggs found in a fully preserved nest from the Late Cretaceous period.

The eggs' small size, 'irregular egg arrangement, worm-like and nodular ornamentation,' and shell thickness signified that they were unlike any other known predatory dinosaur. 

This fossil marks a significant discovery that could shed light on dinosaurs' reproduction habits and diversity.

Researchers at China University of Geosciences recovered six dinosaur eggs at a construction site in Ganzhou in 2021, which is known as one of the 'richest egg fossil sites in the world.'

The team studied the fossils for three years before confirming they were dinosaur eggs and determined they belong to a new species called Minioolithus ganzhouensis. 

Researchers are planning to conduct more studies that they hope will shed light on the type of dinosaur the egg came from and what the species looked like. Researchers reported that they discovered the new egg at a construction site in Meilin, a town located along the perimeter of Ganzhou City. 

A rare, well-preserved nest of six eggs was found at the site and according to Han Fenglu, an associate professor in vertebrate paleontology at CUG. 

Fenglu reported that the eggs were from small theropods - a species of carnivorous and predatory dinosaurs that include the well-known Tyrannosaurus Rex.

Their findings were published in the journal of Historical Biology last week.

The team used scanning electron microscopy (SEM) and electron backscatter diffraction to analyze the shell's shape and type of ornamentation.

This type of scanning technique works by directing an electron beam to examine the surface material of an object and generate high-resolution images.

Election backscatter diffraction then scans the SEM findings and analyzes the material's texture and structure to discern patterns in the shell.

The technology revealed that the fossil's size, thickness, pore system and its ultrastructure was significantly different from other previously discovered egg fossils.Their findings allowed the researchers to classify the fossils as part of the ellipsoidal family of oval-shaped dinosaur eggs.

More research needs to be conducted to uncover how the Minioolithus ganzhouensis species evolved and its strategies for reproduction millions of years ago.

The next phase of research aims to gain 'deeper insights into the life and ecology of these ancient creatures,' according to the Global Times. 

The fossil, named the Ganzhou mini egg, shattered the previous record-holder for smallest egg which measured 1.3 inches in length.

Paleontologists at CUG, the Institute of Vertebrate Paleontology and Paleoanthropology of the Chinese Academy of Sciences' plan to conduct further research on the Ganzhou mini egg using micro-CT scanning which uses X-rays to create a three-dimensional model of the egg's internal structure.They hope this process will help them better understand how the eggs formed and determine the exact type of dinosaur that laid them.

It could also reveal new aspects of dinosaur's biology and the environmental conditions the species experienced during the Late Cretaceous period. 

The researchers said in the study: 'This discovery increases the diversity of dinosaur eggs in the Late Cretaceous and is significant for our understanding of the evolution of theropods in the Late Cretaceous.'


"""

             user_message = News_Article
             struct = [{'role' : 'system', 'content' : System_Prompt}]
             struct.append({"role": "user", "content": user_message})
             chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
             response = chat.choices[0].message.content
             struct.append({"role": "assistant", "content": response})
             st.success("Insight generated successfully!")
             st.subheader("Summary : ")
             st.write(response)

     