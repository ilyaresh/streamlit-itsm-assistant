# Import os to handle environment variables
import os
# Import sreamlit for the UI
import streamlit as st
import pandas as pd
# Import Azure OpenAI and LangChain
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory



st.title("🦜🔗 ITSM Assistant App")

with st.sidebar:
    st.write("This is an ITSM assistant, tell it what issue you are facing and it will try to help you to raise the correct Service Request, based on organisation knowledge")


def generate_response(input_text):
    from langchain_core.prompts import ChatPromptTemplate
    llm = AzureChatOpenAI(
        openai_api_version="2024-02-15-preview",
        azure_deployment="gpt35t-itsm",
    )
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an experienced ITSM Analyst with knowledge of IT and Application systems in large organisations."),
        ("user", "{input}"),
    ])

    chain = prompt | llm | parser
    c = chain.invoke({"input": "Give the description of the below ITSM tickets, what would be the common theme and what title would you allocate to a new ticket that falls in the same category.\
Reply in a valid JSON format using the following keys: common_theme, title, suggested_fields \
basic internet access. SSO ID of Requester sahom3549 User Name Sparsh Arora Band Business Unit / Department Information Technology Seat Number FF/228 Location Name /office Address 90A Cell Phone No 9582767200 User Name; Sparsh Arora SSO ID : SAHOM3549\
Kindly provide an Exception internet access Divya Jyot Kaur dkms0195 Mobile 09815611146"})
    print(c)


def ask_llm(records):
    from langchain_core.prompts import ChatPromptTemplate
    llm = AzureChatOpenAI(
        openai_api_version="2024-02-15-preview",
        azure_deployment="gpt35t-itsm",
    )
    from langchain_core.output_parsers import JsonOutputParser
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an experienced ITSM Analyst with knowledge of IT and Application systems in large organisations."),
        ("user", "{input}"),
    ])

    chain = prompt | llm | parser
    input_text = "Given the description of the below ITSM tickets, what would be the common theme? what title would you allocate to a new ticket that falls in the same category? what would be some of the input fields that a request will need to provide specific to this type of request?\
The Title should be generic enougn to fit all the ticket\
Reply in a valid JSON format using the following keys: common_theme, title, suggested_fields\
For example {\"common_theme\": \"THEME\", \"title\": \"TITLE\", \"suggested_fields\":\"FIELD1, FIELD2, etc.\"}"
    
    input_text = input_text + "\n" + "\n".join(records['Description']) 
    print(f'INPUT_TEXT {input_text}')

    c = chain.invoke({"input": input_text})
    print(c)
    return c


def get_incidents(input_text):
    from langchain_pinecone import PineconeVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    index_name = 'snow-data'
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    docs = vectorstore.similarity_search_with_score(input_text, k=5)
    
    #st.info(docs[0].page_content)
    #docs_content= []
    #docs_score= []
    #for doc,score in docs:
    #    docs_content.append(doc.page_content)
    #    docs_score.append(score)
    #st.write(docs_content)
    #st.write(docs_score)
    #pd.options.display.max_colwidth = 250
    #df = pd.DataFrame(zip(docs_score,docs_content), columns=['Score', 'Incident'])
    #st.write(df)
    docs_content= []
    docs_score= []
    docs_id= []
    
    for doc,score in docs:
        print(str(int(doc.metadata['tid'])))
        docs_content.append(doc.page_content)
        docs_score.append(score)
        docs_id.append(str(int(doc.metadata['tid'])))
    
    #df = pd.DataFrame(zip(docs_id,docs_score,docs_content), columns=['ID','Score', 'Incident'])
    #st.write(docs_content)
    #st.write(docs_score)
    #st.dataframe(df,hide_index=True)
    
    records = get_details(data,docs_id)
    return records
    
@st.cache_resource
def load_data():
    # Importing the necessary libraries
    import pandas as pd
    pd.options.display.max_seq_items = 2000
    # Importing the csv file
    data = pd.read_csv('GMSCRFDump.csv', encoding = 'ISO-8859-1')
    # shape the data
    data.shape
    # removing duplicates
    ID_mins = data.groupby(['Title', 'Description', "CallClosure Description"]).ID.transform("min")
    data_n = data.loc[data.ID == ID_mins]
    return data_n

def get_details(data, ids):
    ids =  [int(id) for id in ids]
    print(ids)
    return data.loc[data['ID'].isin(ids)]
    


data = load_data()

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")


# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)


# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    #config = {"configurable": {"session_id": "any"}}
    #response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write("Let me think about it.")
    records = get_incidents(prompt)
    with st.chat_message("ai"):
        st.write('I found a few tickets that are similar to your request:')
        st.dataframe(records[['Title','Description']], hide_index=True)
    llm_response  = ask_llm(records)
    suggested_fields = llm_response['suggested_fields'].split(', ')
    suggested_fields = "- " + "\n- ".join(suggested_fields)
 
    nl = "  \n"
    st.chat_message("ai").markdown(f"It looks that in order to help you, you will need to raise a new **\"{llm_response['title']}\"**.{nl}\
When raising this request please provide some of the required information like:{nl}{suggested_fields}")
