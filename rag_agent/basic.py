from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

embedding_function = OpenAIEmbeddings()

docs = [Document(page_content = "Electric vehicles (EVs) offer significant environmental advantages over internal combustion engine cars. By running on electricity rather than gasoline or diesel, EVs produce zero tailpipe emissions, which helps reduce air pollution in urban areas. This contributes to better public health by lowering exposure to harmful pollutants like nitrogen oxides and particulate matter. Furthermore, as more electricity grids incorporate renewable energy sources like wind and solar, the overall carbon footprint of EVs continues to shrink. Compared to conventional vehicles, EVs help slow the pace of climate change and represent a major step toward a more sustainable and cleaner transportation future.Electric vehicles (EVs) significantly reduce environmental impact by producing zero tailpipe emissions. According to the U.S. Department of Energy, a typical EV emits about 4,450 fewer pounds of CO₂ annually compared to a gasoline car. EVs also help improve urban air quality—transportation accounts for over 25% of global greenhouse gas emissions, much of which comes from cars. Moreover, when powered by renewables, EVs can reduce lifecycle emissions by up to 70%. As countries like Norway, where EVs now make up over 80% of new car sales, show, cleaner transportation is not only possible—it’s already happening at scale."

, metadata = {"source" :"about.txt"}),

Document(page_content= "Governments worldwide are pushing EV adoption through strong incentives. In India, subsidies under the FAME II scheme can reduce EV costs by up to ₹1.5 lakh. The U.S. offers federal tax credits of up to $7,500 on EV purchases. EVs also offer long-term savings: charging an EV typically costs less than ₹2 per km, compared to ₹6–8 per km for fuel cars. Maintenance costs are 30–40% lower, thanks to fewer moving parts. The global EV market, valued at $388 billion in 2023, is projected to exceed $950 billion by 2030, showing strong consumer and investor confidence.Despite their benefits, electric vehicles face infrastructure and scalability challenges. One of the biggest hurdles is the availability of fast and convenient charging stations, especially in rural or underdeveloped regions. Charging time is another concern, though it is steadily decreasing with advancements in battery and charger technologies. Grid capacity and sustainable energy sourcing must also keep pace with increased EV demand. However, public and private sectors are heavily investing in charging networks and smart grid systems. As urban planners, utility companies, and car manufacturers collaborate, the future looks promising for a seamless and efficient EV ecosystem worldwide.", metadata = {"source": "EV.txt"})

]

db = Chroma.from_documents(docs, embedding_function)

retriever = db.as_retriever(search_type = 'mmr', search_kwargs = {"k":3})

# print (retriever.invoke("What are leading EV automakers "))

template = '''
Answer the question based on the following context : {context}
Question : {question}
'''

prompt = ChatPromptTemplate.from_template(template=template)

llm = ChatOpenAI()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = ({
    "context" : lambda x: format_docs(retriever.invoke(x)),
    "question": lambda x: x
} | prompt | llm  | StrOutputParser())

print (qa_chain.invoke("What are governments doing worldwide for EV push?"))


