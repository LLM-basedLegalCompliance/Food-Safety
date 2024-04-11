import spacy
from spacy.language import Language

"""###Custom Sentencizer Component for Spacy NLP Pipeline"""

@Language.component("custom_sentencizer")
def custom_sentencizer(doc):

    for i, token in enumerate(doc):
        # Define sentence start if pipe + titlecase token
        if token.text == ";" :
            doc[i + 1].is_sent_start = True
        if token.text == "." :
            doc[i + 1].is_sent_start = True

    return doc

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("custom_sentencizer", before="parser")  # Insert before the parser

"""####Sample Web Scraping"""

import requests
from bs4 import BeautifulSoup


# Fetch the content of a URL
url = "https://laws-lois.justice.gc.ca/eng/regulations/SOR-2018-108/page-1.html#docCont"
# response = requests.get(url)
# content = response.content
# text = content.decode("utf-8")


response = requests.get(url)
# Parse the HTML content with BeautifulSoup and extract the text
soup = BeautifulSoup(response.content, "html.parser")
text = soup.get_text()

"""### Scraping and Analyzing Text Data from SFCR Pages"""

#On the whole SFCR 
import requests
from bs4 import BeautifulSoup



# Loop through all the pages of SFCR
for page_num in range(1, 23):  
    # Construct the URL for the current page
    url = f"https://laws-lois.justice.gc.ca/eng/regulations/SOR-2018-108/page-{page_num}.html#docCont"
    
    # Make a request to the URL and get the response content
    response = requests.get(url)
    content = response.content
    
    # Parse the HTML content with BeautifulSoup and extract the text
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text()
    
    # Process the text with spaCy
    doc = nlp(text)
    word_count = len(doc)
    sentence_count = len(list(doc.sents))
    wordcountlist.append(word_count)
    sentencecountlist.append(sentence_count)

print(wordcountlist)
print(sentencecountlist)

"""#### Scraping and Analyzing Text Data from FSRG Pages"""

urllist=["https://inspection.canada.ca/food-guidance-by-commodity/egg-and-processed-egg-products/regulatory-requirements/eng/1521741893401/1521746365151",
         "https://inspection.canada.ca/food-guidance-by-commodity/processed-egg-products/eng/1525871209416/1525871209791",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/canadian-standards-of-identity-volume-2/eng/1520644043482/1520644044699",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/canadian-grade-compendium-volume-5/eng/1520869505643/1520869506282",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/canadian-grade-compendium-volume-9/eng/1520647701525/1520647702274?chap=1#s1c1",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/units-of-measurement-for-the-net-quantity-declarat/eng/1521819171564/1521819242968",
         "https://inspection.canada.ca/food-guidance-by-commodity/egg-and-processed-egg-products/regulatory-requirements/eng/1521741893401/1521746365151#a4",
         "https://inspection.canada.ca/food-guidance-by-commodity/processed-egg-products/eng/1525871209416/1525871209791#a5.0",
         "https://inspection.canada.ca/exporting-food-plants-or-animals/food-exports/food-specific-export-requirements/eng/1503941030653/1503941059640",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/personal-use-exemption/eng/1520439688578/1520439689098",
         "https://inspection.canada.ca/preventive-controls/eggs-and-processes-egg-products/eng/1524259297433/1524259297745",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/ante-mortem-procedures/eng/1519662595076/1519662943354",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/post-mortem-defect-management/eng/1520264787148/1520264821772",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/post-mortem-examination/eng/1520258352736/1520258353148",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/biological-hazards-in-meat-products/eng/1519737053960/1519737054373",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/meat-products-containing-shellfish/eng/1567694073332/1567694165992",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/canadian-standards-of-identity-volume-7/eng/1521204102134/1521204102836",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/canadian-grade-compendium-volume-1/eng/1520878338783/1520878339422",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/canadian-grade-compendium-volume-9/eng/1520647701525/1520647702274?chap=1#s1c1",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/units-of-measurement-for-the-net-quantity-declarat/eng/1521819171564/1521819242968",
         "https://inspection.canada.ca/about-cfia/acts-and-regulations/list-of-acts-and-regulations/documents-incorporated-by-reference/personal-use-exemption/eng/1520439688578/1520439689098",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/standards-for-ante-mortem-examination-and-inspecti/eng/1525719099844/1525719141484",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/standards-for-post-mortem-evaluation-of-food-anima/eng/1526303672372/1526303672637",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/standards-to-identify-a-meat-product-as-edible/eng/1526680768561/1526680768858",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/standards-for-the-management-of-condemned-and-ined/eng/1526650487069/1526650487647",
         "https://inspection.canada.ca/food-licences/inspection-services-for-food-animals-and-meat-prod/eng/1526573260316/1526573260644",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/application-of-the-inspection-legend-on-food-anima/eng/1526310314649/1526310314930",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/humane-treatment/eng/1519849250395/1519849250973",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/ritual-slaughter/eng/1519849364873/1519849365434",
         "https://inspection.canada.ca/food-guidance-by-commodity/meat-products-and-food-animals/humane-stunning-and-slaughter/eng/1519849311784/1519849312189"
         ]
for url in urllist:

  response = requests.get(url)
  # Parse the HTML content with BeautifulSoup and extract the text
  soup = BeautifulSoup(response.content, "html.parser")
  text = soup.get_text()
  doc=nlp(text)
  word_count = len(doc)
  sentence_count = len(list(doc.sents))
  wordcountlist.append(word_count)
  sentencecountlist.append(sentence_count)

print(wordcountlist)
print(sentencecountlist)

"""####Creating Dataframe"""

import pandas as pd
data = {'wordcount': wordcountlist, 'sentencecount': sentencecountlist}
# create a dataframe from the dictionary
df = pd.DataFrame(data)
df.to_csv('df_stat.csv')

"""####Creating plots"""

#plot
import matplotlib.pyplot as plt
import numpy as np

list1 = []
list2 = []
for i in range(len(df.index)):

        list1.append(df.loc[i, 'wordcount'])
        list2.append(df.loc[i, 'sentencecount'])


data = [list1, list2]

ticks = ['word count', 'sentence count']

def Average(lst):
    return sum(lst) / len(lst)
  
avg1 = Average(list1)
avg2 = Average(list2)

means = [avg1,avg2]
print("means",means)

data_plot = plt.boxplot(data, positions=np.array(np.arange(0, 0.7,0.5)), widths=0.3,showmeans= True,vert=False,sym='r+')

for i, line in enumerate(data_plot['means']):
    x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
    text  = ' {:.2f} '.format(means[i])
    plt.annotate(text, xy = (x - 0.01, y - 0.007), fontsize = 13)


def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth = 2)

    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)
    #plt.legend()


# setting colors for each groups
define_box_properties(data_plot, 'blue')

plt.yticks(np.array(np.arange(0, 1,0.5)), ticks, fontsize = 13)
plt.xticks(fontsize = 13)

# set the title

plt.ylabel("Results",fontsize = 13)
plt.xlabel("value", fontsize = 13)
plt.title("Data statistics",fontsize = 13)
plt.savefig('data stat',bbox_inches='tight')
plt.show()

list1 = []
for i in range(len(df['sentencecount'].index)):
    list1.append(df.loc[i, 'sentencecount'])

data = [list1]

# calculate mean
avg = df['sentencecount'].mean()
means = [avg]

# plot boxplot
data_plot = plt.boxplot(data, positions=[0.5], widths=0.1, showmeans=True, vert=False, sym='r+')

# add mean values to the plot
for i, line in enumerate(data_plot['means']):
    x, y = line.get_xydata()[0][0], line.get_xydata()[0][1]
    text = ' {:.2f} '.format(means[i])
    plt.annotate(text, xy=(x - 0.01, y - 0.007), fontsize=15,fontweight='bold')

# set boxplot properties
def define_box_properties(plot_name, color_code):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code, linewidth=2)

    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code)

define_box_properties(data_plot, 'blue')

# set axis labels and title
# plt.xticks(fontsize=15)
# plt.ylabel("Results", fontsize=13)
plt.xlabel("Number of sentences per document",fontsize = 14,fontweight='bold')
plt.ylabel("")
plt.yticks([])
#plt.title("Sentence Count",fontsize = 15,fontweight='bold')

# save and show plot
plt.savefig('sentence_count_boxplot.pdf', bbox_inches='tight')
plt.show()

"""###Summary statistics"""

avg = df['sentencecount'].mean()
avg

std_dev = df['sentencecount'].std()
std_dev

variance = df['sentencecount'].var()
variance

#mean absolute deviation
mad = df['sentencecount'].mad()
mad

# Calculate the median
median = df['sentencecount'].median()

# Calculate the absolute deviations from the median
abs_deviations = np.abs(df['sentencecount'] - median)

# Calculate the median of the absolute deviations
mad = abs_deviations.median()
mad

cv = np.std(df['sentencecount']) / np.mean(df['sentencecount']) * 100
cv